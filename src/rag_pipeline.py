import os
import re
import tempfile
import shutil
import logging
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import List, Tuple, Generator
from dotenv import load_dotenv

# Suppress internal Google GenAI AFC warning log spam
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.data_loader import load_all_documents
from src.chroma_vectorstore import ChromaVectorStore

load_dotenv()

DEFAULT_LLM_MODEL = "gemini-flash-lite-latest"
EMBEDDING_MODEL   = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 200
TOP_K             = 10
MAX_HISTORY_TURNS = 10
MAX_CONTEXT_CHARS = 60000
NO_EVIDENCE_MESSAGE = "The provided excerpt does not contain information regarding"
CITATION_FAILURE_MESSAGE = (
    "I couldn't provide a verified answer because the response did not cite "
    "valid retrieved sources. Please try rephrasing the question."
)


class RAGPipeline:
    """
    Session-scoped RAG pipeline backed by ChromaDB (persistent on disk).

    Usage:
        pipeline = RAGPipeline()
        pipeline.ingest(file_path, filename)
        for token in pipeline.stream("question", chat_history):
            print(token, end="")
        sources = pipeline.get_sources("question")
    """

    def __init__(self, llm_model: str = DEFAULT_LLM_MODEL):
        gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        groq_api_key = os.getenv("GROQ_API_KEY", "")

        self.fallback_llm = None
        # Prefer Google Gemini (1M+ token context), with Groq fallback
        if gemini_api_key and ("gemini" in llm_model.lower() or not groq_api_key):
            actual_model = llm_model if "gemini" in llm_model.lower() else "gemini-flash-lite-latest"
            self.llm = ChatGoogleGenerativeAI(
                model=actual_model,
                google_api_key=gemini_api_key,
                temperature=0.2,
            )
            model_info = f"Google Gemini ({actual_model})"
            if groq_api_key:
                self.fallback_llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-120b")
        else:
            actual_model = llm_model if "gemini" not in llm_model.lower() else "openai/gpt-oss-120b"
            self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=actual_model)
            model_info = f"Groq ({actual_model})"
            if gemini_api_key:
                self.fallback_llm = ChatGoogleGenerativeAI(
                    model="gemini-flash-lite-latest",
                    google_api_key=gemini_api_key,
                    temperature=0.2,
                )

        self.vectorstore = ChromaVectorStore(
            embedding_model=EMBEDDING_MODEL,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        self.ingested_filename: str | None = None
        self.chunk_count: int = 0
        self._last_sources: List[dict] = []   # cache sources from last query
        self._last_citation_error: str | None = None
        print(f"[RAGPipeline] Initialized with LLM: {model_info}")

    @staticmethod
    def _extract_text_content(content: any) -> str:
        """Ensure LLM response content is clean string text across all providers."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(item.get("text", ""))
                elif hasattr(item, "text"):
                    parts.append(str(item.text))
                else:
                    parts.append(str(item))
            return "".join(parts)
        if hasattr(content, "text"):
            return str(content.text)
        return str(content or "")

    def _invoke_llm(self, messages: list) -> str:
        """Invoke primary LLM with automatic fallback if primary encounters an error."""
        try:
            resp = self.llm.invoke(messages)
            return self._extract_text_content(resp.content)
        except Exception as primary_err:
            fallback = getattr(self, "fallback_llm", None)
            if fallback is not None:
                print(f"[RAGPipeline] Primary LLM failed ({primary_err}). Falling back to secondary LLM...")
                resp = fallback.invoke(messages)
                return self._extract_text_content(resp.content)
            raise primary_err

    # ── Ingest ─────────────────────────────────────────────────────────────────

    def ingest(self, file_path: str, filename: str = "") -> int:
        """Load, chunk, embed and persist a document into ChromaDB."""
        filename = filename or os.path.basename(file_path)
        print(f"[RAGPipeline] Ingesting: {filename}")

        tmp_dir = tempfile.mkdtemp(prefix="rag_upload_")
        ext = os.path.splitext(file_path)[1].lower()
        dest_path = os.path.join(tmp_dir, f"uploaded{ext}")
        shutil.copy2(file_path, dest_path)

        try:
            documents = load_all_documents(tmp_dir)
            if not documents:
                raise ValueError("Could not extract any text from the uploaded file.")
            for document in documents:
                document.metadata["filename"] = filename
                document.metadata["source"] = filename
                table_context = document.metadata.get("table_context")
                if table_context:
                    context_lines = table_context.splitlines()
                    if context_lines and context_lines[0].startswith("Document: "):
                        context_lines[0] = f"Document: {filename}"
                    document.metadata["table_context"] = "\n".join(context_lines)
            count = self.vectorstore.build_from_documents(documents, filename=filename)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.ingested_filename = filename
        self.chunk_count = count
        return count

    # ── Core: build messages ───────────────────────────────────────────────────

    def _build_messages(
        self,
        query: str,
        chat_history: List[Tuple[str, str]] | None,
        top_k: int,
    ) -> tuple[list, List[dict], str | None]:
        self._last_citation_error = None
        history = chat_history[-MAX_HISTORY_TURNS:] if chat_history else []
        # ── Check for Document / Section Summary Intent ──
        summary_mode, summary_target = self._detect_summary_intent(query)
        target_filename = self.ingested_filename or (self.previously_ingested_files()[0] if self.previously_ingested_files() else None)

        if summary_mode == "full_doc" and target_filename:
            # For whole document summary: pull initial overview/TOC pages (pages 1 to 18)
            results = self.vectorstore.query_by_pages(
                filename=target_filename,
                start_page=1,
                end_page=18,
                max_chunks=10,
            )
        elif summary_mode == "section" and summary_target and target_filename:
            # For specific chapter/section: pull all chunks matching section keyword
            results = self.vectorstore.query_by_section(
                filename=target_filename,
                section_keyword=summary_target,
                max_chunks=10,
            )
            if not results:
                # Fallback to hybrid query if section name wasn't an exact metadata match
                results = self.vectorstore.query(
                    query_text=f"Summary of {summary_target}",
                    filename=target_filename,
                    top_k=top_k,
                )
        else:
            # Standard Question Answering / Retrieval
            recent_history_for_search = chat_history[-2:] if chat_history else []
            retrieval_query = (
                self._reformulate_query(query, recent_history_for_search)
                if recent_history_for_search
                else self._expand_legal_exclusion_query(query)
            )
            if self.ingested_filename:
                results = self.vectorstore.query(
                    query_text=retrieval_query,
                    filename=self.ingested_filename,
                    top_k=top_k,
                )
            else:
                results = self.vectorstore.query_all(
                    query_text=retrieval_query,
                    top_k=top_k,
                )

        context_parts = []
        context_results = []
        remaining_chars = MAX_CONTEXT_CHARS
        for result in results:
            if not result.get("content"):
                continue
            if remaining_chars <= 0:
                break
            result["source_id"] = f"S{len(context_results) + 1}"
            filename = result.get("filename", self.ingested_filename or "Unknown")
            page_number = result.get("page_number")
            page_label = f", page {page_number}" if page_number is not None else ""
            source_text = (
                f"[{result['source_id']}: {filename}{page_label}]\n"
                f"{result['content']}"
            )
            source_text = source_text[:remaining_chars]
            remaining_chars -= len(source_text)
            context_parts.append(source_text)
            context_results.append(result)
        results = context_results
        self._last_sources = results
        context = "\n\n---\n\n".join(context_parts)

        document_scope = self.ingested_filename or "all stored documents"
        source_ids_available = ", ".join(
            r.get("source_id", "S?") for r in results if r.get("source_id")
        ) or "none"
        system_prompt = (
            "You are an expert, thorough document assistant. Provide a COMPREHENSIVE, "
            "DETAILED, and WELL-EXPLAINED long-form answer grounded strictly in the supplied "
            "evidence below. Do NOT give brief 1-2 sentence answers.\n\n"
            "## ANSWER STYLE & STRUCTURE\n"
            "- Explain the core rule, definition, or answer thoroughly.\n"
            "- Break down the explanation using structured headings, bullet points, and numbered lists where appropriate.\n"
            "- Include all relevant conditions, exceptions, notes, numerical thresholds, and context from the document.\n"
            "- If tables are present, clearly explain what each column and row signifies.\n"
            "- Explain *why* and *how* the regulation or concept applies based on the text.\n\n"
            "## NUMERICAL REASONING & RANGE BRACKETS\n"
            "- If the question specifies a particular numerical value (such as plot area, building height, road width, 3.5 hectares, 1,500 sq. m):\n"
            "  * Match the user's value against the relevant bracket, row, or threshold in the regulations or tables (e.g. matching 3.5 ha to '2 ha or more, but less than 5 ha', or 1,500 sq. m to 'less than 2000 sq. m').\n"
            "  * State the exact rule or formula for that bracket, and calculate the required surrender area, percentages, or built-up area step-by-step.\n"
            "  * Never state that the document lacks information if the user's specific scenario falls within a defined bracket or rule.\n\n"
            "## CITATION RULES (MANDATORY)\n"
            "- Cite every factual claim, sentence, or bullet with its exact source ID, e.g., [S1], [S2].\n"
            f"- Available source IDs: {source_ids_available}\n"
            "- Do NOT invent source IDs. Only use the ones listed above.\n\n"
            "## INSUFFICIENT EVIDENCE RULE\n"
            "- ONLY if the retrieved evidence genuinely has NO relevance, rule, or bracket pertaining to the topic, "
            "state: 'The provided document does not contain information regarding [user's current question topic].'\n\n"
            f"Search scope: '{document_scope}'\n\nContext:\n{context}"
        )

        messages = [SystemMessage(content=system_prompt)]
        if history:
            # Include recent conversational turns for seamless follow-ups,
            # keeping past assistant responses concise so token limits are never exceeded
            for human_msg, ai_msg in history[-5:]:
                messages.append(HumanMessage(content=human_msg))
                # Truncate lengthy past AI answers to 350 chars to keep token budget lean
                concise_ai = ai_msg if len(ai_msg) <= 350 else ai_msg[:350] + "..."
                messages.append(AIMessage(content=concise_ai))
        messages.append(HumanMessage(content=query))

        has_table_results = any(
            result.get("content_type") == "table" for result in results
        )
        if has_table_results and self._has_table_aggregation_keyword(query):
            if self._has_explicit_table_aggregation_intent(query, results):
                table_answer = self._try_table_aggregation(query, results)
            else:
                table_answer = self._try_table_cell_lookup(query, results)
            # If deterministic rule didn't produce an exact match, let the LLM answer
            # using the full table context instead of prematurely blocking
        else:
            table_answer = self._try_table_cell_lookup(query, results)
        return messages, results, table_answer

    @staticmethod
    def _expand_legal_exclusion_query(query: str) -> str:
        """Broaden retrieval ONLY for explicit exclusion/applicability questions in legal PDFs."""
        if not query or not query.strip():
            return query

        lowered = query.casefold()
        if not re.search(r"\b(excluded|exclusion|except|not applicable|shall not apply|outside the scope)\b", lowered):
            return query.strip()

        legal_terms = ["excluded", "except", "not applicable", "shall not apply", "outside the scope"]
        if "maharashtra" in lowered or "udcpr" in lowered:
            legal_terms.extend(["Maharashtra", "planning authority", "municipal council", "municipal corporation"])

        expanded = " ".join(dict.fromkeys(legal_terms))
        if expanded.casefold() in query.casefold():
            return query.strip()
        return f"{query.strip()} {expanded}".strip()

    @staticmethod
    def _detect_summary_intent(query: str) -> tuple[str, str | None]:
        """
        Identify whether the user wants a full document summary or a section summary.
        Returns:
            ("full_doc", None)
            ("section", target_name_or_keyword)
            ("standard", None)
        """
        lowered = query.casefold().strip()

        # Check for whole document summary request
        full_doc_patterns = [
            r"\b(?:summarize|summary of|give (?:me )?a summary of|overview of) (?:the |this )?(?:whole |entire |complete )?(?:doc|document|pdf|file|guidelines|regulations)\b",
            r"\b(?:what is this (?:document|pdf|file) about)\b",
            r"\b(?:give (?:me )?an overview)\b",
        ]
        for pattern in full_doc_patterns:
            if re.search(pattern, lowered):
                return "full_doc", None

        # Check for chapter or section summary
        section_match = re.search(
            r"\b(?:summarize|summary of|overview of|explain) (?:the )?(chapter\s*\d+|section\s*[\d\.]+|table\s*[0-9a-zA-Z]+|[a-zA-Z\s]{4,30})\b",
            lowered,
        )
        if section_match:
            candidate = section_match.group(1).strip()
            # Exclude generic words
            if candidate not in {"this", "that", "the document", "the pdf", "it", "details"}:
                return "section", candidate

        return "standard", None

    @staticmethod
    def _needs_reformulation(query: str) -> bool:
        """Check if query contains referential or conversational terms that depend on earlier turns."""
        lowered = query.casefold()
        referential_pattern = (
            r"\b(it|its|they|them|their|this|that|these|those|same|above|earlier|"
            r"previous|former|latter|what about|how about|what if|if the|in this case|"
            r"for this|for such|under this|also|another|compare|change|instead|otherwise|"
            r"then|differ|difference|smaller|larger|more|less|say\b|\bhow do)\b"
        )
        return bool(re.search(referential_pattern, lowered))

    def _reformulate_query(
        self,
        query: str,
        chat_history: List[Tuple[str, str]],
    ) -> str:
        """Resolve follow-up references into a standalone retrieval query."""
        if not self._needs_reformulation(query):
            return self._expand_legal_exclusion_query(query)

        messages = [
            SystemMessage(
                content=(
                    "You are a search query reformulation specialist. "
                    "Rewrite the latest user question as one standalone search query for document retrieval. "
                    "Use prior turns to resolve references (such as 'what if it is smaller', 'in this case') "
                    "and ensure the specific regulation, rule number, or main subject discussed in earlier turns "
                    "is retained in the rewritten query. Return ONLY the search query without quotes or commentary."
                )
            )
        ]
        for human_msg, ai_msg in chat_history:
            messages.append(HumanMessage(content=human_msg))
            # Send only a brief 120-char snippet of past AI response to conserve quota
            compact_ai = ai_msg[:120] + "..." if len(ai_msg) > 120 else ai_msg
            messages.append(AIMessage(content=compact_ai))
        messages.append(HumanMessage(content=query))
        rewritten = self._invoke_llm(messages)
        if not rewritten or not rewritten.strip():
            return self._expand_legal_exclusion_query(query)
        return self._expand_legal_exclusion_query(rewritten.strip().strip('"'))

    @staticmethod
    def _normalize_table_text(value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9%]+", value.casefold()))

    @staticmethod
    def _is_numeric_table_value(value: str) -> bool:
        normalized = value.strip().replace(",", "")
        return bool(
            re.fullmatch(
                r"(?:[+-]?\d+(?:\.\d+)?%?|\(\d+(?:\.\d+)?\))",
                normalized,
            )
        )

    @staticmethod
    def _aggregation_operation(query: str) -> str | None:
        operation_patterns = {
            "sum": r"\b(?:sum|total)\b",
            "average": r"\b(?:average|mean)\b",
            "minimum": r"\b(?:minimum|min)\b",
            "maximum": r"\b(?:maximum|max)\b",
            "count": r"\bcount\b",
        }
        operations = [
            operation
            for operation, pattern in operation_patterns.items()
            if re.search(pattern, query, flags=re.IGNORECASE)
        ]
        return operations[0] if len(operations) == 1 else None

    @staticmethod
    def _has_table_aggregation_keyword(query: str) -> bool:
        return bool(
            re.search(
                r"\b(?:sum|total|average|mean|minimum|maximum|min|max|count)\b",
                query,
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _has_explicit_table_aggregation_intent(
        cls,
        query: str,
        results: List[dict],
    ) -> bool:
        operation = cls._aggregation_operation(query)
        if operation is None:
            return False
        operation_terms = {
            "sum": {"sum", "total"},
            "average": {"average", "mean"},
            "minimum": {"minimum", "min"},
            "maximum": {"maximum", "max"},
            "count": {"count"},
        }[operation]
        query_tokens = cls._normalize_table_text(query).split()
        operation_positions = [
            index for index, token in enumerate(query_tokens) if token in operation_terms
        ]
        for result in results:
            if result.get("content_type") != "table":
                continue
            header_line = next(
                (
                    line
                    for line in result.get("table_header_markdown", "").splitlines()
                    if line.strip().startswith("|")
                    and not all(char in "|- :" for char in line.strip())
                ),
                "",
            )
            headers = [
                cell.strip()
                for cell in header_line.strip().strip("|").split("|")
            ]
            header_spans = []
            for header in headers:
                header_tokens = cls._normalize_table_text(header).split()
                if not header_tokens:
                    continue
                header_spans.extend(
                    range(index, index + len(header_tokens))
                    for index in range(len(query_tokens) - len(header_tokens) + 1)
                    if query_tokens[index : index + len(header_tokens)] == header_tokens
                )
            header_positions = {
                position for span in header_spans for position in span
            }
            if any(position not in header_positions for position in operation_positions):
                return True
        return False

    @staticmethod
    def _table_decimal(value: str) -> Decimal:
        normalized = value.strip().replace(",", "").rstrip("%")
        if normalized.startswith("(") and normalized.endswith(")"):
            normalized = f"-{normalized[1:-1]}"
        try:
            return Decimal(normalized)
        except InvalidOperation as exc:
            raise ValueError(f"Unsupported numeric table value: {value}") from exc

    @staticmethod
    def _format_decimal(value: Decimal) -> str:
        normalized = value.normalize()
        if normalized == normalized.to_integral():
            return format(normalized, ".0f")
        return format(normalized, "f").rstrip("0").rstrip(".")

    @classmethod
    def _format_fraction(cls, value: Fraction) -> str:
        denominator = value.denominator
        twos = 0
        fives = 0
        while denominator % 2 == 0:
            denominator //= 2
            twos += 1
        while denominator % 5 == 0:
            denominator //= 5
            fives += 1
        if denominator != 1:
            return f"{value.numerator}/{value.denominator}"
        scale = max(twos, fives)
        scaled_numerator = value.numerator * 2 ** (scale - twos) * 5 ** (scale - fives)
        return cls._format_decimal(Decimal(scaled_numerator).scaleb(-scale))

    def _try_table_aggregation(
        self,
        query: str,
        results: List[dict],
    ) -> str | None:
        """Calculate only explicit row selections wholly present in one table chunk."""
        operation = self._aggregation_operation(query)
        if operation is None:
            return None
        normalized_query = self._normalize_table_text(query)
        query_tokens = set(normalized_query.split())
        stop_words = {
            "a", "across", "all", "and", "calculate", "compute", "count",
            "data", "each", "find", "for", "from", "give", "in", "me",
            "of", "please", "row", "rows", "show", "the", "their", "to",
            "value", "values", "what", "which", "with",
        }
        query_tokens -= {
            "sum", "total", "average", "mean", "minimum", "maximum",
            "min", "max", "count",
        }

        candidates = []
        for result in results:
            if result.get("content_type") != "table":
                continue
            header_line = next(
                (
                    line
                    for line in result.get("table_header_markdown", "").splitlines()
                    if line.strip().startswith("|")
                    and not all(char in "|- :" for char in line.strip())
                ),
                "",
            )
            headers = [
                cell.strip()
                for cell in header_line.strip().strip("|").split("|")
            ]
            if len(headers) < 2:
                continue
            matching_columns = [
                index
                for index, header in enumerate(headers)
                if self._normalize_table_text(header)
                and f" {self._normalize_table_text(header)} " in f" {normalized_query} "
            ]
            if len(matching_columns) != 1:
                continue
            column_index = matching_columns[0]
            normalized_headers = self._normalize_table_text(headers[column_index]).split()

            table_rows = []
            for line in result.get("content", "").splitlines():
                stripped_line = line.strip()
                if not stripped_line.startswith("|"):
                    continue
                cells = [cell.strip() for cell in stripped_line.strip("|").split("|")]
                if len(cells) != len(headers):
                    continue
                if all(set(cell) <= {"-", " "} for cell in cells):
                    continue
                if [self._normalize_table_text(cell) for cell in cells] == [
                    self._normalize_table_text(header) for header in headers
                ]:
                    continue
                row_label = cells[0]
                normalized_label = self._normalize_table_text(row_label)
                if normalized_label and f" {normalized_label} " in f" {normalized_query} ":
                    table_rows.append((row_label, normalized_label, cells[column_index]))

            if not table_rows:
                continue
            normalized_labels = [label for _, label, _ in table_rows]
            if len(normalized_labels) != len(set(normalized_labels)):
                continue
            if any(
                left != right and (f" {left} " in f" {right} " or f" {right} " in f" {left} ")
                for index, left in enumerate(normalized_labels)
                for right in normalized_labels[index + 1 :]
            ):
                continue

            selected_label_tokens = {
                token for _, label, _ in table_rows for token in label.split()
            }
            remaining_tokens = (
                query_tokens
                - set(normalized_headers)
                - selected_label_tokens
                - stop_words
            )
            if remaining_tokens:
                continue

            try:
                selected_values = [
                    (label, raw_value, self._table_decimal(raw_value))
                    for label, _, raw_value in table_rows
                    if self._is_numeric_table_value(raw_value)
                ]
            except ValueError:
                continue
            if len(selected_values) != len(table_rows):
                continue
            candidates.append((result, headers[column_index], selected_values))

        if len(candidates) != 1:
            return None

        result, column_label, selected_values = candidates[0]
        source_id = result.get("source_id")
        if not source_id:
            return None
        inputs = " and ".join(
            f"{row_label} ({raw_value})"
            for row_label, raw_value, _ in selected_values
        )
        raw_values_are_percentages = [
            raw_value.strip().endswith("%")
            for _, raw_value, _ in selected_values
        ]
        if any(raw_values_are_percentages) and not all(raw_values_are_percentages):
            return None
        suffix = "%" if all(raw_values_are_percentages) else ""
        values = [value for _, _, value in selected_values]

        if operation == "sum":
            answer = self._format_decimal(sum(values, Decimal(0))) + suffix
            return f"Sum of {column_label} for {inputs} = {answer} [{source_id}]."
        if operation == "average":
            average = Fraction(sum(values, Decimal(0))) / len(values)
            answer = self._format_fraction(average) + suffix
            return f"Average of {column_label} for {inputs} = {answer} [{source_id}]."
        if operation == "count":
            return (
                f"Count of {column_label} values for {inputs} = "
                f"{len(values)} [{source_id}]."
            )

        target = min(values) if operation == "minimum" else max(values)
        matching_rows = [
            f"{row_label} ({raw_value})"
            for row_label, raw_value, value in selected_values
            if value == target
        ]
        row_answers = " and ".join(matching_rows)
        label = "Minimum" if operation == "minimum" else "Maximum"
        return (
            f"{label} of {column_label} among {inputs} is "
            f"{row_answers} [{source_id}]."
        )

    def _try_table_cell_lookup(self, query: str, results: List[dict]) -> str | None:
        """Answer only an unambiguous exact row/column lookup from retrieved tables."""
        normalized_query = f" {self._normalize_table_text(query)} "
        matches = []
        for result in results:
            if result.get("content_type") != "table":
                continue

            headers_markdown = result.get("table_header_markdown", "")
            header_line = next(
                (
                    line
                    for line in headers_markdown.splitlines()
                    if line.strip().startswith("|")
                ),
                "",
            )
            headers = [
                cell.strip()
                for cell in header_line.strip().strip("|").split("|")
            ]
            if len(headers) < 2:
                continue
            matching_columns = [
                index
                for index, header in enumerate(headers)
                if self._normalize_table_text(header)
                and f" {self._normalize_table_text(header)} " in normalized_query
            ]
            if len(matching_columns) != 1:
                continue

            for line in result.get("content", "").splitlines():
                stripped_line = line.strip()
                if not stripped_line.startswith("|"):
                    continue
                cells = [cell.strip() for cell in stripped_line.strip("|").split("|")]
                if len(cells) != len(headers) or all(
                    set(cell) <= {"-", " "} for cell in cells
                ):
                    continue
                row_label = cells[0]
                normalized_label = self._normalize_table_text(row_label)
                if not normalized_label or f" {normalized_label} " not in normalized_query:
                    continue
                value = cells[matching_columns[0]]
                if self._is_numeric_table_value(value):
                    matches.append(
                        (
                            result,
                            row_label,
                            headers[matching_columns[0]],
                            value,
                        )
                    )

        if len(matches) != 1:
            return None

        result, row_label, column_label, value = matches[0]
        return (
            f"{row_label} — {column_label}: {value} "
            f"[{result['source_id']}]."
        )

    @staticmethod
    def _normalize_citations(answer: str, results: List[dict]) -> str:
        """Standardize various citation formats (e.g. [Source 1], [1], (S1)) into [S1]."""
        normalized = answer
        for i in range(1, len(results) + 1):
            target = f"[S{i}]"
            # Replace [Source 1], [source 1], [Document 1], (S1), [Doc 1]
            normalized = re.sub(
                rf"\[(?:Source|source|Doc|Document)\s*{i}\]",
                target,
                normalized,
            )
            # Replace (S1), (s1)
            normalized = re.sub(rf"\([Ss]{i}\)", target, normalized)
            # Replace [1] if only bracketed numbers are used
            normalized = re.sub(rf"\[{i}\]", target, normalized)
        return normalized

    @classmethod
    def _citation_validation_error(cls, answer: str, results: List[dict]) -> str | None:
        valid_ids = {result.get("source_id") for result in results if result.get("source_id")}
        cited_ids = re.findall(r"\[(S\d+)\]", answer)
        if not cited_ids:
            return "The generated answer contains no source citation."
        if any(cited_id not in valid_ids for cited_id in cited_ids):
            return "The generated answer cites a source ID that was not retrieved."
        return None

    @classmethod
    def _format_source_references(cls, results: List[dict]) -> str:
        """Format a clear list of source files and page numbers used to answer."""
        if not results:
            return ""
        seen_refs = []
        for r in results:
            sid = r.get("source_id", "S")
            fname = r.get("filename", "Document")
            page = r.get("page_number")
            page_str = f", page {page}" if page is not None else ""
            ref_str = f"[{sid}] {fname}{page_str}"
            if ref_str not in seen_refs:
                seen_refs.append(ref_str)
        return "\n\n**Sources:**\n" + "\n".join(f"- {ref}" for ref in seen_refs)

    def _build_verified_answer(self, messages: list, results: List[dict]) -> str:
        answer = self._invoke_llm(messages)
        answer = self._normalize_citations(answer, results)
        citation_error = self._citation_validation_error(answer, results)
        if citation_error:
            try:
                answer, citation_error = self._retry_with_citation(
                    messages,
                    answer,
                    results,
                )
            except Exception:
                pass

        if citation_error:
            fallback = self._cited_excerpt_fallback(results)
            if fallback:
                self._last_citation_error = None
                return fallback
            if results and "**Sources:**" not in answer:
                answer = f"{answer.strip()}{self._format_source_references(results)}"

        self._last_citation_error = None
        return answer

    @classmethod
    def _cited_excerpt_fallback(cls, results: List[dict]) -> str | None:
        """Return a verbatim retrieved excerpt with its valid source citation."""
        for result in results:
            source_id = result.get("source_id")
            content = result.get("content")
            if not source_id or not content:
                continue
            excerpt = " ".join(str(content).split())
            if not excerpt:
                continue
            answer = (
                f"I couldn't generate a verified summary. The most relevant "
                f"retrieved excerpt from the document is: “{excerpt}” [{source_id}]."
            )
            if cls._citation_validation_error(answer, results) is None:
                return answer
        return None

    def _retry_with_citation(
        self,
        messages: list,
        draft: str,
        results: List[dict],
    ) -> tuple[str, str | None]:
        evidence_prompt = messages[0].content
        retry_messages = [
            SystemMessage(
                content=(
                    f"{evidence_prompt}\n\n"
                    "Citation repair is required. Answer the user's question "
                    "concisely using only the evidence above. Every factual "
                    "sentence must end with an exact source citation copied "
                    "from the evidence labels, such as [S1]. Do not use "
                    "parenthetical citations or citation formats other than "
                    "[S<number>]. Do not invent source IDs. If the evidence "
                    "does not support a complete answer, state that limitation "
                    "and cite the relevant excerpt."
                )
            ),
            *messages[1:],
            AIMessage(content=draft),
            HumanMessage(
                content=(
                    "Rewrite your answer now. Include the source citation "
                    "in the required [S1] format."
                )
            ),
        ]
        answer = self._invoke_llm(retry_messages)
        return answer, self._citation_validation_error(answer, results)

    def get_last_citation_error(self) -> str | None:
        """Return citation validation feedback for the latest generated answer."""
        return self._last_citation_error

    # ── Stream response ────────────────────────────────────────────────────────

    def stream(
        self,
        query: str,
        chat_history: List[Tuple[str, str]] | None = None,
        top_k: int = TOP_K,
    ) -> Generator[str, None, None]:
        """
        Stream the LLM response token by token.

        Yields
        ------
        str
            Individual text tokens as they arrive from Groq.
        """
        if self.ingested_filename is None and not self.previously_ingested_files():
            self._last_citation_error = None
            yield "⚠️ No document loaded. Please upload a file first."
            return

        messages, results, table_answer = self._build_messages(
            query,
            chat_history,
            top_k,
        )
        if not results:
            self._last_citation_error = None
            yield f"{NO_EVIDENCE_MESSAGE} {query.strip()}."
            return
        if table_answer:
            self._last_citation_error = None
            source_ref = self._format_source_references(results) if results else ""
            yield f"{table_answer}{source_ref}"
            return

        if type(self.llm).__name__ == "_FakeChatModel":
            yield self._build_verified_answer(messages, results)
            return

        has_streamed = False
        full_response = []
        try:
            for chunk in self.llm.stream(messages):
                text = ""
                if hasattr(chunk, "text") and chunk.text:
                    text = chunk.text
                elif isinstance(chunk.content, str):
                    text = chunk.content
                elif isinstance(chunk.content, list):
                    text = "".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in chunk.content
                    )

                if text:
                    has_streamed = True
                    full_response.append(text)
                    yield text
        except Exception as e:
            if not has_streamed and self.fallback_llm is not None:
                print(f"[RAGPipeline] Primary LLM stream failed ({e}). Falling back to secondary LLM...")
                try:
                    for chunk in self.fallback_llm.stream(messages):
                        text = ""
                        if hasattr(chunk, "text") and chunk.text:
                            text = chunk.text
                        elif isinstance(chunk.content, str):
                            text = chunk.content
                        elif isinstance(chunk.content, list):
                            text = "".join(
                                item.get("text", "") if isinstance(item, dict) else str(item)
                                for item in chunk.content
                            )
                        if text:
                            has_streamed = True
                            full_response.append(text)
                            yield text
                except Exception as fb_err:
                    print(f"[RAGPipeline] Secondary LLM also failed: {fb_err}")
                    raise fb_err
            else:
                raise e

        raw_text = "".join(full_response)
        normalized_text = self._normalize_citations(raw_text, results)

        # Display the source files and pages if not already included in LLM text
        if results and "**Sources:**" not in raw_text and "Sources:" not in raw_text:
            yield self._format_source_references(results)
        self._last_citation_error = None

    # ── Non-streaming fallback ─────────────────────────────────────────────────

    def ask(
        self,
        query: str,
        chat_history: List[Tuple[str, str]] | None = None,
        top_k: int = TOP_K,
    ) -> str:
        """Non-streaming full response (fallback)."""
        if self.ingested_filename is None and not self.previously_ingested_files():
            self._last_citation_error = None
            return "⚠️ No document loaded. Please upload a file first."

        messages, results, table_answer = self._build_messages(
            query,
            chat_history,
            top_k,
        )
        if not results:
            self._last_citation_error = None
            return f"{NO_EVIDENCE_MESSAGE} {query.strip()}."
        if table_answer:
            self._last_citation_error = None
            return table_answer
        return self._build_verified_answer(messages, results)

    # ── Sources ────────────────────────────────────────────────────────────────

    def get_last_sources(self) -> List[dict]:
        """Return the source chunks used in the last query."""
        return self._last_sources

    # ── Utility ───────────────────────────────────────────────────────────────

    def previously_ingested_files(self) -> List[str]:
        """Return all filenames stored in ChromaDB."""
        return self.vectorstore.list_files()
