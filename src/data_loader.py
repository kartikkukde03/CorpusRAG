import hashlib
import re
from pathlib import Path
from statistics import median
from typing import Any, List

import pymupdf
from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, JSONLoader, TextLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_core.documents import Document


def _content_type(value: Any) -> str:
    """Normalize legacy prose labels to the canonical text/table split."""
    normalized = str(value or "").strip().lower()
    if normalized == "table":
        return "table"
    if normalized in {"text", "prose", "paragraph", "body"}:
        return "text"
    return "text"


def _document_id(document_name: str) -> str:
    return hashlib.sha1(document_name.encode("utf-8")).hexdigest()[:16]


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _clean_cell(value: Any) -> str:
    """Make a PDF table cell safe to place in a Markdown table."""
    if value is None:
        return ""
    return str(value).replace("\r", " ").replace("\n", " ").replace("|", r"\|").strip()


def _markdown_table(table: Any) -> tuple[str, str]:
    """Convert a PyMuPDF table to row text and a reusable Markdown header."""
    rows = table.extract() or []
    if not rows:
        return "", ""

    width = max((len(row) for row in rows if row), default=0)
    if width == 0:
        return "", ""

    headers = [_clean_cell(value) for value in rows[0]]
    headers.extend([""] * (width - len(headers)))
    headers = [value or f"Column {index + 1}" for index, value in enumerate(headers)]
    header_markdown = (
        "| " + " | ".join(headers) + " |\n"
        + "| " + " | ".join("---" for _ in headers) + " |"
    )
    body_rows = []
    for row in rows[1:]:
        cells = [_clean_cell(value) for value in row]
        cells.extend([""] * (width - len(cells)))
        body_rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(body_rows), header_markdown


def _page_context(page: Any, table: Any) -> str:
    """Capture nearby text before a table as a useful, non-authoritative caption."""
    table_bbox = getattr(table, "bbox", None)
    if not table_bbox:
        return ""
    preceding_blocks = []
    for block in page.get_text("blocks"):
        if len(block) <= 4:
            continue
        text = str(block[4]).strip()
        if text and block[3] <= table_bbox[1]:
            preceding_blocks.append(text)
    return " ".join(preceding_blocks[-3:])[-500:]


def _page_section_title(page: Any, table: Any) -> str:
    """Return the nearest clearly styled heading above a table, if available."""
    table_bbox = getattr(table, "bbox", None)
    if not table_bbox:
        return ""

    text_blocks = [
        block
        for block in page.get_text("dict").get("blocks", [])
        if block.get("type") == 0 and block.get("bbox", (0, 0, 0, 0))[3] <= table_bbox[1]
    ]
    if not text_blocks:
        return ""
    font_sizes = [
        span["size"]
        for block in text_blocks
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if span.get("text", "").strip()
    ]
    if not font_sizes:
        return ""
    body_font_size = median(font_sizes)

    candidates = []
    for block in text_blocks:
        line_parts = []
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = " ".join(span.get("text", "").strip() for span in spans).strip()
            if not text:
                continue
            sizes = [span["size"] for span in spans if span.get("text", "").strip()]
            is_bold = any("bold" in span.get("font", "").lower() for span in spans)
            is_large = bool(sizes and max(sizes) >= body_font_size * 1.15)
            if len(text) <= 160 and (is_bold or is_large):
                line_parts.append(text)
        if line_parts:
            candidates.append((block.get("bbox", (0, 0, 0, 0))[3], " ".join(line_parts)))

    return max(candidates, default=(0, ""))[1]


def _content_metadata(document_name: str, page_number: int, page_count: int, section: str = "") -> dict:
    return {
        "document_id": _document_id(document_name),
        "document_name": document_name,
        "source": document_name,
        "filename": document_name,
        "page_number": page_number,
        "page_count": page_count,
        "section": section or "",
    }


def _clean_pdf_page_text(page: Any, page_number: int) -> str:
    blocks = []
    seen = set()
    for block in page.get_text("blocks"):
        if len(block) <= 4:
            continue
        text = _normalize_whitespace(str(block[4]))
        if not text:
            continue
        low = text.lower()
        if low.startswith("page ") and str(page_number) in low:
            continue
        if low.startswith("figure") or low.startswith("table"):
            continue
        key = low
        if key in seen:
            continue
        seen.add(key)
        blocks.append(text)
    text = "\n\n".join(blocks)
    if len(text) < 30:
        return ""
    return text


def _load_pdf_documents(pdf_path: Path) -> List[Document]:
    """Extract page text and table rows while preserving page and section metadata."""
    documents: List[Document] = []
    with pymupdf.open(str(pdf_path)) as pdf:
        page_count = len(pdf)
        for page_index, page in enumerate(pdf):
            page_number = page_index + 1
            section_title = _page_section_title(page, table=None) if False else ""
            try:
                page_text = _clean_pdf_page_text(page, page_number)
            except Exception as exc:
                print(f"[WARNING] Failed to clean page text for {pdf_path.name}, page {page_number}: {exc}")
                page_text = ""

            base_metadata = _content_metadata(pdf_path.name, page_number, page_count, section_title)
            if page_text:
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={**base_metadata, "content_type": "text"},
                    )
                )

            try:
                tables = page.find_tables().tables
            except Exception as exc:
                print(
                    f"[WARNING] Could not extract tables from "
                    f"{pdf_path.name}, page {page_number}: {exc}"
                )
                continue

            for table_index, table in enumerate(tables, start=1):
                row_text, header_markdown = _markdown_table(table)
                if not header_markdown or not row_text:
                    continue
                nearby_context = _page_context(page, table)
                section_title = _page_section_title(page, table) or ""
                table_title = section_title or f"Table {table_index}"
                table_context_parts = [
                    f"Document: {pdf_path.name}",
                    f"Page: {page_number}",
                ]
                if section_title:
                    table_context_parts.append(f"Section: {section_title}")
                if nearby_context:
                    table_context_parts.append(f"Nearby text: {nearby_context}")
                table_context_parts.append(f"Table: {table_title}")
                table_headers = " | ".join(
                    part.strip() for part in header_markdown.splitlines()[0].strip("|").split("|")
                )
                table_metadata = {
                    **base_metadata,
                    "section": section_title,
                    "content_type": "table",
                    "table_id": f"{_document_id(pdf_path.name)}-p{page_number}-t{table_index}",
                    "table_title": table_title,
                    "table_index": table_index,
                    "table_headers": table_headers,
                    "table_header_markdown": header_markdown,
                    "table_context": "\n".join(table_context_parts),
                }
                documents.append(
                    Document(
                        page_content=row_text,
                        metadata=table_metadata,
                    )
                )

    return documents


def _ensure_document_metadata(document: Any, fallback_name: str, page_number: int | None = None, page_count: int | None = None, section: str = "") -> Any:
    metadata = dict(document.metadata or {})
    if not metadata.get("document_id"):
        metadata["document_id"] = _document_id(fallback_name)
    metadata.setdefault("document_name", fallback_name)
    metadata.setdefault("source", fallback_name)
    metadata.setdefault("filename", fallback_name)
    if page_number is not None:
        metadata.setdefault("page_number", page_number)
    if page_count is not None:
        metadata.setdefault("page_count", page_count)
    metadata.setdefault("section", section)
    metadata["content_type"] = _content_type(metadata.get("content_type", "text"))
    if metadata["content_type"] == "table":
        metadata.setdefault("table_id", f"{metadata['document_id']}-table-{metadata.get('page_number', 1)}")
        metadata.setdefault("table_title", metadata.get("section") or "Table")
    return Document(page_content=document.page_content, metadata=metadata)


def load_all_documents(data_dir: str) -> List[Any]:
    """Load supported files into LangChain documents while retaining page, section, and content-type metadata."""
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents: List[Any] = []

    loaders = {
        ".txt": TextLoader,
        ".csv": CSVLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".docx": Docx2txtLoader,
        ".json": JSONLoader,
    }

    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files.")
    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: {pdf_file}")
        try:
            loaded = _load_pdf_documents(pdf_file)
            print(f"[DEBUG] Loaded {len(loaded)} PDF documents from {pdf_file}")
            documents.extend(loaded)
        except Exception as exc:
            print(f"[ERROR] Failed to load PDF {pdf_file}: {exc}")

    for extension, loader_type in loaders.items():
        files = list(data_path.glob(f"**/*{extension}"))
        print(f"[DEBUG] Found {len(files)} {extension} files.")
        for file_path in files:
            print(f"[DEBUG] Loading {extension} file: {file_path}")
            try:
                loaded = loader_type(str(file_path)).load()
                for document in loaded:
                    document = _ensure_document_metadata(document, file_path.name)
                    document.metadata.setdefault("is_loaded_text", True)
                    if document.metadata.get("content_type") == "table":
                        document.metadata["table_title"] = document.metadata.get("table_title") or file_path.name
                documents.append(document)
            except Exception as exc:
                print(f"[ERROR] Failed to load {extension} file {file_path}: {exc}")

    print(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents


if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)
