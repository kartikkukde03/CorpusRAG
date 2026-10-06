# DocChat AI — Complete RAG Architecture & Execution Flow Specification

This document provides an end-to-end technical explanation of the **DocChat AI** Retrieval-Augmented Generation (RAG) system. It details the system architecture, component contracts, lifecycle states, exact execution paths, data structures, and function-call sequences down to the function and parameter level.

---

## 1. System Architecture Overview

DocChat AI is an enterprise-grade document intelligence system designed for complex, heterogeneous PDF/DOCX documents (such as building codes, legal regulations, and engineering standards). It features **hybrid retrieval (Dense Semantic + Sparse BM25)**, **cross-encoder reranking**, **table structure extraction**, **conversational context memory**, **citation verification**, and **real-time token streaming**.

```
                           +--------------------------+
                           |  User Interface (Streamlit)
                           |    streamlit_app.py      |
                           +------------+-------------+
                                        |
                   User Query / Ingest  |  Streamed Tokens & Citations
                                        v
                           +--------------------------+
                           |       RAG Pipeline       |
                           |   src/rag_pipeline.py    |
                           +------------+-------------+
                                        |
       +--------------------------------+--------------------------------+
       |                                                                 |
       v                                                                 v
+-----------------------------+                           +-----------------------------+
|    Ingestion & Chunking     |                           |    Hybrid Vector Store      |
|     src/data_loader.py      |                           |  src/chroma_vectorstore.py  |
|     src/embeddings.py       |                           +--------------+--------------+
+-----------------------------+                                          |
                                                 +-----------------------+-----------------------+
                                                 |                                               |
                                                 v                                               v
                                  +-----------------------------+                 +-----------------------------+
                                  |    ChromaDB (Persistent)    |                 |   SQLite FTS5 (BM25 Index)  |
                                  | Dense Embeddings (384-dim)  |                 |  Sparse Exact Token Search  |
                                  +--------------+--------------+                 +--------------+--------------+
                                                 |                                               |
                                                 +-----------------------+-----------------------+
                                                                         |
                                                                         v
                                                          +-----------------------------+
                                                          |  Cross-Encoder Reranker     |
                                                          | ms-marco-MiniLM-L-6-v2 (CPU)|
                                                          +--------------+--------------+
                                                                         | Top Candidates
                                                                         v
                                                          +-----------------------------+
                                                          |  Groq LLM Engine            |
                                                          |  openai/gpt-oss-120b        |
                                                          +-----------------------------+
```

---

## 2. Core Modules & Component Architecture

### 2.1 UI Layer: `streamlit_app.py`
- **Role:** Web interface, user interaction management, streaming render loop, chat history buffer, and source inspector.
- **Key State Variables (`st.session_state`):**
  - `pipeline`: Instance of `RAGPipeline` (cached across reruns via `@st.cache_resource`).
  - `chat_history`: Array of past user-AI message tuples `[(user_turn, ai_turn), ...]`, capped at 10 turns.
  - `messages`: UI message log containing role, message content, timestamp, and source excerpts.
  - `doc_ready`: Boolean flag indicating if a single active document is selected.
  - `doc_name`: Name of the active file or empty string if searching all stored documents.

### 2.2 Orchestration Engine: `src/rag_pipeline.py`
- **Class:** `RAGPipeline`
- **Role:** Central coordinator handling document ingestion, query reformulation, intent detection, prompt synthesis, deterministic table arithmetic, streaming execution, and citation validation.
- **Configured Constants:**
  - `DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"`
  - `EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"`
  - `TOP_K = 8`
  - `MAX_HISTORY_TURNS = 10`
  - `MAX_CONTEXT_CHARS = 32000`

### 2.3 Storage & Retrieval Engine: `src/chroma_vectorstore.py`
- **Class:** `ChromaVectorStore`
- **Role:** Persistent dual-index datastore combining dense vector storage (ChromaDB) with sparse full-text search (SQLite FTS5 BM25) and local Cross-Encoder reranking.
- **Database Paths:**
  - ChromaDB collections: `data/vector_store/`
  - BM25 SQLite database: `data/vector_store/bm25_index.sqlite3`
- **Reranker Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (pre-warmed at initialization).

### 2.4 Document Parsing & Table Extraction: `src/data_loader.py`
- **Functions:** `load_all_documents()`, `_clean_pdf_page_text()`, `_markdown_table()`, `_page_context()`
- **Role:** Uses `pymupdf` (PyMuPDF) to detect paragraphs, section headers, and tabular boundaries, converting visual document grids into Markdown tables with context annotations.

### 2.5 Chunking & Vectorization: `src/embeddings.py`
- **Class:** `EmbeddingPipeline`
- **Role:** Applies adaptive chunking based on document length and content type, appending table headers to divided row blocks, and generating dense vector representations via `SentenceTransformer`.

---

## 3. Execution Lifecycle Phase 1: Document Ingestion

When a user uploads a PDF or DOCX file and clicks **"Process & Index"**, the system executes the following deterministic sequence:

```
[User clicks 'Process & Index']
       |
       v
streamlit_app.py: get_pipeline()
       |
       v
pipeline.ingest(tmp_path, filename)
       |
       +---> data_loader.py: load_all_documents(tmp_dir)
       |        |
       |        +---> pymupdf.open(file)
       |        +---> Extract page text & headers
       |        +---> page.find_tables() -> Convert to Markdown Table
       |        +---> Attach metadata (page_number, section, content_type)
       |
       +---> vectorstore.build_from_documents(documents, filename)
                |
                +---> embeddings.py: EmbeddingPipeline.chunk_documents()
                |        |
                |        +---> Adaptive chunk size (1800 chars for >15 pages, 800 for <=15)
                |        +---> Prepend table header markdown to split table rows
                |
                +---> embeddings.py: EmbeddingPipeline.embed_chunks()
                |        |
                |        +---> SentenceTransformer.encode() -> np.ndarray (384-dim)
                |
                +---> vectorstore.delete_by_filename(filename)  (Clean upsert)
                +---> ChromaDB Collection.add(ids, embeddings, metadatas, documents)
                +---> SQLite FTS5 chunk_fts INSERT (BM25 sparse index sync)
```

### Detailed Ingestion Steps:

1. **`streamlit_app.py:318-340`**:
   - Saves uploaded bytes to a named temporary file (`tempfile.NamedTemporaryFile`).
   - Calls `pipeline.ingest(tmp_path, filename=uploaded_file.name)`.
2. **`RAGPipeline.ingest(file_path, filename)`** (`src/rag_pipeline.py:61`):
   - Creates a temporary working folder and copies the target file.
   - Invokes `load_all_documents(tmp_dir)`.
3. **`data_loader.py:load_all_documents(dir_path)`**:
   - Opens the PDF using `pymupdf.open()`.
   - Iterates through each page:
     - Extracts clean body text via `_clean_pdf_page_text()`.
     - Detects structural tables via `page.find_tables().tables`.
     - Converts table cells into a Markdown table using `_markdown_table(table)`.
     - Formats contextual breadcrumbs: Document name, page number, preceding text, section title.
     - Creates `Document` objects with `metadata["content_type"] = "table"` or `"text"`.
4. **`ChromaVectorStore.build_from_documents(documents, filename)`** (`src/chroma_vectorstore.py:227`):
   - Initializes `EmbeddingPipeline(model=self.model)`.
   - Calls `emb_pipe.chunk_documents(documents)`:
     - Applies dynamic sizing: 1,800 characters with 300 overlap for documents over 15 pages; 800 characters with 150 overlap for smaller documents.
     - For tables, subtracts the prefix header size so total chunk size is not exceeded, then prepends `table_header_markdown` to every sub-chunk.
   - Calls `emb_pipe.embed_chunks(chunks)` to generate 384-dimensional dense vectors.
   - Calls `delete_by_filename(filename)` to clean out any prior version of the collection and BM25 index.
   - Persists dense vectors into the document's Chroma collection.
   - Executes SQL `INSERT` statements into the `chunk_fts` virtual table in `bm25_index.sqlite3` to synchronize sparse search.

---

## 4. Execution Lifecycle Phase 2: Query Processing & Retrieval

When the user enters a question into the chat input, the request passes through intent detection, contextual rewriting, hybrid candidate retrieval, reranking, and prompt packaging.

```
[User submits prompt in chat input]
       |
       v
streamlit_app.py: pipeline.stream(query, chat_history, top_k)
       |
       v
RAGPipeline._build_messages(query, chat_history, top_k)
       |
       +---> 1. Intent Detection: _detect_summary_intent(query)
       |        |
       |        +-- "full_doc" -> vectorstore.query_by_pages(p. 1-18) [TOC/Overview]
       |        +-- "section"  -> vectorstore.query_by_section(keyword)
       |        +-- "standard" -> Proceed to standard search
       |
       +---> 2. Fast-Path Query Reformulation:
       |        |
       |        +-- _needs_reformulation(query) is FALSE -> Use query directly + expansion
       |        +-- _needs_reformulation(query) is TRUE  -> LLM rewrites using chat_history[-2:]
       |
       +---> 3. Hybrid Retrieval: vectorstore.query() / query_all()
       |        |
       |        +---> Dense Search: ChromaDB collection.query() [Cosine/L2 distance]
       |        +---> Sparse Search: SQLite FTS5 chunk_fts MATCH BM25
       |        +---> Reciprocal Rank Fusion (RRF):
       |                 score = sum( 0.5 / (60 + rank) )
       |
       +---> 4. Cross-Encoder Reranker:
       |        |
       |        +---> Take candidate pool: max(top_k + 2, 6)
       |        +---> CrossEncoder.predict([(query, chunk), ...])
       |        +---> Sort descending by rerank score -> Select top_k chunks
       |
       +---> 5. Context Construction:
       |        |
       |        +---> Format: "[S1: Document.pdf, page 64]\n<Chunk Content>"
       |        +---> Pack up to 32,000 characters
       |        +---> Build SystemMessage with Long-Form rules & Citation instructions
       |        +---> Return (messages, results, table_answer)
```

### Detailed Retrieval & Context Assembly Steps:

1. **Summary Intent Detection (`_detect_summary_intent`)**:
   - Inspects the query against regular expressions:
     - Whole-document queries (e.g., *"Summarize this document"*) trigger `query_by_pages(1, 18)`.
     - Chapter/section queries (e.g., *"Summarize Chapter 14"*) trigger `query_by_section("Chapter 14")`.
2. **Selective Reformulation (`_needs_reformulation`)**:
   - Queries without pronouns (e.g., *"What is Table 3A?"*) skip the LLM rewrite step entirely, saving 1.5–2 seconds.
   - Queries containing referential tokens (*"it"*, *"that"*, *"there"*, *"compare"*) trigger an LLM rewrite using only the **last 2 conversation turns** to avoid context bleed from older turns.
3. **Dual-Index Search & Reciprocal Rank Fusion (`_hybrid_query`)**:
   - **Dense vector search:** Encodes the query into an embedding and queries ChromaDB for the closest vector neighbors.
   - **Sparse text search (`_bm25_candidates`):** Queries SQLite FTS5 using an `OR` combination of tokens to match exact numbers, section codes, and acronyms.
   - **Reciprocal Rank Fusion (RRF):** Merges both ranked lists using the formula:
     $$\text{RRF Score} = \sum \frac{0.5}{60 + \text{Rank}}$$
4. **Cross-Encoder Reranker (`_rerank`)**:
   - The top candidates (sized to `max(top_k + 2, 6)`) are evaluated by `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`.
   - Produces fine-grained cross-attention scores and sorts the candidates to extract the final `top_k` results.
5. **Context Token Assembly**:
   - Assigns unique sequential source IDs: `S1`, `S2`, `S3`, etc.
   - Formats chunk blocks: `[S1: filename.pdf, page X]\n<content>`.
   - Injects the blocks into the system prompt along with long-form response guidelines.

---

## 5. Execution Lifecycle Phase 3: Response Generation & Citation Validation

```
[Context messages assembled]
       |
       v
RAGPipeline.stream(query, chat_history, top_k)
       |
       +---> Check: Did deterministic table arithmetic produce an answer?
       |        |
       |        YES -> Yield answer + _format_source_references() -> Exit
       |        NO  -> Proceed to LLM Streaming
       |
       +---> LLM Streaming: ChatGroq.stream(messages)
       |        |
       |        +---> Iterate token by token -> yield chunk.content (TTFT < 400ms)
       |        +---> Accumulate full text in background
       |
       +---> Post-Stream Processing:
       |        |
       |        +---> Normalize citations: [Source 1], [1], (S1) -> [S1]
       |        +---> Check _citation_validation_error()
       |        +---> Always append structured source references:
       |                 **Sources:**
       |                 - [S1] Document.pdf, page 64
       |                 - [S2] Document.pdf, page 65
       |
       v
[Complete response rendered in Streamlit UI]
```

### Detailed Generation Steps:

1. **Deterministic Fast-Path Check**:
   - If the query targets a specific numeric cell or an aggregation (sum/average/min/max) within a single retrieved table, `_try_table_aggregation()` or `_try_table_cell_lookup()` evaluates the mathematical result in Python using `Decimal` arithmetic.
   - If an exact deterministic match is found, it is yielded immediately with its source reference tag.
   - If no exact deterministic match is found, the system falls back to the LLM to generate the answer using full table context.
2. **Real-Time Token Streaming**:
   - Calls `ChatGroq.stream(messages)` using `openai/gpt-oss-120b`.
   - Iterates through token chunks and immediately yields each `chunk.content` to Streamlit's `write_stream()`.
   - Time-to-First-Token (TTFT) is typically **under 400ms**.
3. **Citation Normalization & Source Attachment**:
   - Normalizes alternative citation formats (e.g., `[Source 1]`, `[1]`, `(S1)`) to standard `[S1]` tags.
   - Appends formatted source references at the end of the response:
     ```markdown
     **Sources:**
     - [S1] UDCPR_compressed_2.pdf, page 64
     - [S2] UDCPR_compressed_2.pdf, page 65
     ```
   - If the LLM omitted inline citations, the pipeline appends the retrieved sources rather than withholding the answer.
4. **Chat State Persistence**:
   - Appends the assistant turn to `st.session_state.messages`.
   - Appends `(query, response_text)` to `st.session_state.chat_history`, maintaining a rolling window of the last 10 turns.

---

## 6. Detailed File-by-File Technical Reference

### `src/rag_pipeline.py`
| Function / Method | Signature | Description |
| :--- | :--- | :--- |
| `__init__` | `(llm_model: str)` | Initializes `ChatGroq`, instantiates `ChromaVectorStore`, and configures session state variables. |
| `ingest` | `(file_path: str, filename: str) -> int` | Copies the target document to a temporary directory, parses documents, builds vector and BM25 indices, and returns the chunk count. |
| `_build_messages` | `(query, chat_history, top_k) -> (messages, results, table_ans)` | Coordinates intent detection, query reformulation, hybrid search, context bounding (32k chars), and system prompt construction. |
| `_detect_summary_intent` | `(query: str) -> (str, str \| None)` | Analyzes query phrasing using regex to identify full-document summary requests or chapter/section-specific summary targets. |
| `_needs_reformulation` | `(query: str) -> bool` | Checks for referential terms (*it, that, there, previous, compare*) to decide whether to run LLM query reformulation. |
| `_reformulate_query` | `(query: str, chat_history) -> str` | Prompts the LLM to rewrite a referential follow-up query into a standalone retrieval query using the last 2 history turns. |
| `_expand_legal_exclusion_query`| `(query: str) -> str` | Enriches legal and exclusion-related questions with common statutory terms (*except, not applicable, municipal council*). |
| `_normalize_citations` | `(answer: str, results: list) -> str` | Converts non-standard citations (e.g., `[Source 1]`, `(S1)`, `[1]`) into canonical `[S1]` source tags. |
| `_citation_validation_error` | `(answer: str, results: list) -> str \| None` | Verifies that all cited IDs exist within the retrieved source set. |
| `_format_source_references` | `(results: List[dict]) -> str` | Generates a Markdown list of source files and page numbers used to answer the query. |
| `stream` | `(query, chat_history, top_k) -> Generator[str]` | Main UI entry point. Streams tokens in real time and appends formatted source citations. |
| `ask` | `(query, chat_history, top_k) -> str` | Non-streaming fallback execution method. |

---

### `src/chroma_vectorstore.py`
| Function / Method | Signature | Description |
| :--- | :--- | :--- |
| `__init__` | `(persist_dir, embedding_model, chunk_size, chunk_overlap)` | Sets up the ChromaDB client, opens the SQLite BM25 database, and pre-warms the `CrossEncoder` model. |
| `build_from_documents` | `(documents: List[Any], filename: str) -> int` | Chunks and embeds documents, cleans existing records, and populates both ChromaDB and the SQLite BM25 index. |
| `query` | `(query_text: str, filename: str, top_k: int) -> List[dict]` | Runs hybrid search and cross-encoder reranking scoped to a single document collection. |
| `query_all` | `(query_text: str, top_k: int) -> List[dict]` | Runs hybrid search across all active document collections in the database. |
| `query_by_pages` | `(filename, start_page, end_page, max_chunks) -> List[dict]` | Retrieves chunks in sequential page order within a specific page range (used for document overviews). |
| `query_by_section` | `(filename, section_keyword, max_chunks) -> List[dict]` | Retrieves chunks matching a specific chapter or section title in page order. |
| `_bm25_candidates` | `(query_text, candidate_k, collection_id) -> List[dict]` | Executes SQL full-text search against the `chunk_fts` virtual table using SQLite FTS5. |
| `_rerank` | `(query_text, candidates: List[dict]) -> List[dict]` | Computes cross-attention similarity scores across candidate pairs using `CrossEncoder.predict()` on CPU. |
| `_hybrid_query` | `(query_text, top_k, filename) -> List[dict]` | Blends ChromaDB dense vector results with SQLite BM25 results using Reciprocal Rank Fusion (RRF), then reranks candidates. |

---

### `src/data_loader.py`
| Function / Method | Signature | Description |
| :--- | :--- | :--- |
| `load_all_documents` | `(directory_path: str) -> List[Document]` | Scans a folder for PDF, DOCX, XLSX, and TXT files and routes each to its respective loader. |
| `_load_pdf` | `(pdf_path: Path) -> List[Document]` | Uses `pymupdf` to extract text blocks, detect tables via `.find_tables()`, and build structured document objects. |
| `_markdown_table` | `(table: Any) -> tuple[str, str]` | Transforms a detected PyMuPDF table into an aligned Markdown grid string along with its header row. |
| `_page_context` | `(page: Any, table: Any) -> str` | Extracts up to 500 characters of text immediately preceding a table to capture headings and captions. |

---

### `src/embeddings.py`
| Function / Method | Signature | Description |
| :--- | :--- | :--- |
| `chunk_documents` | `(documents: List[Any]) -> List[Any]` | Splits documents into chunks using `RecursiveCharacterTextSplitter`. Dynamically adjusts chunk size for long documents and prepends headers to table rows. |
| `embed_chunks` | `(chunks: List[Any]) -> np.ndarray` | Generates dense embeddings for all text chunks using `SentenceTransformer.encode()`. |
| `_chunk_settings` | `(document, default_size, default_overlap) -> (int, int)` | Returns `(1800, 300)` for documents with more than 15 pages; returns `(800, 150)` for shorter documents. |

---

## 7. Edge Case Handling & Guardrails

1. **Answer Withholding Prevention:**
   - If the model generates a response without inline citations, the pipeline normalizes common citation variants (`[Source 1]`, `[1]`, `(S1)`) and automatically appends the retrieved sources (`Sources: [S1] [S2] ...`) rather than withholding the answer.
2. **Context Bleed Protection:**
   - Query reformulation uses only the **last 2 conversation turns** (`chat_history[-2:]`), preventing topics discussed several turns earlier from contaminating current searches.
3. **Out-of-Scope Query Guardrails:**
   - The system prompt explicitly instructs the model to declare when information is missing from the retrieved context for the *current* question, preventing it from falling back to topics from earlier chat turns.
4. **Clean Upsert on Document Ingestion:**
   - Re-uploading an existing document automatically purges its prior Chroma collection and deletes its BM25 index rows from SQLite, avoiding duplicate chunks.
5. **Deterministic Arithmetic Precision:**
   - Table cell lookups and aggregations (sum, min, max, average, count) are calculated using Python's `Decimal` and `Fraction` modules to eliminate arithmetic errors from the LLM.

