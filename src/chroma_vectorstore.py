import hashlib
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embeddings import EmbeddingPipeline

try:
    import chromadb
except ImportError:
    raise ImportError("chromadb is required. Run: pip install chromadb")

EMBEDDING_DIM = 384       # all-MiniLM-L6-v2
CHROMA_DIR    = "data/vector_store"   # persisted on disk
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60


def _normalize_content_type(value: Any) -> str:
    """Canonicalize legacy content labels to the text/table contract used by the app."""
    normalized = str(value or "").strip().lower()
    if normalized == "table":
        return "table"
    if normalized in {"text", "prose", "paragraph", "body", ""}:
        return "text"
    return "text"


class ChromaVectorStore:
    """
    ChromaDB-backed vector store.

    Embeddings are persisted to disk at CHROMA_DIR.
    Each uploaded file gets its own Chroma collection so queries
    are always scoped to the active document.

    Usage
    -----
        store = ChromaVectorStore()
        store.build_from_documents(docs, filename="attention.pdf")
        results = store.query("what is attention?", filename="attention.pdf")
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_DIR,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(embedding_model)

        # Persistent Chroma client — data saved to disk automatically
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.bm25_path = os.path.join(persist_dir, "bm25_index.sqlite3")
        self.reranker_model_name = RERANKER_MODEL
        # Pre-warm reranker at startup so queries don't hit cold-start lag
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.reranker_model_name)
        except Exception:
            self._reranker = None
        self._initialize_bm25_index()
        self._sync_bm25_index()
        print(f"[ChromaVectorStore] Ready. Persisting at: {persist_dir}")

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _collection_name(filename: str) -> str:
        """
        Convert a filename to a valid Chroma collection name.
        Chroma names: 3-63 chars, alphanumeric + hyphens, no leading/trailing hyphens.
        """
        import re
        name = os.path.splitext(filename)[0]          # drop extension
        name = re.sub(r"[^a-zA-Z0-9\-]", "-", name)  # replace illegal chars
        name = re.sub(r"-+", "-", name).strip("-")    # collapse/strip hyphens
        name = name[:63]                               # max length
        if len(name) < 3:
            name = name.ljust(3, "0")                 # min length
        return name.lower()

    def _get_or_create_collection(self, filename: str):
        col_name = self._collection_name(filename)
        return self.client.get_or_create_collection(
            name=col_name,
            metadata={"filename": filename, "hnsw:space": "l2"},
        )

    def _initialize_bm25_index(self) -> None:
        with self._bm25_connection() as connection:
            connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5("
                "collection_id UNINDEXED, chunk_id UNINDEXED, content, "
                "metadata_json UNINDEXED, tokenize='unicode61')"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS bm25_state ("
                "collection_id TEXT PRIMARY KEY, chunk_count INTEGER NOT NULL, "
                "signature TEXT NOT NULL)"
            )

    @contextmanager
    def _bm25_connection(self):
        connection = sqlite3.connect(self.bm25_path, timeout=30)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _collection_signature(ids: List[str], documents: List[str], metadatas: List[dict]) -> str:
        records = sorted(zip(ids, documents, metadatas), key=lambda record: record[0])
        payload = json.dumps(
            records,
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _replace_bm25_collection(
        self,
        collection_id: str,
        ids: List[str],
        documents: List[str],
        metadatas: List[dict],
        signature: str,
    ) -> None:
        rows = [
            (
                collection_id,
                chunk_id,
                content,
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            )
            for chunk_id, content, metadata in zip(ids, documents, metadatas)
        ]
        with self._bm25_connection() as connection:
            connection.execute(
                "DELETE FROM chunk_fts WHERE collection_id = ?",
                (collection_id,),
            )
            connection.executemany(
                "INSERT INTO chunk_fts "
                "(collection_id, chunk_id, content, metadata_json) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
            connection.execute(
                "INSERT OR REPLACE INTO bm25_state "
                "(collection_id, chunk_count, signature) VALUES (?, ?, ?)",
                (collection_id, len(rows), signature),
            )

    @staticmethod
    def _set_collection_signature(collection: Any, signature: str) -> None:
        metadata = dict(collection.metadata or {})
        metadata.pop("hnsw:space", None)
        metadata["bm25_signature"] = signature
        collection.modify(metadata=metadata)

    def _sync_bm25_index(self) -> None:
        """Rebuild missing or outdated BM25 entries from the persistent Chroma index."""
        collections = list(self.client.list_collections())
        active_names = {collection_info.name for collection_info in collections}
        with self._bm25_connection() as connection:
            indexed_names = {
                row[0] for row in connection.execute("SELECT collection_id FROM bm25_state")
            }
            stale_names = indexed_names - active_names
            for collection_id in stale_names:
                connection.execute(
                    "DELETE FROM chunk_fts WHERE collection_id = ?",
                    (collection_id,),
                )
                connection.execute(
                    "DELETE FROM bm25_state WHERE collection_id = ?",
                    (collection_id,),
                )

        for collection_info in collections:
            collection = self.client.get_collection(collection_info.name)
            count = collection.count()
            current_signature = (collection.metadata or {}).get("bm25_signature", "")
            with self._bm25_connection() as connection:
                state = connection.execute(
                    "SELECT chunk_count, signature FROM bm25_state WHERE collection_id = ?",
                    (collection.name,),
                ).fetchone()
            if count == 0:
                self._replace_bm25_collection(
                    collection.name,
                    [],
                    [],
                    [],
                    current_signature,
                )
                continue
            if state and state == (count, current_signature) and current_signature:
                continue

            stored = collection.get(include=["documents", "metadatas"])
            ids = stored["ids"]
            documents = stored["documents"]
            metadatas = stored["metadatas"]
            signature = self._collection_signature(ids, documents, metadatas)
            self._replace_bm25_collection(
                collection.name,
                ids,
                documents,
                metadatas,
                signature,
            )
            self._set_collection_signature(collection, signature)

    # ── Ingest ─────────────────────────────────────────────────────────────────

    def build_from_documents(self, documents: List[Any], filename: str = "unknown") -> int:
        """
        Chunk, embed, and persist documents under a per-filename collection.

        Re-uploading the same file deletes the old collection first (clean upsert).

        Returns
        -------
        int
            Number of chunks stored.
        """
        print(f"[ChromaVectorStore] Building index for '{filename}'...")

        # Chunk + embed
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model_name,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            model=self.model,
        )
        chunks = emb_pipe.chunk_documents(documents)
        if not chunks:
            raise ValueError(f"No indexable text was extracted from '{filename}'.")
        embeddings = emb_pipe.embed_chunks(chunks)
        chunk_ids = [f"chunk-{i}" for i in range(len(chunks))]
        chunk_documents = [chunk.page_content for chunk in chunks]
        chunk_metadatas = [
            {
                **{
                    key: _normalize_content_type(value) if key == "content_type" else value
                    for key, value in chunk.metadata.items()
                    if isinstance(value, (str, int, float, bool))
                },
                "filename": filename,
                "chunk_index": i,
            }
            for i, chunk in enumerate(chunks)
        ]
        signature = self._collection_signature(
            chunk_ids,
            chunk_documents,
            chunk_metadatas,
        )

        # Keep the old index intact unless extraction, chunking and embedding succeeded.
        self.delete_by_filename(filename)

        # Store in Chroma
        collection = self._get_or_create_collection(filename)
        collection.add(
            ids=chunk_ids,
            embeddings=embeddings.tolist(),
            documents=chunk_documents,
            metadatas=chunk_metadatas,
        )
        self._set_collection_signature(collection, signature)
        self._replace_bm25_collection(
            collection.name,
            chunk_ids,
            chunk_documents,
            chunk_metadatas,
            signature,
        )

        print(f"[ChromaVectorStore] Stored {len(chunks)} chunks for '{filename}'.")
        return len(chunks)

    def delete_by_filename(self, filename: str):
        """Delete the collection for this filename if it exists."""
        col_name = self._collection_name(filename)
        existing_names = {collection.name for collection in self.client.list_collections()}
        if col_name in existing_names:
            self.client.delete_collection(col_name)
            print(f"[ChromaVectorStore] Deleted old collection for '{filename}'.")
        with self._bm25_connection() as connection:
            connection.execute(
                "DELETE FROM chunk_fts WHERE collection_id = ?",
                (col_name,),
            )
            connection.execute(
                "DELETE FROM bm25_state WHERE collection_id = ?",
                (col_name,),
            )

    # ── Search ─────────────────────────────────────────────────────────────────

    def query(self, query_text: str, filename: str, top_k: int = 5) -> List[dict]:
        """
        Return hybrid-search and reranked chunks from one document.
        """
        print(f"[ChromaVectorStore] Querying '{filename}' for: '{query_text}'")
        return self._hybrid_query(query_text, top_k, filename=filename)

    def query_all(self, query_text: str, top_k: int = 5) -> List[dict]:
        """Return hybrid-search and reranked chunks across all documents."""
        print(f"[ChromaVectorStore] Querying all stored files for: '{query_text}'")
        return self._hybrid_query(query_text, top_k)

    def query_by_pages(self, filename: str, start_page: int, end_page: int, max_chunks: int = 15) -> List[dict]:
        """Retrieve sequential chunks within a specific page range (e.g. for TOC/Overview)."""
        collections = self._collections_in_scope(filename)
        if not collections:
            return []
        collection = collections[0]
        results = collection.get(
            include=["documents", "metadatas"],
        )
        matched = []
        for content, meta in zip(results.get("documents", []), results.get("metadatas", [])):
            page = meta.get("page_number", 0)
            if start_page <= page <= end_page:
                matched.append({**dict(meta), "content": content})
        matched.sort(key=lambda x: (x.get("page_number", 0), x.get("chunk_index", 0)))
        return matched[:max_chunks]

    def query_by_section(self, filename: str, section_keyword: str, max_chunks: int = 15) -> List[dict]:
        """Retrieve chunks that belong to a specific section or chapter keyword."""
        collections = self._collections_in_scope(filename)
        if not collections:
            return []
        collection = collections[0]
        results = collection.get(
            include=["documents", "metadatas"],
        )
        matched = []
        keyword_clean = section_keyword.casefold()
        for content, meta in zip(results.get("documents", []), results.get("metadatas", [])):
            sec = str(meta.get("section", "")).casefold()
            content_lower = content.casefold()
            if keyword_clean in sec or keyword_clean in content_lower[:150]:
                matched.append({**dict(meta), "content": content})
        matched.sort(key=lambda x: (x.get("page_number", 0), x.get("chunk_index", 0)))
        return matched[:max_chunks]

    def _collections_in_scope(self, filename: str | None) -> List[Any]:
        target_name = self._collection_name(filename) if filename else None
        collections = []
        try:
            available_collections = self.client.list_collections()
        except Exception as e:
            print(f"[ChromaVectorStore] Error listing collections: {e}")
            return []

        for collection_info in available_collections:
            if target_name and collection_info.name != target_name:
                continue
            try:
                collection = self.client.get_collection(collection_info.name)
                if collection.count() > 0:
                    collections.append(collection)
            except Exception as e:
                print(f"[ChromaVectorStore] Warning: Skipping unreadable collection '{collection_info.name}': {e}")
                continue
        return collections

    def _bm25_candidates(
        self,
        query_text: str,
        candidate_k: int,
        collection_id: str | None,
    ) -> List[dict]:
        stop_words = {
            "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or",
            "is", "are", "be", "been", "by", "as", "with", "can", "what", "which",
            "how", "under", "must", "if", "that", "this", "from", "it", "its",
            "according", "does", "do", "shall", "may"
        }
        all_terms = list(dict.fromkeys(re.findall(r"\w+", query_text, flags=re.UNICODE)))
        terms = [t for t in all_terms if t.lower() not in stop_words]
        if not terms:
            terms = all_terms
        if not terms:
            return []

        match_expression = " OR ".join(f'"{term}"' for term in terms)
        sql = (
            "SELECT collection_id, chunk_id, content, metadata_json "
            "FROM chunk_fts WHERE chunk_fts MATCH ?"
        )
        parameters: List[Any] = [match_expression]
        if collection_id:
            sql += " AND collection_id = ?"
            parameters.append(collection_id)
        sql += " ORDER BY bm25(chunk_fts) LIMIT ?"
        parameters.append(candidate_k)

        with self._bm25_connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        results = []
        for collection, chunk_id, content, metadata_json in rows:
            metadata = json.loads(metadata_json)
            results.append(
                {
                    **metadata,
                    "content": content,
                    "_result_id": f"{collection}:{chunk_id}",
                    "_collection_id": collection,
                }
            )
        return results

    def _rerank(self, query_text: str, candidates: List[dict]) -> List[dict]:
        if not candidates:
            return []
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.reranker_model_name)

        pairs = [(query_text, candidate["content"]) for candidate in candidates]
        scores = self._reranker.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            zip(candidates, scores),
            key=lambda result: float(result[1]),
            reverse=True,
        )
        return [
            {
                **{key: value for key, value in candidate.items() if not key.startswith("_")},
                "rerank_score": float(score),
            }
            for candidate, score in ranked
        ]

    def _hybrid_query(
        self,
        query_text: str,
        top_k: int,
        filename: str | None = None,
    ) -> List[dict]:
        if top_k <= 0:
            return []

        collections = self._collections_in_scope(filename)
        total_chunks = sum(collection.count() for collection in collections)
        if total_chunks == 0:
            return []

        candidate_k = min(total_chunks, max(top_k * 4, 30))
        query_embedding = self.model.encode([query_text]).tolist()
        dense_candidates = []
        for collection in collections:
            try:
                count = collection.count()
                if count == 0:
                    continue
                results = collection.query(
                    query_embeddings=query_embedding,
                    n_results=min(candidate_k, count),
                    include=["documents", "distances", "metadatas"],
                )
                collection_filename = (collection.metadata or {}).get(
                    "filename",
                    collection.name,
                )
                for chunk_id, content, distance, metadata in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["distances"][0],
                    results["metadatas"][0],
                ):
                    normalized_metadata = dict(metadata or {})
                    if "content_type" in normalized_metadata:
                        normalized_metadata["content_type"] = _normalize_content_type(
                            normalized_metadata["content_type"]
                        )
                    dense_candidates.append(
                        {
                            **normalized_metadata,
                            "content": content,
                            "distance": distance,
                            "filename": collection_filename,
                            "_result_id": f"{collection.name}:{chunk_id}",
                            "_collection_id": collection.name,
                        }
                    )
            except Exception as e:
                print(f"[ChromaVectorStore] Warning: Could not query collection '{collection.name}': {e}")
                continue
        dense_candidates.sort(key=lambda result: result["distance"])
        dense_candidates = dense_candidates[:candidate_k]

        collection_id = self._collection_name(filename) if filename else None
        sparse_candidates = self._bm25_candidates(
            query_text,
            candidate_k,
            collection_id,
        )

        fused = {}
        for candidates in (dense_candidates, sparse_candidates):
            for rank, candidate in enumerate(candidates, start=1):
                result_id = candidate["_result_id"]
                if result_id not in fused:
                    fused[result_id] = {**candidate, "hybrid_score": 0.0}
                fused[result_id]["hybrid_score"] += 0.5 / (RRF_K + rank)

        fusion_ranked = sorted(
            fused.values(),
            key=lambda result: result["hybrid_score"],
            reverse=True,
        )
        reranked = self._rerank(query_text, fusion_ranked)
        return self._expand_neighbor_chunks(reranked, top_k)

    def _expand_neighbor_chunks(self, candidates: List[dict], max_results: int) -> List[dict]:
        """
        Preserve all top_k reranked candidates, and augment with contiguous sibling chunks
        on the same page so tables, sub-clauses, and headings are never fragmented.
        """
        if not candidates:
            return []

        primary_candidates = candidates[:max_results]
        seen_keys = {(c.get("filename"), c.get("chunk_index")) for c in primary_candidates}
        supplemental = []

        for candidate in primary_candidates:
            chunk_idx = candidate.get("chunk_index")
            filename = candidate.get("filename")
            page_num = candidate.get("page_number")
            col_id = candidate.get("_collection_id") or (self._collection_name(filename) if filename else None)

            if chunk_idx is None or not col_id or page_num is None:
                continue

            try:
                col = self.client.get_collection(col_id)
                sibling_ids = []
                if chunk_idx > 0:
                    sibling_ids.append((chunk_idx - 1, f"chunk-{chunk_idx - 1}"))
                sibling_ids.append((chunk_idx + 1, f"chunk-{chunk_idx + 1}"))

                lookup_ids = [sid for _, sid in sibling_ids]
                s_res = col.get(ids=lookup_ids, include=["documents", "metadatas"])
                for sid, doc, meta in zip(s_res.get("ids", []), s_res.get("documents", []), s_res.get("metadatas", [])):
                    s_idx = meta.get("chunk_index")
                    s_page = meta.get("page_number")
                    s_key = (filename, s_idx)
                    # Sibling must be on the EXACT same page to complete clauses/tables
                    if s_key not in seen_keys and s_page == page_num:
                        seen_keys.add(s_key)
                        normalized_meta = dict(meta or {})
                        if "content_type" in normalized_meta:
                            normalized_meta["content_type"] = _normalize_content_type(normalized_meta["content_type"])
                        supplemental.append({
                            **normalized_meta,
                            "content": doc,
                            "filename": filename or col.name,
                            "rerank_score": candidate.get("rerank_score", 0.0),
                            "_result_id": f"{col_id}:{sid}",
                            "_collection_id": col_id,
                        })
            except Exception:
                continue

        return primary_candidates + supplemental[:3]

    # ── Utility ────────────────────────────────────────────────────────────────

    def list_files(self) -> List[str]:
        """Return display names of all stored documents (non-empty collections only)."""
        files = []
        try:
            collections = self.client.list_collections()
            for col in collections:
                # In ChromaDB 1.x, metadata is not populated in list_collections().
                # We must fetch the full collection object to get metadata.
                try:
                    full_col = self.client.get_collection(col.name)
                    # Skip empty/stale collections
                    if full_col.count() == 0:
                        continue
                    meta = full_col.metadata or {}
                    filename = meta.get("filename", col.name)
                    files.append(filename)
                except Exception as inner_e:
                    print(f"[ChromaVectorStore] Skipping problematic collection '{col.name}': {inner_e}")
                    continue
        except Exception as e:
            print(f"[ChromaVectorStore] Error listing files: {e}")
        return sorted(files)

    def chunk_count(self, filename: str) -> int:
        """Return number of chunks stored for a given filename."""
        try:
            col = self._get_or_create_collection(filename)
            return col.count()
        except Exception:
            return 0
