from typing import List, Any

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src.data_loader import load_all_documents


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model: SentenceTransformer | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model if model is not None else SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    @staticmethod
    def _chunk_settings(document: Any, default_size: int, default_overlap: int) -> tuple[int, int]:
        """Choose PDF chunk settings based on its total page count."""
        page_count = document.metadata.get("page_count")
        if not isinstance(page_count, int):
            return default_size, default_overlap
        if page_count <= 15:
            return 800, 150
        return 1800, 300

    @staticmethod
    def _table_chunk_content(document: Any, chunk: Any) -> str:
        table_context = document.metadata.get("table_context", "")
        table_header = document.metadata.get("table_header_markdown", "")
        body = chunk.page_content.strip()
        return "\n".join(part for part in (table_context, table_header, body) if part)

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        chunks = []
        splitters = {}
        for document in documents:
            content_type = str(document.metadata.get("content_type", "text")).lower()
            chunk_size, chunk_overlap = self._chunk_settings(
                document,
                self.chunk_size,
                self.chunk_overlap,
            )

            if content_type == "table":
                prefix_size = len(document.metadata.get("table_context", "")) + len(
                    document.metadata.get("table_header_markdown", "")
                ) + 2
                chunk_size = max(1, chunk_size - prefix_size)
                chunk_overlap = min(chunk_overlap, max(0, chunk_size - 1))

            splitter_key = (chunk_size, chunk_overlap)
            if splitter_key not in splitters:
                splitters[splitter_key] = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""],
                )

            document_chunks = splitters[splitter_key].split_documents([document])
            if content_type == "table":
                for chunk in document_chunks:
                    chunk.page_content = self._table_chunk_content(document, chunk)
            chunks.extend(document_chunks)

        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings


if __name__ == "__main__":
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)