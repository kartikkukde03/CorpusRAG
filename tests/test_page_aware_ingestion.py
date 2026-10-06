import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import os

import numpy as np
import pymupdf
import chromadb
from langchain_core.documents import Document

from src.chroma_vectorstore import ChromaVectorStore
from src.data_loader import load_all_documents
from src.embeddings import EmbeddingPipeline
from src.rag_pipeline import (
    CITATION_FAILURE_MESSAGE,
    RAGPipeline,
)


class _FakeCollection:
    def __init__(self, name, filename, document, distance, metadata):
        self.name = name
        self.metadata = {"filename": filename}
        self.document = document
        self.distance = distance
        self.chunk_metadata = metadata

    def count(self):
        return 1

    def query(self, query_embeddings, n_results, include):
        return {
            "ids": [["chunk-0"]],
            "documents": [[self.document]],
            "distances": [[self.distance]],
            "metadatas": [[self.chunk_metadata]],
        }


class _FakeClient:
    def __init__(self):
        self.collections = {
            "sample": _FakeCollection(
                "sample",
                "sample.pdf",
                "Table row population 42",
                0.2,
                {"page_number": 12, "content_type": "table"},
            ),
            "other": _FakeCollection(
                "other",
                "other.pdf",
                "Unrelated words",
                0.1,
                {"page_number": 4, "content_type": "text"},
            ),
        }

    def list_collections(self):
        return [SimpleNamespace(name=name) for name in self.collections]

    def get_collection(self, name):
        return self.collections[name]

    def get_or_create_collection(self, name, metadata):
        return self.collections[name]


class _FakeReranker:
    def predict(self, pairs, show_progress_bar):
        return [1.0 if "population" in content else 0.1 for _, content in pairs]


class _FakeEmbeddingModel:
    def encode(self, texts, show_progress_bar=False):
        return np.asarray([[0.1, 0.2] for _ in texts])


class _FakeVectorStore:
    def __init__(self, results):
        self.results = results
        self.query_text = None

    def query(self, query_text, filename, top_k):
        self.query_text = query_text
        return [dict(result) for result in self.results[:top_k]]

    def query_all(self, query_text, top_k):
        self.query_text = query_text
        return [dict(result) for result in self.results[:top_k]]


class _FakeChatModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        return SimpleNamespace(content=self.responses.pop(0))

    def stream(self, messages):
        yield SimpleNamespace(content="An unsupported answer.")


class TestPageAwareIngestion(unittest.TestCase):
    def test_pdf_text_and_tables_keep_file_and_page_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "sample.pdf"
            pdf = pymupdf.open()
            page = pdf.new_page(width=400, height=300)
            page.insert_text((50, 35), "Annual report")
            page.insert_text((50, 55), "Regional population statistics")

            x_positions = [50, 180, 300]
            y_positions = [80, 120, 160]
            for x in x_positions:
                page.draw_line((x, y_positions[0]), (x, y_positions[-1]))
            for y in y_positions:
                page.draw_line((x_positions[0], y), (x_positions[-1], y))
            for row_index, row in enumerate((("Region", "Count"), ("North", "42"))):
                for column_index, value in enumerate(row):
                    page.insert_text(
                        (x_positions[column_index] + 5, y_positions[row_index] + 25),
                        value,
                    )

            pdf.new_page(width=400, height=300).insert_text((50, 50), "Page two text")
            pdf.save(pdf_path)
            pdf.close()

            documents = load_all_documents(temp_dir)

        prose = next(doc for doc in documents if doc.metadata["content_type"] == "text")
        table = next(doc for doc in documents if doc.metadata["content_type"] == "table")
        self.assertEqual(prose.metadata["filename"], "sample.pdf")
        self.assertEqual(prose.metadata["page_number"], 1)
        self.assertEqual(prose.metadata["page_count"], 2)
        self.assertEqual(table.metadata["page_number"], 1)
        self.assertEqual(table.metadata["table_headers"], "Region | Count")

        pipeline = object.__new__(EmbeddingPipeline)
        pipeline.chunk_size = 1000
        pipeline.chunk_overlap = 200
        table_chunks = pipeline.chunk_documents([table])
        self.assertIn("Document: sample.pdf", table_chunks[0].page_content)
        self.assertIn("Page: 1", table_chunks[0].page_content)
        self.assertIn("| Region | Count |", table_chunks[0].page_content)
        self.assertIn("| North | 42 |", table_chunks[0].page_content)

    def test_pdf_chunk_size_uses_page_count_boundary(self):
        pipeline = object.__new__(EmbeddingPipeline)
        pipeline.chunk_size = 1000
        pipeline.chunk_overlap = 200
        text = "word " * 1000
        documents = [
            Document(page_content=text, metadata={"page_count": 15}),
            Document(page_content=text, metadata={"page_count": 16}),
        ]

        small_chunks = pipeline.chunk_documents(documents[:1])
        large_chunks = pipeline.chunk_documents(documents[1:])

        self.assertTrue(all(len(chunk.page_content) <= 800 for chunk in small_chunks))
        self.assertTrue(all(len(chunk.page_content) <= 1800 for chunk in large_chunks))

    def test_400_page_pdf_retains_late_page_numbers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "long_manual.pdf"
            pdf = pymupdf.open()
            for page_number in range(1, 401):
                page = pdf.new_page(width=400, height=300)
                page.insert_text(
                    (50, 50),
                    f"Manual content for page {page_number}; late-page-marker-{page_number}",
                )
            pdf.save(pdf_path)
            pdf.close()

            documents = load_all_documents(temp_dir)

        self.assertEqual(len(documents), 400)
        final_page = documents[-1]
        self.assertEqual(final_page.metadata["page_count"], 400)
        self.assertEqual(final_page.metadata["page_number"], 400)
        self.assertIn("late-page-marker-400", final_page.page_content)

    def test_vector_queries_return_page_and_file_metadata(self):
        vectorstore = object.__new__(ChromaVectorStore)
        vectorstore.client = _FakeClient()
        vectorstore.model = SimpleNamespace(
            encode=lambda texts: np.asarray([[0.1, 0.2]])
        )
        vectorstore._reranker = _FakeReranker()
        vectorstore.reranker_model_name = "test-reranker"

        with tempfile.TemporaryDirectory() as temp_dir:
            vectorstore.bm25_path = os.path.join(temp_dir, "bm25.sqlite3")
            vectorstore._initialize_bm25_index()
            vectorstore._replace_bm25_collection(
                "sample",
                ["chunk-0"],
                ["Table row population 42"],
                [{"filename": "sample.pdf", "page_number": 12}],
                "sample-signature",
            )
            vectorstore._replace_bm25_collection(
                "other",
                ["chunk-0"],
                ["Unrelated words"],
                [{"filename": "other.pdf", "page_number": 4}],
                "other-signature",
            )
            sparse_results = vectorstore._bm25_candidates(
                "population",
                candidate_k=2,
                collection_id=None,
            )
            selected_result = vectorstore.query("population", "sample.pdf")[0]
            all_results = vectorstore.query_all("population", top_k=2)

        self.assertEqual(sparse_results[0]["filename"], "sample.pdf")
        self.assertEqual(selected_result["filename"], "sample.pdf")
        self.assertEqual(selected_result["page_number"], 12)
        self.assertEqual(selected_result["content_type"], "table")
        self.assertEqual(all_results[0]["filename"], "sample.pdf")
        self.assertEqual(all_results[0]["rerank_score"], 1.0)
        self.assertEqual(len(all_results), 2)

    def test_bm25_sync_works_with_chroma_collection_metadata(self):
        client = chromadb.EphemeralClient()
        collection = client.create_collection(
            name="stage2-test",
            metadata={"filename": "sample.pdf", "hnsw:space": "l2"},
        )
        collection.add(
            ids=["chunk-0"],
            embeddings=[[0.1, 0.2]],
            documents=["population statistics North 42"],
            metadatas=[{"filename": "sample.pdf", "page_number": 12}],
        )

        vectorstore = object.__new__(ChromaVectorStore)
        vectorstore.client = client
        vectorstore.model = SimpleNamespace(
            encode=lambda texts: np.asarray([[0.1, 0.2]])
        )
        vectorstore._reranker = _FakeReranker()
        vectorstore.reranker_model_name = "test-reranker"

        with tempfile.TemporaryDirectory() as temp_dir:
            vectorstore.bm25_path = os.path.join(temp_dir, "bm25.sqlite3")
            vectorstore._initialize_bm25_index()
            vectorstore._sync_bm25_index()
            result = vectorstore.query_all("population", top_k=1)[0]

        self.assertEqual(result["filename"], "sample.pdf")
        self.assertEqual(result["page_number"], 12)
        self.assertTrue(client.get_collection("stage2-test").metadata["bm25_signature"])

    def test_ingestion_writes_chunks_to_chroma_and_bm25(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            vectorstore = object.__new__(ChromaVectorStore)
            vectorstore.persist_dir = temp_dir
            vectorstore.bm25_path = os.path.join(temp_dir, "bm25.sqlite3")
            vectorstore.client = chromadb.EphemeralClient()
            vectorstore.model = _FakeEmbeddingModel()
            vectorstore.embedding_model_name = "test-embedding-model"
            vectorstore.chunk_size = 1000
            vectorstore.chunk_overlap = 200
            vectorstore._reranker = _FakeReranker()
            vectorstore.reranker_model_name = "test-reranker"
            vectorstore._initialize_bm25_index()

            count = vectorstore.build_from_documents(
                [
                    Document(
                        page_content="Population statistics show North has 42 residents.",
                        metadata={
                            "filename": "manual.pdf",
                            "page_number": 27,
                            "page_count": 400,
                            "content_type": "prose",
                        },
                    )
                ],
                filename="manual.pdf",
            )
            result = vectorstore.query("population statistics", "manual.pdf", top_k=1)[0]

        self.assertEqual(count, 1)
        self.assertEqual(result["page_number"], 27)
        self.assertEqual(result["filename"], "manual.pdf")
        self.assertEqual(result["content_type"], "text")
        self.assertGreater(result["hybrid_score"], 0)

    def test_followup_query_is_reformulated_and_only_five_turns_are_sent(self):
        pipeline = object.__new__(RAGPipeline)
        pipeline.ingested_filename = "manual.pdf"
        pipeline._last_sources = []
        pipeline._last_citation_error = None
        pipeline.vectorstore = _FakeVectorStore(
            [
                {
                    "content": "The report specifies a limit.",
                    "filename": "manual.pdf",
                    "page_number": 15,
                }
            ]
        )
        pipeline.llm = _FakeChatModel(["annual reporting limit"])
        history = [(f"Question {index}", f"Answer {index}") for index in range(7)]

        messages, results, table_answer = pipeline._build_messages(
            "What about the limit?",
            history,
            top_k=5,
        )

        self.assertEqual(pipeline.vectorstore.query_text, "annual reporting limit")
        self.assertEqual(len(messages), 12)
        self.assertEqual(results[0]["source_id"], "S1")
        self.assertIsNone(table_answer)
        self.assertIn("[S1: manual.pdf, page 15]", messages[0].content)

    def test_exact_numeric_table_cell_lookup_returns_citation(self):
        pipeline = object.__new__(RAGPipeline)
        result = {
            "source_id": "S1",
            "filename": "statistics.pdf",
            "page_number": 27,
            "content_type": "table",
            "table_header_markdown": "| Region | Population |\n| --- | --- |",
            "content": (
                "Document: statistics.pdf\nPage: 27\n"
                "| Region | Population |\n| --- | --- |\n| North | 1,234 |"
            ),
        }

        answer = pipeline._try_table_cell_lookup(
            "What is the population of North?",
            [result],
        )
        ambiguous_answer = pipeline._try_table_cell_lookup(
            "What is the population?",
            [result, {**result, "source_id": "S2", "page_number": 28}],
        )

        self.assertEqual(answer, "North — Population: 1,234 [S1].")
        self.assertIsNone(ambiguous_answer)

    def test_explicit_table_sum_includes_auditable_inputs_and_citation(self):
        pipeline = object.__new__(RAGPipeline)
        result = {
            "source_id": "S1",
            "filename": "statistics.pdf",
            "page_number": 27,
            "table_index": 1,
            "content_type": "table",
            "table_header_markdown": "| Region | Population |\n| --- | --- |",
            "content": (
                "| Region | Population |\n| --- | --- |\n"
                "| North | 10 |\n| South | 20 |"
            ),
        }

        answer = pipeline._try_table_aggregation(
            "Sum population for North and South",
            [result],
        )

        self.assertEqual(
            answer,
            "Sum of Population for North (10) and South (20) = 30 [S1].",
        )

    def test_explicit_table_average_formats_exact_decimal(self):
        pipeline = object.__new__(RAGPipeline)
        result = {
            "source_id": "S1",
            "filename": "statistics.pdf",
            "page_number": 27,
            "table_index": 1,
            "content_type": "table",
            "table_header_markdown": "| Region | Rate |\n| --- | --- |",
            "content": "| Region | Rate |\n| --- | --- |\n| North | 1 |\n| South | 2 |",
        }

        answer = pipeline._try_table_aggregation(
            "Average Rate for North and South",
            [result],
        )

        self.assertEqual(
            answer,
            "Average of Rate for North (1) and South (2) = 1.5 [S1].",
        )

    def test_table_aggregation_refuses_missing_rows_and_implicit_all_rows(self):
        pipeline = object.__new__(RAGPipeline)
        result = {
            "source_id": "S1",
            "filename": "statistics.pdf",
            "page_number": 27,
            "table_index": 1,
            "content_type": "table",
            "table_header_markdown": "| Region | Population |\n| --- | --- |",
            "content": "| Region | Population |\n| --- | --- |\n| North | 10 |\n| South | 20 |",
        }

        missing_row_answer = pipeline._try_table_aggregation(
            "Sum population for North, South, and West",
            [result],
        )
        implicit_rows_answer = pipeline._try_table_aggregation(
            "Sum population across all regions",
            [result],
        )

        self.assertIsNone(missing_row_answer)
        self.assertIsNone(implicit_rows_answer)

    def test_table_aggregation_supports_minimum_maximum_and_count(self):
        pipeline = object.__new__(RAGPipeline)
        result = {
            "source_id": "S1",
            "filename": "statistics.pdf",
            "page_number": 27,
            "table_index": 1,
            "content_type": "table",
            "table_header_markdown": "| Region | Rate |\n| --- | --- |",
            "content": "| Region | Rate |\n| --- | --- |\n| North | 1 |\n| South | 2 |",
        }

        minimum = pipeline._try_table_aggregation(
            "Minimum Rate for North and South",
            [result],
        )
        maximum = pipeline._try_table_aggregation(
            "Maximum Rate for North and South",
            [result],
        )
        count = pipeline._try_table_aggregation(
            "Count Rate for North and South",
            [result],
        )

        self.assertEqual(
            minimum,
            "Minimum of Rate among North (1) and South (2) is North (1) [S1].",
        )
        self.assertEqual(
            maximum,
            "Maximum of Rate among North (1) and South (2) is South (2) [S1].",
        )
        self.assertEqual(
            count,
            "Count of Rate values for North (1) and South (2) = 2 [S1].",
        )

    def test_ask_uses_verified_table_calculation_without_calling_llm(self):
        pipeline = object.__new__(RAGPipeline)
        pipeline.ingested_filename = "statistics.pdf"
        pipeline._last_sources = []
        pipeline._last_citation_error = None
        pipeline.vectorstore = _FakeVectorStore(
            [
                {
                    "filename": "statistics.pdf",
                    "page_number": 27,
                    "table_index": 1,
                    "content_type": "table",
                    "table_header_markdown": "| Region | Population |\n| --- | --- |",
                    "content": (
                        "| Region | Population |\n| --- | --- |\n"
                        "| North | 10 |\n| South | 20 |"
                    ),
                }
            ]
        )
        pipeline.llm = _FakeChatModel([])

        answer = pipeline.ask("Sum population for North and South")

        self.assertIn("= 30 [S1].", answer)
        self.assertEqual(pipeline.llm.messages, [])

    def test_count_column_remains_an_exact_cell_lookup(self):
        pipeline = object.__new__(RAGPipeline)
        pipeline.ingested_filename = "statistics.pdf"
        pipeline._last_sources = []
        pipeline._last_citation_error = None
        pipeline.vectorstore = _FakeVectorStore(
            [
                {
                    "filename": "statistics.pdf",
                    "page_number": 27,
                    "content_type": "table",
                    "table_header_markdown": "| Region | Count |\n| --- | --- |",
                    "content": (
                        "| Region | Count |\n| --- | --- |\n"
                        "| North | 10 |"
                    ),
                }
            ]
        )
        pipeline.llm = _FakeChatModel([])

        answer = pipeline.ask("What is the Count for North?")

        self.assertEqual(answer, "North — Count: 10 [S1].")
        self.assertEqual(pipeline.llm.messages, [])

    def test_non_stream_answer_falls_back_to_cited_source_excerpt(self):
        pipeline = object.__new__(RAGPipeline)
        pipeline.ingested_filename = "manual.pdf"
        pipeline._last_sources = []
        pipeline._last_citation_error = None
        pipeline.vectorstore = _FakeVectorStore(
            [
                {
                    "content": "The report specifies a limit.",
                    "filename": "manual.pdf",
                    "page_number": 15,
                }
            ]
        )
        pipeline.llm = _FakeChatModel(["The limit is 15.", "The limit is 15."])

        answer = pipeline.ask("What is the limit?")

        self.assertIn("retrieved excerpt", answer)
        self.assertIn("The report specifies a limit.", answer)
        self.assertIn("[S1]", answer)
        self.assertIsNone(pipeline.get_last_citation_error())

    def test_non_stream_answer_retries_once_when_citation_is_missing(self):
        pipeline = object.__new__(RAGPipeline)
        pipeline.ingested_filename = "manual.pdf"
        pipeline._last_sources = []
        pipeline._last_citation_error = None
        pipeline.vectorstore = _FakeVectorStore(
            [
                {
                    "content": "The report specifies a limit of 15.",
                    "filename": "manual.pdf",
                    "page_number": 15,
                }
            ]
        )
        pipeline.llm = _FakeChatModel(
            ["The limit is 15.", "The limit is 15 [S1]."]
        )

        answer = pipeline.ask("What is the limit?")

        self.assertEqual(answer, "The limit is 15 [S1].")
        self.assertIsNone(pipeline.get_last_citation_error())
        self.assertEqual(len(pipeline.llm.messages), 2)

    def test_stream_falls_back_to_cited_source_excerpt(self):
        pipeline = object.__new__(RAGPipeline)
        pipeline.ingested_filename = "manual.pdf"
        pipeline._last_sources = []
        pipeline._last_citation_error = None
        pipeline.vectorstore = _FakeVectorStore(
            [
                {
                    "content": "The report specifies a limit.",
                    "filename": "manual.pdf",
                    "page_number": 15,
                }
            ]
        )
        pipeline.llm = _FakeChatModel(["Still no citation"])

        answer = "".join(pipeline.stream("What is the limit?"))

        self.assertIn("retrieved excerpt", answer)
        self.assertIn("The report specifies a limit.", answer)
        self.assertIn("[S1]", answer)
        self.assertIsNone(pipeline.get_last_citation_error())

    def test_stream_retries_without_exposing_uncited_draft(self):
        pipeline = object.__new__(RAGPipeline)
        pipeline.ingested_filename = "manual.pdf"
        pipeline._last_sources = []
        pipeline._last_citation_error = None
        pipeline.vectorstore = _FakeVectorStore(
            [
                {
                    "content": "The report specifies a limit of 15.",
                    "filename": "manual.pdf",
                    "page_number": 15,
                }
            ]
        )
        pipeline.llm = _FakeChatModel(
            ["The limit is 15 [S1]."]
        )

        answer = "".join(pipeline.stream("What is the limit?"))

        self.assertEqual(answer, "The limit is 15 [S1].")
        self.assertIsNone(pipeline.get_last_citation_error())


if __name__ == "__main__":
    unittest.main()
