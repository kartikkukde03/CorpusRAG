import tempfile
import unittest
from pathlib import Path

import pymupdf

from src.data_loader import load_all_documents
from src.rag_pipeline import RAGPipeline


class TestRetrievalQualityBenchmark(unittest.TestCase):
    def test_large_pdf_table_results_keep_page_metadata_and_exact_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "benchmark.pdf"
            pdf = pymupdf.open()
            for page_number in range(1, 41):
                page = pdf.new_page(width=500, height=300)
                page.insert_text((40, 40), f"Annual report page {page_number}")
                if page_number == 20:
                    x_positions = [40, 180, 300, 420]
                    y_positions = [80, 120, 160]
                    for x in x_positions:
                        page.draw_line((x, y_positions[0]), (x, y_positions[-1]))
                    for y in y_positions:
                        page.draw_line((x_positions[0], y), (x_positions[-1], y))
                    table_rows = (("Region", "Population", "Growth", "Share"), ("North", "42", "5%", "0.25"), ("South", "38", "-2%", "0.18"))
                    for row_index, row in enumerate(table_rows):
                        for col_index, value in enumerate(row):
                            page.insert_text((x_positions[col_index] + 8, y_positions[row_index] + 22), value)
                else:
                    page.insert_text((40, 80), f"This is a page with general narrative text for page {page_number}.")
            pdf.save(pdf_path)
            pdf.close()

            documents = load_all_documents(temp_dir)

        table_documents = [doc for doc in documents if doc.metadata.get("content_type") == "table"]
        self.assertTrue(table_documents)
        page_20_table = next(doc for doc in table_documents if doc.metadata.get("page_number") == 20)
        self.assertEqual(page_20_table.metadata["page_count"], 40)
        self.assertIn("Region", page_20_table.metadata["table_headers"])
        self.assertIn("North", page_20_table.page_content)
        self.assertIn("42", page_20_table.page_content)

    def test_table_exact_lookups_and_aggregation_regression_suite(self):
        pipeline = object.__new__(RAGPipeline)
        result = {
            "source_id": "S1",
            "filename": "statistics.pdf",
            "page_number": 27,
            "content_type": "table",
            "table_header_markdown": "| Region | Population | Growth |\n| --- | --- | --- |",
            "content": (
                "| Region | Population | Growth |\n"
                "| --- | --- | --- |\n"
                "| North | 42 | 5% |\n"
                "| South | 38 | -2% |"
            ),
        }

        lookup = pipeline._try_table_cell_lookup("What is the population of North?", [result])
        self.assertEqual(lookup, "North — Population: 42 [S1].")

        sum_answer = pipeline._try_table_aggregation(
            "Sum population for North and South",
            [result],
        )
        self.assertEqual(sum_answer, "Sum of Population for North (42) and South (38) = 80 [S1].")

        average_answer = pipeline._try_table_aggregation(
            "Average population for North and South",
            [result],
        )
        self.assertEqual(average_answer, "Average of Population for North (42) and South (38) = 40 [S1].")

        invalid_answer = pipeline._try_table_aggregation(
            "Sum population for North, South, and East",
            [result],
        )
        self.assertIsNone(invalid_answer)

    def test_legal_exclusion_query_finds_municipal_and_authority_exceptions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "udcpr_exclusion.pdf"
            pdf = pymupdf.open()
            page = pdf.new_page(width=800, height=500)
            page.insert_text((40, 60), "Planning Authorities and Regional Plan Areas")
            page.insert_textbox(
                pymupdf.Rect(40, 90, 760, 450),
                "Applicable to all Planning Authorities and Regional Plan Areas except Municipal "
                "Corporation of Greater Mumbai, Other Planning Authorities / Special Planning "
                "Authorities / Development Authorities within the limit of Municipal Corporation "
                "of Greater Mumbai, MIDC, NAINA, Jawaharlal Nehru Port Trust, Hill Station "
                "Municipal Councils, Eco-sensitive / Eco-fragile region notified by MoEF & CC and "
                "Lonavala Municipal Council, in Maharashtra.",
                fontsize=10,
            )
            pdf.save(pdf_path)
            pdf.close()

            pipeline = RAGPipeline()
            count = pipeline.ingest(str(pdf_path), "udcpr_exclusion.pdf")
            self.assertGreater(count, 0)

            query = (
                "Which areas, planning authorities, or municipal councils in Maharashtra are "
                "specifically excluded from the applicability of UDCPR-2020?"
            )
            results = pipeline.vectorstore.query(query, filename="udcpr_exclusion.pdf", top_k=10)
            self.assertTrue(results)
            joined = "\n".join(result.get("content", "") for result in results)
            self.assertIn("Planning Authorities", joined)
            self.assertIn("Municipal Corporation of Greater Mumbai", joined)
            self.assertIn("Maharashtra", joined)
            self.assertTrue(
                any(
                    phrase in joined
                    for phrase in [
                        "Lonavala Municipal Council",
                        "MIDC",
                        "NAINA",
                        "Hill Station Municipal Councils",
                    ]
                )
            )


if __name__ == "__main__":
    unittest.main()
