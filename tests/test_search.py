import pickle
import tempfile
import unittest
from pathlib import Path

from src.indexer import DualIndex, load_documents
from src.retriever import Retriever


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.data_dir = root / "data"
        self.index_dir = root / "index"
        self.data_dir.mkdir()
        (self.data_dir / "transformers.md").write_text(
            "Transformers use attention mechanisms in neural networks.",
            encoding="utf-8",
        )
        (self.data_dir / "gardening.txt").write_text(
            "Tomatoes need sunlight, water, and healthy soil.",
            encoding="utf-8",
        )
        self.index = DualIndex(str(self.data_dir), str(self.index_dir))
        self.index.build()
        self.retriever = Retriever(self.index)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_all_search_modes_find_matching_document(self):
        for mode in ("bm25", "vector", "hybrid"):
            with self.subTest(mode=mode):
                results = self.retriever.search(
                    "transformers neural networks", mode=mode, top_k=3
                )
                self.assertTrue(results)
                self.assertEqual(results[0]["doc"]["title"], "transformers.md")
                self.assertEqual(results[0]["source"], mode)

    def test_bm25_does_not_return_documents_without_token_overlap(self):
        results = self.retriever.search("volcano submarine", mode="bm25")

        self.assertEqual(results, [])

    def test_index_can_be_reloaded(self):
        reloaded_index = DualIndex(str(self.data_dir), str(self.index_dir))
        reloaded_index.ensure_loaded()

        self.assertEqual(reloaded_index.num_docs, 2)
        results = Retriever(reloaded_index).search("transformers", mode="bm25")
        self.assertEqual(results[0]["doc"]["title"], "transformers.md")

    def test_outdated_index_cache_is_rebuilt(self):
        with (self.index_dir / "index.pkl").open("wb") as cache_file:
            pickle.dump({"format_version": 1}, cache_file)

        reloaded_index = DualIndex(str(self.data_dir), str(self.index_dir))
        reloaded_index.ensure_loaded()

        self.assertEqual(reloaded_index.num_docs, 2)
        results = Retriever(reloaded_index).search("transformers", mode="bm25")
        self.assertEqual(results[0]["doc"]["title"], "transformers.md")

    def test_supported_text_and_html_formats_are_loaded(self):
        (self.data_dir / "notes.rst").write_text(
            "A reStructuredText note.", encoding="utf-8"
        )
        (self.data_dir / "events.csv").write_text(
            "date,event\n2026-01-01,launch", encoding="utf-8"
        )
        (self.data_dir / "details.json").write_text(
            '{"topic": "search indexing"}', encoding="utf-8"
        )
        (self.data_dir / "page.html").write_text(
            "<html><style>hidden style</style><body>Visible <b>HTML</b>"
            "<script>hidden script</script></body></html>",
            encoding="utf-8",
        )

        documents = load_documents(str(self.data_dir))
        by_title = {document["title"]: document["text"] for document in documents}

        self.assertIn("notes.rst", by_title)
        self.assertIn("2026-01-01,launch", by_title["events.csv"])
        self.assertIn("search indexing", by_title["details.json"])
        self.assertIn("Visible HTML", by_title["page.html"])
        self.assertNotIn("hidden", by_title["page.html"])

    def test_search_matches_document_titles_and_prioritizes_title_hits(self):
        (self.data_dir / "machine-learning.md").write_text(
            "Personal notes about experiments and evaluation.",
            encoding="utf-8",
        )
        (self.data_dir / "general-notes.md").write_text(
            "Machine learning experiments and evaluation notes.",
            encoding="utf-8",
        )
        self.index.build()
        retriever = Retriever(self.index)

        for mode in ("bm25", "vector", "hybrid"):
            with self.subTest(mode=mode):
                results = retriever.search("machine learning", mode=mode)
                self.assertEqual(
                    results[0]["doc"]["title"], "machine-learning.md"
                )

    def test_search_validates_mode_and_result_count(self):
        with self.assertRaisesRegex(ValueError, "Unsupported search mode"):
            self.retriever.search("transformers", mode="unknown")
        with self.assertRaisesRegex(ValueError, "top_k"):
            self.retriever.search("transformers", top_k=0)
        self.assertEqual(self.retriever.search("!!!", mode="vector"), [])

    def test_rebuilding_empty_collection_clears_previous_index(self):
        for document in self.data_dir.iterdir():
            document.unlink()

        self.index.build()

        self.assertEqual(self.index.num_docs, 0)
        self.assertEqual(self.retriever.search("transformers"), [])


if __name__ == "__main__":
    unittest.main()