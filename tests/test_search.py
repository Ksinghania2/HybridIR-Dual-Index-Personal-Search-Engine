import pickle
import tempfile
import unittest
from pathlib import Path

from src.indexer import DualIndex
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


if __name__ == "__main__":
    unittest.main()