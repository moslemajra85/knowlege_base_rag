import tempfile
import unittest
from pathlib import Path

import pandas as pd

from rag.cache import build_cache_metadata, load_embedding_cache, save_embedding_cache


class CacheTests(unittest.TestCase):
    def test_saved_cache_loads_when_metadata_matches(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "data.csv"
            cache_path = Path(tmp_dir) / ".cache" / "document_embeddings.json"
            dataset_path.write_text("title,content\nhello,world\n", encoding="utf-8")
            metadata = build_cache_metadata(str(dataset_path), 1024, "test-model")
            dataframe = pd.DataFrame(
                [{"chunk": "world", "embedding": [0.1, 0.2, 0.3]}]
            )

            save_embedding_cache(str(cache_path), metadata, dataframe)
            loaded = load_embedding_cache(str(cache_path), metadata)

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.to_dict(orient="records"), dataframe.to_dict(orient="records"))

    def test_cache_is_ignored_when_metadata_does_not_match(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "data.csv"
            cache_path = Path(tmp_dir) / ".cache" / "document_embeddings.json"
            dataset_path.write_text("title,content\nhello,world\n", encoding="utf-8")
            metadata = build_cache_metadata(str(dataset_path), 1024, "test-model")
            dataframe = pd.DataFrame(
                [{"chunk": "world", "embedding": [0.1, 0.2, 0.3]}]
            )

            save_embedding_cache(str(cache_path), metadata, dataframe)
            stale_metadata = dict(metadata)
            stale_metadata["embedding_model"] = "other-model"

            self.assertIsNone(load_embedding_cache(str(cache_path), stale_metadata))


if __name__ == "__main__":
    unittest.main()
