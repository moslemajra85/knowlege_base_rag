import unittest

import pandas as pd

from rag.retrieval import retrieve_top_chunks


class RetrievalTests(unittest.TestCase):
    def test_retrieve_top_chunks_returns_most_similar_chunks_first(self):
        dataframe = pd.DataFrame(
            [
                {"chunk": "least similar", "embedding": [0.0, 1.0]},
                {"chunk": "most similar", "embedding": [1.0, 0.0]},
                {"chunk": "also similar", "embedding": [0.8, 0.2]},
            ]
        )

        chunks = retrieve_top_chunks(dataframe, question_embedding=[1.0, 0.0], top_k=2)

        self.assertEqual(chunks, ["most similar", "also similar"])


if __name__ == "__main__":
    unittest.main()
