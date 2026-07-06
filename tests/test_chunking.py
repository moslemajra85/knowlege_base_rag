import unittest

from rag.chunking import split_into_chunks


class ChunkingTests(unittest.TestCase):
    def test_split_text_into_fixed_size_chunks(self):
        self.assertEqual(split_into_chunks("ABCDEFGHIJ", chunk_size=4), ["ABCD", "EFGH", "IJ"])

    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(split_into_chunks("", chunk_size=4), [])

    def test_chunk_size_must_be_positive(self):
        with self.assertRaises(ValueError):
            split_into_chunks("ABC", chunk_size=0)


if __name__ == "__main__":
    unittest.main()
