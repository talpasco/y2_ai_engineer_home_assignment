from __future__ import annotations

import unittest

from app.retrieval import RetrievalEntry
from app.vector_retrieval import LocalVectorRetriever, char_ngrams, cosine, hashed_embedding


class VectorRetrievalTests(unittest.TestCase):
    def test_hashed_embedding_is_normalized(self) -> None:
        vector = hashed_embedding("טויוטה קורולה")

        self.assertEqual(len(vector), 256)
        self.assertAlmostEqual(cosine(vector, vector), 1.0, places=5)

    def test_char_ngrams_are_generated(self) -> None:
        grams = char_ngrams("קורולה")

        self.assertTrue(any("קור" in gram for gram in grams))

    def test_local_vector_retriever_finds_transliterated_alias(self) -> None:
        retriever = LocalVectorRetriever(
            [
                RetrievalEntry(
                    category="רכב",
                    field="דגם",
                    value="קורולה",
                    aliases=("corolla", "korola"),
                    weight=4.0,
                )
            ]
        )

        candidates = retriever.retrieve("toyota korola 2020")

        self.assertTrue(candidates)
        self.assertEqual(candidates[0].value, "קורולה")


if __name__ == "__main__":
    unittest.main()

