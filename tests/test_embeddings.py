from __future__ import annotations

import math
import unittest

from app.embeddings import (
    HashedEmbedder,
    OpenAIEmbedder,
    build_embedder,
    cosine,
)


class EmbeddingsTests(unittest.TestCase):
    def test_build_embedder_defaults_to_hashed(self) -> None:
        self.assertIsInstance(build_embedder("hashed"), HashedEmbedder)

    def test_build_embedder_falls_back_without_api_key(self) -> None:
        # openai requested but no key -> must still start with the hashed backend
        self.assertIsInstance(build_embedder("openai", openai_api_key=None), HashedEmbedder)

    def test_build_embedder_uses_openai_when_key_present(self) -> None:
        embedder = build_embedder("openai", openai_api_key="sk-test")
        self.assertIsInstance(embedder, OpenAIEmbedder)
        self.assertEqual(embedder.dim, 1536)

    def test_hashed_embedder_is_unit_length(self) -> None:
        vector = HashedEmbedder().embed("טויוטה קורולה")
        self.assertEqual(len(vector), 256)
        self.assertAlmostEqual(cosine(vector, vector), 1.0, places=5)

    def test_openai_embedder_normalizes_and_caches(self) -> None:
        embedder = OpenAIEmbedder("sk-test", dim=3)
        calls: list[str] = []

        def fake_remote(text: str) -> list[float]:
            calls.append(text)
            return [3.0, 0.0, 4.0]

        embedder._embed_remote = fake_remote  # type: ignore[method-assign]

        first = embedder.embed("ספה  עור")
        second = embedder.embed("ספה עור")  # normalized to same key -> cache hit

        self.assertEqual(len(calls), 1)
        self.assertEqual(first, second)
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in first)), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
