"""Pluggable embedding engine for semantic candidate retrieval.

The service is designed so the retrieval layer is model-agnostic. Two backends
are provided behind one `Embedder` contract:

- `HashedEmbedder` (default): deterministic, dependency-free hashed character
  n-gram vectors. Catches spelling/transliteration overlap, needs no model
  download, and keeps local/Docker runs and tests fully reproducible.
- `OpenAIEmbedder` (optional): real learned embeddings (`text-embedding-3-small`
  by default) for genuine semantic recall over long-tail Hebrew/transliterated
  phrasing. Vectors are L2-normalized and cached in-process so repeated taxonomy
  values and popular queries do not re-pay the embedding cost.

Because both implement the same `embed(text) -> list[float]` / `dim` contract,
the parser, local retriever, and Qdrant retriever do not change when the backend
is swapped. In production this seam can also host a self-hosted multilingual
model (E5 / bge-m3 / a Hebrew-tuned encoder) served via an inference container.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from typing import Protocol, runtime_checkable


EMBEDDING_DIM = 256


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]:
        ...


class HashedEmbedder:
    """Deterministic hashed character n-gram embedder (no external calls)."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        return hashed_embedding(text, dim=self.dim)


class OpenAIEmbedder:
    """Learned-embedding backend backed by the OpenAI embeddings API.

    Results are L2-normalized (so dot product == cosine similarity, matching the
    local backend and Qdrant's Cosine distance) and memoized in a bounded LRU so
    that taxonomy/alias vectors and repeated queries are embedded at most once.
    """

    _DEFAULT_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        *,
        dim: int | None = None,
        cache_size: int = 50_000,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.dim = dim or self._DEFAULT_DIMS.get(model, 1536)
        self._cache_size = cache_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._client = None

    def embed(self, text: str) -> list[float]:
        key = " ".join(text.split())
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        vector = self._normalize(self._embed_remote(key))
        self._cache[key] = vector
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return vector

    def _embed_remote(self, text: str) -> list[float]:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        response = self._client.embeddings.create(model=self.model, input=text or " ")
        return list(response.data[0].embedding)

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        length = math.sqrt(sum(value * value for value in vector))
        if length == 0:
            return vector
        return [value / length for value in vector]


def build_embedder(
    backend: str,
    *,
    openai_api_key: str | None = None,
    openai_model: str = "text-embedding-3-small",
) -> Embedder:
    """Return an embedder for the requested backend, falling back to hashed.

    `openai` is only selected when an API key is present; otherwise the safe,
    reproducible hashed backend is used so the service always starts.
    """
    if backend == "openai" and openai_api_key:
        return OpenAIEmbedder(openai_api_key, openai_model)
    return HashedEmbedder()


def hashed_embedding(text: str, *, dim: int = EMBEDDING_DIM) -> list[float]:
    from app.normalizer import normalize_text

    normalized = normalize_text(text, {})
    vector = [0.0] * dim
    for gram in char_ngrams(normalized):
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    length = math.sqrt(sum(value * value for value in vector))
    if length == 0:
        return vector
    return [value / length for value in vector]


def char_ngrams(text: str) -> list[str]:
    compact = " " + " ".join(text.split()) + " "
    grams: list[str] = []
    for ngram_size in (3, 4, 5):
        for index in range(0, max(0, len(compact) - ngram_size + 1)):
            grams.append(compact[index : index + ngram_size])
    return grams


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
