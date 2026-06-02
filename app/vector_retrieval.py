from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict
from typing import Iterable

from app.embeddings import (
    EMBEDDING_DIM,
    Embedder,
    HashedEmbedder,
    char_ngrams,
    cosine,
    hashed_embedding,
)
from app.retrieval import RetrievalCandidate, RetrievalEntry


__all__ = [
    "EMBEDDING_DIM",
    "VectorRetriever",
    "LocalVectorRetriever",
    "QdrantVectorRetriever",
    "hashed_embedding",
    "char_ngrams",
    "cosine",
]


class VectorRetriever:
    def retrieve(self, query: str, *, limit: int = 12) -> list[RetrievalCandidate]:
        raise NotImplementedError


class LocalVectorRetriever(VectorRetriever):
    """In-process semantic retrieval over an injectable embedding backend.

    Defaults to the dependency-free hashed embedder so tests and Docker runs stay
    reproducible, but accepts any `Embedder` (e.g. `OpenAIEmbedder`) without
    changing the candidate contract returned to the parser.
    """

    def __init__(
        self,
        entries: Iterable[RetrievalEntry],
        *,
        embedder: Embedder | None = None,
    ) -> None:
        self.embedder = embedder or HashedEmbedder()
        self._points: list[tuple[list[float], RetrievalEntry, str]] = []
        for entry in entries:
            documents = [entry.value, *entry.aliases]
            document_text = " ".join(str(doc) for doc in documents if doc)
            self._points.append(
                (self.embedder.embed(document_text), entry, document_text)
            )

    def retrieve(self, query: str, *, limit: int = 12) -> list[RetrievalCandidate]:
        query_vector = self.embedder.embed(query)
        scored: list[RetrievalCandidate] = []
        for vector, entry, evidence in self._points:
            score = cosine(query_vector, vector)
            if score < 0.18:
                continue
            scored.append(
                RetrievalCandidate(
                    category=entry.category,
                    field=entry.field,
                    value=entry.value,
                    score=score * entry.weight,
                    evidence=evidence,
                    match_type="vector_local",
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]


class QdrantVectorRetriever(VectorRetriever):
    """Optional Qdrant-backed vector retriever.

    Uses the injected `Embedder` for both indexing and query time, so the same
    Qdrant collection works whether vectors come from the deterministic hashed
    baseline or a learned model (OpenAI / multilingual E5 / bge-m3 / a Hebrew-tuned
    encoder). The vector size is taken from the embedder, so swapping backends only
    requires a fresh collection name.
    """

    def __init__(
        self,
        entries: Iterable[RetrievalEntry],
        *,
        url: str,
        collection: str = "yad2_taxonomy_candidates",
        timeout_seconds: float = 2.0,
        embedder: Embedder | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.collection = collection
        self.timeout_seconds = timeout_seconds
        self.embedder = embedder or HashedEmbedder()
        self.ready = False
        self._entries = list(entries)
        try:
            self._ensure_collection()
            self._upsert_entries()
            self.ready = True
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            self.ready = False

    def retrieve(self, query: str, *, limit: int = 12) -> list[RetrievalCandidate]:
        if not self.ready:
            return []
        payload = {
            "vector": self.embedder.embed(query),
            "limit": limit,
            "with_payload": True,
        }
        data = self._request(
            "POST", f"/collections/{self.collection}/points/search", payload
        )
        result = data.get("result", [])
        candidates: list[RetrievalCandidate] = []
        for item in result:
            payload = item.get("payload") or {}
            if not payload:
                continue
            candidates.append(
                RetrievalCandidate(
                    category=payload["category"],
                    field=payload["field"],
                    value=payload["value"],
                    score=float(item.get("score", 0.0)) * float(payload.get("weight", 1.0)),
                    evidence=payload.get("evidence", ""),
                    match_type="vector_qdrant",
                )
            )
        return candidates

    def _ensure_collection(self) -> None:
        config = {
            "vectors": {
                "size": self.embedder.dim,
                "distance": "Cosine",
            }
        }
        try:
            self._request("PUT", f"/collections/{self.collection}", config)
        except urllib.error.HTTPError as exc:
            if exc.code not in {409}:
                raise

    def _upsert_entries(self) -> None:
        points = []
        for index, entry in enumerate(self._entries):
            evidence = " ".join([entry.value, *entry.aliases])
            points.append(
                {
                    "id": index,
                    "vector": self.embedder.embed(evidence),
                    "payload": {
                        **asdict(entry),
                        "aliases": list(entry.aliases),
                        "evidence": evidence,
                    },
                }
            )
        self._request(
            "PUT",
            f"/collections/{self.collection}/points?wait=true",
            {"points": points},
        )

    def _request(self, method: str, path: str, body: dict) -> dict:
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

