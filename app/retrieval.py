from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

from app.normalizer import canonical_phrase


@dataclass(frozen=True, slots=True)
class RetrievalEntry:
    category: str
    field: str
    value: str
    aliases: tuple[str, ...]
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    category: str
    field: str
    value: str
    score: float
    evidence: str
    match_type: str


class LocalCandidateRetriever:
    """Small local candidate retriever for taxonomy values and aliases.

    This is the local implementation of the production retrieval seam. A larger
    deployment can replace it with embeddings/vector search while preserving the
    same candidate contract.
    """

    def __init__(self, entries: Iterable[RetrievalEntry]) -> None:
        self.entries = list(entries)

    def retrieve(self, query: str, *, limit: int = 32) -> list[RetrievalCandidate]:
        candidates: list[RetrievalCandidate] = []
        for entry in self.entries:
            best = self._best_entry_match(query, entry)
            if best:
                candidates.append(best)
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return candidates[:limit]

    def _best_entry_match(
        self, query: str, entry: RetrievalEntry
    ) -> RetrievalCandidate | None:
        best: RetrievalCandidate | None = None
        phrases = (entry.value, *entry.aliases)
        for phrase in phrases:
            phrase = canonical_phrase(str(phrase))
            if not phrase:
                continue
            if phrase_in_text(query, phrase):
                score = entry.weight
                candidate = RetrievalCandidate(
                    entry.category, entry.field, entry.value, score, phrase, "exact"
                )
            elif fuzzy_phrase_in_text(query, phrase):
                score = entry.weight * 0.82
                candidate = RetrievalCandidate(
                    entry.category, entry.field, entry.value, score, phrase, "fuzzy"
                )
            else:
                continue
            if best is None or candidate.score > best.score:
                best = candidate
        return best


def phrase_in_text(text: str, phrase: str) -> bool:
    phrase = canonical_phrase(str(phrase))
    if not phrase:
        return False
    return bool(
        re.search(rf"(?<![\wא-ת]){re.escape(phrase)}(?![\wא-ת])", text, re.IGNORECASE)
    )


def any_phrase(text: str, phrases: list[str] | tuple[str, ...]) -> bool:
    return any(phrase_in_text(text, phrase) for phrase in phrases)


def any_phrase_fuzzy(
    text: str, phrases: list[str] | tuple[str, ...], *, threshold: float = 0.88
) -> bool:
    return any(fuzzy_phrase_in_text(text, phrase, threshold=threshold) for phrase in phrases)


def fuzzy_phrase_in_text(text: str, phrase: str, *, threshold: float = 0.88) -> bool:
    phrase = canonical_phrase(str(phrase))
    if not can_fuzzy_match(phrase):
        return False
    tokens = text.split()
    phrase_tokens = phrase.split()
    token_count = len(phrase_tokens)
    if token_count == 0 or token_count > 4:
        return False

    candidates: list[str] = []
    for index in range(0, max(0, len(tokens) - token_count + 1)):
        candidates.append(" ".join(tokens[index : index + token_count]))
    if token_count == 1:
        candidates.extend(tokens)

    for candidate in candidates:
        candidate = candidate.strip(".,;:!?()[]{}\"'")
        if not can_fuzzy_match(candidate):
            continue
        if abs(len(candidate) - len(phrase)) > max(2, round(len(phrase) * 0.3)):
            continue
        if SequenceMatcher(None, candidate, phrase).ratio() >= threshold:
            return True
    return False


def can_fuzzy_match(value: str) -> bool:
    value = value.strip()
    if len(value) < 4:
        return False
    if value.isdigit():
        return False
    return bool(re.search(r"[א-תa-zA-Z]", value))

