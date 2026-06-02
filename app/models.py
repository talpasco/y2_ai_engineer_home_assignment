from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass(slots=True)
class ParseResult:
    category: str
    params: dict[str, Any]
    confidence: float
    notes: list[str] = field(default_factory=list)
    source: str = "rules"
    cache_hit: bool = False
    model_name: str | None = None
    model_call_success: bool = False
    model_call_failure: bool = False
    usage: TokenUsage = field(default_factory=TokenUsage)

    def public_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "category": self.category,
            "params": self.params,
            "confidence": round(max(0.0, min(1.0, self.confidence)), 3),
        }
        if self.notes:
            body["notes"] = self.notes
        body["resolution"] = {
            "source": self.source,
            "model": self.model_name,
            "cache_hit": self.cache_hit,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "cost_usd": round(self.usage.estimated_cost_usd, 8),
        }
        return body


class QueryValidationError(ValueError):
    """Raised when the user query cannot be accepted for parsing."""
