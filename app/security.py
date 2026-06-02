from __future__ import annotations

import hmac
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from threading import Lock

from app.models import QueryValidationError


INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore\s+(all\s+)?previous",
        r"system\s+prompt",
        r"developer\s+message",
        r"act\s+as\s+",
        r"return\s+.*json",
        r"התעלם\s+מה",
        r"תשכח\s+את",
        r"הוראות\s+מערכת",
        r"פרומפט",
    )
)

BOT_USER_AGENT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bcurl\b",
        r"\bwget\b",
        r"\bpython-requests\b",
        r"\bscrapy\b",
        r"\bselenium\b",
        r"\bheadless\b",
        r"\bphantomjs\b",
        r"\bspider\b",
        r"\bcrawler\b",
        r"\bbot\b",
    )
)

SUSPICIOUS_QUERY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"<script",
        r"javascript:",
        r"select\s+.+\s+from",
        r"union\s+select",
        r"\.\./",
        r"base64",
        r"BEGIN\s+RSA",
    )
)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?972|0)?5\d[\s-]?\d{3}[\s-]?\d{4}(?!\d)")


def clean_control_and_symbols(text: str) -> str:
    chars: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category.startswith("C"):
            continue
        if category == "So" and char not in {"₪"}:
            continue
        chars.append(char)
    return "".join(chars)


def redact_query(text: str, *, max_chars: int = 160) -> str:
    redacted = EMAIL_RE.sub("[email]", text)
    redacted = PHONE_RE.sub("[phone]", redacted)
    if len(redacted) > max_chars:
        redacted = redacted[:max_chars] + "...[truncated]"
    return redacted


def validate_query_shape(query: object, max_chars: int) -> str:
    if not isinstance(query, str):
        raise QueryValidationError("q must be a string")
    cleaned = clean_control_and_symbols(query).strip()
    if not cleaned:
        raise QueryValidationError("q must not be empty")
    if len(cleaned) > max_chars:
        raise QueryValidationError(f"q must be at most {max_chars} characters")
    return cleaned


def detect_security_flags(text: str) -> list[str]:
    flags: list[str] = []
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            flags.append("prompt_injection_text_detected")
            break
    if len(text) > 300:
        flags.append("long_query")
    for pattern in SUSPICIOUS_QUERY_PATTERNS:
        if pattern.search(text):
            flags.append("abuse_payload_text_detected")
            break
    return flags


def new_request_id() -> str:
    return uuid.uuid4().hex


def client_identifier(
    client_host: str | None,
    forwarded_for: str | None = None,
    real_ip: str | None = None,
) -> str:
    if real_ip:
        return real_ip.strip()
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return client_host or "unknown"


def api_key_is_valid(provided: str | None, expected: str | None) -> bool:
    if not expected:
        return False
    if not provided:
        return False
    return hmac.compare_digest(provided, expected)


def is_suspicious_user_agent(user_agent: str | None) -> bool:
    if not user_agent:
        return True
    return any(pattern.search(user_agent) for pattern in BOT_USER_AGENT_PATTERNS)


@dataclass(slots=True)
class TokenBucket:
    capacity: float
    refill_per_second: float
    tokens: float
    updated_at: float

    def allow(self, cost: float = 1.0) -> tuple[bool, float]:
        now = time.monotonic()
        elapsed = max(0.0, now - self.updated_at)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
        self.updated_at = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True, 0.0
        missing = cost - self.tokens
        wait = missing / self.refill_per_second if self.refill_per_second else 60.0
        return False, wait


@dataclass
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int = 0
    reason: str | None = None


@dataclass
class RateLimiter:
    rate_per_minute: int
    burst: int
    suspicious_rate_per_minute: int
    _buckets: dict[str, TokenBucket] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def check(self, identifier: str, *, suspicious: bool = False) -> RateLimitDecision:
        rate = self.suspicious_rate_per_minute if suspicious else self.rate_per_minute
        capacity = min(self.burst, max(1, rate))
        key = f"{'suspicious' if suspicious else 'normal'}:{identifier}"
        with self._lock:
            bucket = self._buckets.get(key)
            now = time.monotonic()
            if bucket is None:
                bucket = TokenBucket(
                    capacity=float(capacity),
                    refill_per_second=rate / 60.0,
                    tokens=float(capacity),
                    updated_at=now,
                )
                self._buckets[key] = bucket
            allowed, wait = bucket.allow()
            if allowed:
                return RateLimitDecision(True)
            return RateLimitDecision(
                False,
                retry_after_seconds=max(1, int(wait) + 1),
                reason="rate_limited_suspicious" if suspicious else "rate_limited",
            )


def security_headers(path: str = "/") -> dict[str, str]:
    if path.startswith("/docs") or path.startswith("/redoc"):
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data: https://fastapi.tiangolo.com; "
            "connect-src 'self'; "
            "frame-ancestors 'none'"
        )
    else:
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'"
        )
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        "Content-Security-Policy": csp,
    }
