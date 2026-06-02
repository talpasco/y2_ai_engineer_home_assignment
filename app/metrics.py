from __future__ import annotations

import time
from collections import Counter, deque
from threading import Lock

from app.models import ParseResult


class ServiceMetrics:
    def __init__(self, max_latency_samples: int = 10_000) -> None:
        self._lock = Lock()
        self.started_at = time.time()
        self.requests_total = 0
        self.errors_total = 0
        self.category_total: Counter[str] = Counter()
        self.source_total: Counter[str] = Counter()
        self.model_success_total = 0
        self.model_failure_total = 0
        self.cache_hit_total = 0
        self.input_tokens_total = 0
        self.output_tokens_total = 0
        self.cost_usd_total = 0.0
        self.latencies_ms: deque[float] = deque(maxlen=max_latency_samples)

    def record_success(self, result: ParseResult, latency_ms: float) -> None:
        with self._lock:
            self.requests_total += 1
            self.category_total[result.category] += 1
            self.source_total[result.source] += 1
            self.latencies_ms.append(latency_ms)
            if result.cache_hit:
                self.cache_hit_total += 1
            if result.model_call_success:
                self.model_success_total += 1
            if result.model_call_failure:
                self.model_failure_total += 1
            self.input_tokens_total += result.usage.input_tokens
            self.output_tokens_total += result.usage.output_tokens
            self.cost_usd_total += result.usage.estimated_cost_usd

    def record_error(self, latency_ms: float) -> None:
        with self._lock:
            self.requests_total += 1
            self.errors_total += 1
            self.latencies_ms.append(latency_ms)

    def record_model_failure(self) -> None:
        with self._lock:
            self.model_failure_total += 1

    def render_prometheus(self) -> str:
        with self._lock:
            latencies = sorted(self.latencies_ms)
            p50 = _percentile(latencies, 0.50)
            p95 = _percentile(latencies, 0.95)
            cache_ratio = (
                self.cache_hit_total / self.requests_total if self.requests_total else 0.0
            )
            avg_cost = (
                self.cost_usd_total / self.requests_total if self.requests_total else 0.0
            )
            lines = [
                "# HELP yad2_requests_total Total parse requests.",
                "# TYPE yad2_requests_total counter",
                f"yad2_requests_total {self.requests_total}",
                "# HELP yad2_errors_total Total parse errors.",
                "# TYPE yad2_errors_total counter",
                f"yad2_errors_total {self.errors_total}",
                "# HELP yad2_request_latency_ms Request latency percentile gauges.",
                "# TYPE yad2_request_latency_ms gauge",
                f'yad2_request_latency_ms{{quantile="0.50"}} {p50:.3f}',
                f'yad2_request_latency_ms{{quantile="0.95"}} {p95:.3f}',
                "# HELP yad2_cache_hit_ratio Ratio of requests served from cache.",
                "# TYPE yad2_cache_hit_ratio gauge",
                f"yad2_cache_hit_ratio {cache_ratio:.6f}",
                "# HELP yad2_token_usage_total Token usage by type.",
                "# TYPE yad2_token_usage_total counter",
                f'yad2_token_usage_total{{type="input"}} {self.input_tokens_total}',
                f'yad2_token_usage_total{{type="output"}} {self.output_tokens_total}',
                "# HELP yad2_cost_usd_total Estimated OpenAI cost in USD.",
                "# TYPE yad2_cost_usd_total counter",
                f"yad2_cost_usd_total {self.cost_usd_total:.8f}",
                "# HELP yad2_cost_usd_per_request Average estimated cost in USD.",
                "# TYPE yad2_cost_usd_per_request gauge",
                f"yad2_cost_usd_per_request {avg_cost:.10f}",
                "# HELP yad2_model_calls_total Model call counts.",
                "# TYPE yad2_model_calls_total counter",
                f'yad2_model_calls_total{{status="success"}} {self.model_success_total}',
                f'yad2_model_calls_total{{status="failure"}} {self.model_failure_total}',
                "# HELP yad2_uptime_seconds Service uptime in seconds.",
                "# TYPE yad2_uptime_seconds gauge",
                f"yad2_uptime_seconds {time.time() - self.started_at:.3f}",
            ]
            for category, count in sorted(self.category_total.items()):
                lines.append(f'yad2_requests_by_category_total{{category="{category}"}} {count}')
            for source, count in sorted(self.source_total.items()):
                lines.append(f'yad2_requests_by_source_total{{source="{source}"}} {count}')
            return "\n".join(lines) + "\n"


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, round((len(values) - 1) * percentile))
    return values[index]
