from __future__ import annotations

import json
import logging
import time
from html import escape
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from app.cache import TTLCache
from app.config import Settings
from app.embeddings import build_embedder
from app.llm import OpenAIResolver
from app.metrics import ServiceMetrics
from app.models import ParseResult, QueryValidationError
from app.knowledge import KnowledgeBase
from app.parser import SearchParser
from app.security import (
    RateLimiter,
    api_key_is_valid,
    client_identifier,
    is_suspicious_user_agent,
    new_request_id,
    redact_query,
    security_headers,
)
from app.taxonomy import REAL_ESTATE, SECOND_HAND, VEHICLES, Taxonomy


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("yad2_parser")

settings = Settings.from_env()
taxonomy = Taxonomy.from_file(settings.taxonomy_path)
knowledge = KnowledgeBase.from_files(settings.enrichment_path, settings.learned_synonyms_path)
llm = (
    OpenAIResolver(
        settings.openai_api_key,
        settings.openai_model,
        max_output_tokens=settings.llm_max_output_tokens,
    )
    if settings.enable_llm_fallback and settings.openai_api_key
    else None
)
embedder = build_embedder(
    settings.embedding_backend,
    openai_api_key=settings.openai_api_key,
    openai_model=settings.openai_embedding_model,
)
parser = SearchParser(
    taxonomy,
    knowledge=knowledge,
    max_query_chars=settings.max_query_chars,
    cache=TTLCache(settings.cache_max_items, settings.cache_ttl_seconds),
    llm_fallback=llm,
    llm_confidence_threshold=settings.llm_confidence_threshold,
    vector_backend=settings.vector_backend,
    qdrant_url=settings.qdrant_url,
    qdrant_collection=settings.qdrant_collection,
    vector_top_k=settings.vector_top_k,
    embedder=embedder,
    include_debug_notes=settings.include_debug_notes,
)
metrics = ServiceMetrics()
rate_limiter = RateLimiter(
    rate_per_minute=settings.rate_limit_per_minute,
    burst=settings.rate_limit_burst,
    suspicious_rate_per_minute=settings.suspicious_rate_limit_per_minute,
)

app = FastAPI(
    title="Yad2 Hebrew Search Understanding",
    version=settings.app_version,
    description=(
        "Converts Hebrew Yad2 free-text searches into strict structured filters. "
        "The default path is deterministic and taxonomy-grounded; OpenAI fallback "
        "is optional and post-validated."
    ),
)

if settings.allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["content-type", "x-api-key", "x-request-id"],
    )


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or new_request_id()
    client_id = client_identifier(
        request.client.host if request.client else None,
        forwarded_for=request.headers.get("x-forwarded-for"),
        real_ip=request.headers.get("x-real-ip"),
    )
    user_agent = request.headers.get("user-agent")
    suspicious = is_suspicious_user_agent(user_agent)
    content_length = int(request.headers.get("content-length") or 0)

    if content_length > settings.max_request_bytes:
        _log_event(
            "request_rejected",
            request_id=request_id,
            client=client_id,
            reason="request_too_large",
            content_length=content_length,
        )
        response = JSONResponse(
            status_code=413,
            content={"detail": "request body too large", "request_id": request_id},
        )
        _apply_security_headers(response, request_id, request.url.path)
        return response

    protected_path = request.url.path in {"/parse", "/metrics", "/cache/clear"}
    if settings.require_api_key and protected_path:
        if not api_key_is_valid(request.headers.get("x-api-key"), settings.service_api_key):
            _log_event(
                "request_rejected",
                request_id=request_id,
                client=client_id,
                reason="invalid_api_key",
                path=request.url.path,
            )
            response = JSONResponse(
                status_code=401,
                content={"detail": "invalid api key", "request_id": request_id},
            )
            _apply_security_headers(response, request_id, request.url.path)
            return response

    limit = rate_limiter.check(client_id, suspicious=suspicious)
    if not limit.allowed:
        _log_event(
            "request_rejected",
            request_id=request_id,
            client=client_id,
            reason=limit.reason,
            user_agent=user_agent,
            retry_after_seconds=limit.retry_after_seconds,
        )
        response = JSONResponse(
            status_code=429,
            headers={"Retry-After": str(limit.retry_after_seconds)},
            content={
                "detail": "rate limit exceeded",
                "request_id": request_id,
                "retry_after_seconds": limit.retry_after_seconds,
            },
        )
        _apply_security_headers(response, request_id, request.url.path)
        return response

    request.state.request_id = request_id
    request.state.client_id = client_id
    request.state.suspicious_client = suspicious
    response = await call_next(request)
    _apply_security_headers(response, request_id, request.url.path)
    return response


class ParseRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=settings.max_query_chars)


class Resolution(BaseModel):
    source: str
    model: str | None = None
    cache_hit: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class ParseResponse(BaseModel):
    category: Literal["נדל״ן", "רכב", "יד_שנייה"]
    params: dict[str, Any]
    confidence: float = Field(..., ge=0, le=1)
    notes: list[str] | None = None
    resolution: Resolution | None = None


@app.post("/parse", response_model=ParseResponse)
def parse(request: ParseRequest, http_request: Request) -> dict[str, Any]:
    started = time.perf_counter()
    request_id = getattr(http_request.state, "request_id", new_request_id())
    try:
        result = parser.parse(request.q)
    except QueryValidationError as exc:
        latency_ms = _elapsed_ms(started)
        metrics.record_error(latency_ms)
        _log_event(
            "parse_rejected",
            request_id=request_id,
            query=redact_query(request.q),
            latency_ms=latency_ms,
            error=str(exc),
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - API safety net
        latency_ms = _elapsed_ms(started)
        metrics.record_error(latency_ms)
        _log_event(
            "parse_error",
            request_id=request_id,
            query=redact_query(request.q),
            latency_ms=latency_ms,
            error=exc.__class__.__name__,
        )
        raise HTTPException(status_code=500, detail="parse failed") from exc

    latency_ms = _elapsed_ms(started)
    metrics.record_success(result, latency_ms)
    _log_parse_success(
        request.q,
        result,
        latency_ms,
        request_id=request_id,
        client_id=getattr(http_request.state, "client_id", None),
    )
    return result.public_dict()


@app.post("/cache/clear")
def clear_cache(http_request: Request) -> dict[str, Any]:
    request_id = getattr(http_request.state, "request_id", new_request_id())
    cleared = parser.cache.clear()
    _log_event(
        "cache_cleared",
        request_id=request_id,
        client=getattr(http_request.state, "client_id", None),
        cleared_entries=cleared,
    )
    return {"cleared_entries": cleared}


@app.get("/health")
def health(request: Request):
    payload = {
        "status": "ok",
        "version": settings.app_version,
        "taxonomy_version": taxonomy.data.get("גרסה"),
        "llm_fallback_enabled": settings.enable_llm_fallback,
        "api_key_required": settings.require_api_key,
        "demo_ui_enabled": settings.enable_demo_ui,
        "vector_backend": settings.vector_backend,
        "qdrant_collection": settings.qdrant_collection,
        "embedding_backend": settings.embedding_backend,
    }
    if _wants_html(request):
        rows = "".join(
            f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
            for key, value in payload.items()
        )
        return HTMLResponse(
            _page(
                "Health",
                "Runtime status and deployment flags",
                f"""
                <section class="panel">
                  <div class="status-large ok">OK</div>
                  <table class="kv">{rows}</table>
                </section>
                """,
            )
        )
    return payload


@app.get("/metrics")
def prometheus_metrics(request: Request):
    rendered = metrics.render_prometheus()
    if _wants_html(request):
        parsed = _parse_metrics(rendered)
        cards = "".join(
            f"""
            <article class="metric-card">
              <div class="metric-name">{escape(item["name"])}</div>
              <div class="metric-value">{escape(item["value"])}</div>
              <div class="metric-label">{escape(item["label"])}</div>
            </article>
            """
            for item in parsed[:8]
        )
        return HTMLResponse(
            _page(
                "Metrics",
                "Human view of the Prometheus endpoint",
                f"""
                <section class="metric-grid">{cards}</section>
                <section class="panel">
                  <h2>Prometheus Output</h2>
                  <pre>{escape(rendered)}</pre>
                </section>
                """,
            )
        )
    return PlainTextResponse(rendered)


@app.get("/", response_class=HTMLResponse)
def demo() -> str:
    if not settings.enable_demo_ui:
        raise HTTPException(status_code=404, detail="demo disabled")
    return DEMO_HTML


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _log_parse_success(
    query: str,
    result: ParseResult,
    latency_ms: float,
    *,
    request_id: str,
    client_id: str | None,
) -> None:
    _log_event(
        "parse_success",
        request_id=request_id,
        client=client_id,
        query=redact_query(query),
        category=result.category,
        source=result.source,
        confidence=round(result.confidence, 3),
        latency_ms=round(latency_ms, 3),
        cache_hit=result.cache_hit,
        params=list(result.params.keys()),
        model=result.model_name,
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        cost_usd=round(result.usage.estimated_cost_usd, 8),
    )


def _log_event(event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}, ensure_ascii=False))


def _wants_html(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return "text/html" in accept and "application/json" not in accept


def _parse_metrics(rendered: str) -> list[dict[str, str]]:
    interesting_prefixes = (
        "yad2_requests_total",
        "yad2_errors_total",
        "yad2_cache_hit_ratio",
        "yad2_cost_usd_per_request",
        "yad2_uptime_seconds",
        "yad2_request_latency_ms",
    )
    output: list[dict[str, str]] = []
    for line in rendered.splitlines():
        if not line or line.startswith("#"):
            continue
        if not line.startswith(interesting_prefixes):
            continue
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        name = parts[0]
        label = ""
        if "{" in name and "}" in name:
            base = name.split("{", 1)[0]
            label = name.split("{", 1)[1].split("}", 1)[0].replace('"', "")
            name = base
        output.append({"name": name, "label": label or "current", "value": parts[1]})
    return output


def _page(title: str, subtitle: str, body: str) -> str:
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(title)} | Yad2 Parser</title>
  <style>{BASE_CSS}</style>
</head>
<body>
  <main>
    <nav class="topbar">
      <a class="brand" href="/">Yad2 Parser</a>
      <div class="navlinks">
        <a href="/docs">Swagger</a>
        <a href="/health">Health</a>
        <a href="/metrics">Metrics</a>
      </div>
    </nav>
    <header class="page-header">
      <h1>{escape(title)}</h1>
      <p>{escape(subtitle)}</p>
    </header>
    {body}
  </main>
</body>
</html>
"""


def _apply_security_headers(response, request_id: str, path: str = "/") -> None:
    for key, value in security_headers(path).items():
        response.headers[key] = value
    response.headers["X-Request-ID"] = request_id


BASE_CSS = """
:root {
  color-scheme: light;
  --ink: #18212f;
  --muted: #5b6472;
  --line: #d8dee8;
  --panel: #ffffff;
  --bg: #f5f7fb;
  --accent: #0f766e;
  --accent-2: #b45309;
  --danger: #b42318;
  font-family: Arial, "Noto Sans Hebrew", sans-serif;
}
body {
  margin: 0;
  background: radial-gradient(circle at 20% 0%, #e9f5f3 0, transparent 28rem), var(--bg);
  color: var(--ink);
}
main {
  width: min(1100px, calc(100% - 32px));
  margin: 24px auto 40px;
  display: grid;
  gap: 16px;
}
.topbar {
  min-height: 54px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(255,255,255,0.88);
  padding: 0 12px;
}
.brand {
  color: var(--ink);
  font-weight: 800;
  text-decoration: none;
}
.navlinks {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.navlinks a, .link-button {
  min-height: 36px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0 12px;
  color: var(--ink);
  background: #fff;
  text-decoration: none;
}
.navlinks a:hover, .link-button:hover {
  border-color: var(--accent);
}
.page-header {
  display: grid;
  gap: 6px;
}
.page-header h1 {
  margin: 0;
  font-size: 30px;
  line-height: 1.18;
}
.page-header p {
  margin: 0;
  color: var(--muted);
}
.panel {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  padding: 16px;
}
.panel h2 {
  margin: 0 0 12px;
  font-size: 18px;
}
.kv {
  width: 100%;
  border-collapse: collapse;
  overflow: hidden;
}
.kv th, .kv td {
  text-align: left;
  border-bottom: 1px solid var(--line);
  padding: 10px;
}
.kv th {
  color: var(--muted);
  width: 260px;
}
.status-large {
  display: inline-flex;
  align-items: center;
  min-height: 42px;
  border-radius: 8px;
  padding: 0 14px;
  margin-bottom: 12px;
  font-weight: 800;
}
.status-large.ok {
  background: #e7f7f1;
  color: #087443;
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}
.metric-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  padding: 14px;
  min-height: 96px;
}
.metric-name {
  color: var(--muted);
  font-size: 12px;
  word-break: break-word;
}
.metric-value {
  margin-top: 10px;
  font-size: 24px;
  font-weight: 800;
}
.metric-label {
  margin-top: 6px;
  color: var(--accent-2);
  font-size: 12px;
}
pre {
  margin: 0;
  direction: ltr;
  text-align: left;
  white-space: pre-wrap;
  overflow: auto;
  background: #101827;
  color: #eef5ff;
  border-radius: 8px;
  padding: 14px;
  font-size: 14px;
  line-height: 1.5;
}
@media (max-width: 820px) {
  main {
    width: min(100% - 20px, 1100px);
    margin: 12px auto 24px;
  }
  .topbar {
    align-items: flex-start;
    flex-direction: column;
    padding: 10px;
  }
  .metric-grid {
    grid-template-columns: 1fr;
  }
}
"""


DEMO_HTML = """
<!doctype html>
<html lang="he" dir="rtl">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Yad2 Search Parser</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #18212f;
      --muted: #5b6472;
      --line: #d8dee8;
      --panel: #ffffff;
      --bg: #f5f7fb;
      --accent: #0f766e;
      --accent-2: #b45309;
      font-family: Arial, "Noto Sans Hebrew", sans-serif;
    }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
    }
    main {
      width: min(1040px, calc(100% - 32px));
      margin: 32px auto;
      display: grid;
      gap: 16px;
    }
    .topbar {
      min-height: 54px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,0.9);
      padding: 0 12px;
    }
    .brand {
      color: var(--ink);
      font-weight: 800;
      text-decoration: none;
    }
    .navlinks {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .navlinks a {
      min-height: 36px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0 12px;
      color: var(--ink);
      background: #fff;
      text-decoration: none;
    }
    .navlinks a:hover {
      border-color: var(--accent);
    }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }
    h1 {
      margin: 0;
      font-size: 28px;
      line-height: 1.2;
    }
    .subtitle {
      color: var(--muted);
      margin-top: 6px;
    }
    .status {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 10px 12px;
      min-width: 190px;
    }
    .workspace {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(360px, 0.9fr);
      gap: 16px;
    }
    section {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 16px;
    }
    label {
      display: block;
      font-weight: 700;
      margin-bottom: 8px;
    }
    textarea {
      box-sizing: border-box;
      width: 100%;
      min-height: 120px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      font: inherit;
      color: var(--ink);
    }
    .actions {
      display: flex;
      gap: 8px;
      margin-top: 12px;
      flex-wrap: wrap;
    }
    button {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      min-height: 38px;
      padding: 0 12px;
      cursor: pointer;
      font: inherit;
    }
    button.primary {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
      font-weight: 700;
    }
    .examples {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
    }
    .examples button {
      color: var(--muted);
      max-width: 100%;
    }
    pre {
      min-height: 260px;
      margin: 0;
      direction: ltr;
      text-align: left;
      white-space: pre-wrap;
      overflow: auto;
      background: #101827;
      color: #eef5ff;
      border-radius: 8px;
      padding: 14px;
      font-size: 14px;
      line-height: 1.5;
    }
    .meta {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }
    .pill {
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 4px 9px;
      color: var(--muted);
      background: #fafcff;
    }
    .pill strong {
      color: var(--accent-2);
    }
    button.danger {
      border-color: var(--accent-2);
      color: var(--accent-2);
    }
    .res-title {
      margin: 0 0 8px;
      font-size: 14px;
      color: var(--muted);
    }
    .resolution {
      display: grid;
      gap: 6px;
      margin-bottom: 12px;
    }
    .res-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 6px 10px;
      background: #fafcff;
      font-size: 13px;
    }
    .res-key {
      color: var(--muted);
      font-weight: 700;
    }
    .res-val {
      color: var(--ink);
      text-align: left;
      direction: ltr;
    }
    .res-row.source .res-val {
      color: var(--accent);
      font-weight: 700;
    }
    @media (max-width: 820px) {
      main {
        width: min(100% - 20px, 1040px);
        margin: 16px auto;
      }
      .workspace {
        grid-template-columns: 1fr;
      }
      .quick-links {
        grid-template-columns: 1fr;
      }
      .topbar {
        align-items: flex-start;
        flex-direction: column;
        padding: 10px;
      }
    }
  </style>
</head>
<body>
  <main>
    <nav class="topbar">
      <a class="brand" href="/">Yad2 Parser</a>
      <div class="navlinks">
        <a href="/docs">Swagger</a>
        <a href="/health">Health</a>
        <a href="/metrics">Metrics</a>
      </div>
    </nav>
    <header>
      <div>
        <h1>Yad2 Hebrew Search Parser</h1>
        <div class="subtitle">Taxonomy-grounded parsing with optional OpenAI fallback</div>
      </div>
      <div class="status" id="health">בודק סטטוס...</div>
    </header>
    <div class="workspace">
      <section>
        <label for="query">שאילתת חיפוש</label>
        <textarea id="query">דירת 3 חדרים בירושלים עד מליון שח עם מעלית וחניה</textarea>
        <div class="actions">
          <button class="primary" id="parse">Parse</button>
          <button id="clear">Clear</button>
          <button class="danger" id="clearCache" title="Clears the in-memory result cache so the next query is recomputed">Clear cache &amp; re-run</button>
        </div>
        <div class="examples" id="examples"></div>
      </section>
      <section>
        <div class="meta">
          <span class="pill">Latency: <strong id="latency">-</strong></span>
          <span class="pill">Confidence: <strong id="confidence">-</strong></span>
          <span class="pill">Category: <strong id="category">-</strong></span>
        </div>
        <h2 class="res-title">How this result was produced</h2>
        <div id="resolution" class="resolution"></div>
        <pre id="output">{}</pre>
      </section>
    </div>
  </main>
  <script>
    const examples = [
      "דירת 3 חדרים בירושלים עד מליון שח",
      "טויוטה קורולה 2018-2021 עד 70 אלף שח צבע לבן",
      "אייפון 13 פרו 256 גיגה כחול כמו חדש עד 2500",
      "ספה עור לסלון במרכז עד 1200",
      "יונדאי טוסון יד 2 עד 90 אלף קמ אוטומטית"
    ];
    examples.push(
      "\u05e8\u05db\u05d1 \u05d7\u05e1\u05db\u05d5\u05e0\u05d9 \u05dc\u05e1\u05d8\u05d5\u05d3\u05e0\u05d8 \u05e2\u05d3 40000",
      "\u05de\u05d7\u05e9\u05d1 \u05dc\u05dc\u05d9\u05de\u05d5\u05d3\u05d9\u05dd \u05dc\u05d0 \u05d9\u05e7\u05e8",
      "\u05e7\u05d5\u05e8\u05e7\u05d9\u05e0\u05d8 \u05dc\u05d9\u05dc\u05d3 \u05e2\u05d3 1000",
      "\u05e8\u05db\u05d1 \u05d0\u05de\u05e8\u05d9\u05e7\u05d0\u05d9 \u05d2\u05d3\u05d5\u05dc"
    );
    const q = document.getElementById("query");
    const out = document.getElementById("output");
    const latency = document.getElementById("latency");
    const confidence = document.getElementById("confidence");
    const category = document.getElementById("category");
    const resolutionEl = document.getElementById("resolution");
    const examplesEl = document.getElementById("examples");
    const engines = {vector: "-", embedding: "-", llm: false};
    examples.forEach((example) => {
      const button = document.createElement("button");
      button.textContent = example;
      button.onclick = () => { q.value = example; parse(); };
      examplesEl.appendChild(button);
    });
    function describeSource(src, resolution) {
      if (src === "cache") return "Cache \u2014 served a previously computed answer";
      if (src === "llm") return "LLM fallback \u2014 OpenAI model resolved it";
      if (resolution && resolution.model) return "Rules kept after LLM fallback \u2014 model was consulted, validator kept the safer result";
      return "Rules engine \u2014 deterministic retrieval + extraction";
    }
    function renderResolution(body) {
      const r = (body && body.resolution) || {};
      const rows = [];
      rows.push(["source", describeSource(r.source, r), "source"]);
      if (r.cache_hit) rows.push(["cache hit", "yes", ""]);
      if (r.model) rows.push(["model", r.model, ""]);
      if (r.input_tokens || r.output_tokens)
        rows.push(["tokens", `${r.input_tokens} in / ${r.output_tokens} out`, ""]);
      if (r.cost_usd) rows.push(["est. cost", `$${r.cost_usd}`, ""]);
      rows.push(["vector engine", engines.vector, ""]);
      rows.push(["embedding engine", engines.embedding, ""]);
      resolutionEl.innerHTML = rows.map(([k, v, cls]) =>
        `<div class="res-row ${cls}"><span class="res-key">${k}</span><span class="res-val">${v}</span></div>`
      ).join("");
    }
    async function parse() {
      const started = performance.now();
      out.textContent = "Parsing...";
      resolutionEl.innerHTML = "";
      const response = await fetch("/parse", {
        method: "POST",
        headers: {"content-type": "application/json"},
        body: JSON.stringify({q: q.value})
      });
      const body = await response.json();
      latency.textContent = `${Math.round(performance.now() - started)}ms`;
      confidence.textContent = body.confidence ?? "-";
      category.textContent = body.category ?? "-";
      renderResolution(body);
      out.textContent = JSON.stringify(body, null, 2);
    }
    async function clearCache() {
      const btn = document.getElementById("clearCache");
      const original = btn.textContent;
      btn.disabled = true;
      btn.textContent = "Clearing...";
      try {
        const response = await fetch("/cache/clear", {method: "POST"});
        const body = await response.json();
        btn.textContent = `Cleared ${body.cleared_entries} \u2713`;
      } catch (err) {
        btn.textContent = "Clear failed";
      }
      setTimeout(() => { btn.textContent = original; btn.disabled = false; }, 1200);
      await parse();
    }
    async function health() {
      const response = await fetch("/health");
      const body = await response.json();
      engines.vector = body.vector_backend || "-";
      engines.embedding = body.embedding_backend || "-";
      engines.llm = !!body.llm_fallback_enabled;
      document.getElementById("health").textContent =
        `status: ${body.status} | llm: ${body.llm_fallback_enabled ? "on" : "off"}`;
    }
    document.getElementById("parse").onclick = parse;
    document.getElementById("clear").onclick = () => { q.value = ""; out.textContent = "{}"; resolutionEl.innerHTML = ""; };
    document.getElementById("clearCache").onclick = clearCache;
    (async () => { await health(); await parse(); })();
  </script>
</body>
</html>
"""
