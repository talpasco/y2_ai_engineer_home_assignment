# Yad2 Free-Text Search Understanding

Dockerized search-understanding service for converting Hebrew Yad2 free-text queries into strict structured filters.

The implementation is intentionally **not an LLM wrapper**. The default path is a deterministic, taxonomy-grounded parser with hybrid retrieval, normalization, validation, caching, metrics, and abuse controls. OpenAI is implemented as an optional low-confidence fallback, but is disabled by default so the system is cheap, reproducible, and safe to run during review.

## What Was Built

The service accepts a query such as:

```json
{ "q": "דירת 3 חדרים בירושלים עד מליון שח" }
```

and returns:

```json
{
  "category": "נדל״ן",
  "params": {
    "סוגי_נכס": ["דירה"],
    "עיר": "ירושלים",
    "מס׳_חדרים": 3,
    "מחיר": { "max": 1000000 }
  },
  "confidence": 0.88
}
```

The response contract is strict: `category`, `params`, `confidence`, optional `notes`, and a small `resolution` block for reviewer-facing traceability. `resolution` shows whether the result came from rules, cache, or LLM fallback, plus model/token/cost data when relevant. Debug match evidence is not exposed by default; set `INCLUDE_DEBUG_NOTES=true` to surface per-field evidence in `notes` during debugging.

Supported verticals:

- `נדל״ן`
- `רכב`
- `יד_שנייה`

The output is always validated against the provided taxonomy and enrichment allowlists before being returned.

## Quick Start

Run the production-like local stack:

```bash
docker compose up --build
```

Open:

```text
http://localhost:8000
http://localhost:8000/docs
http://localhost:8000/health
http://localhost:8000/metrics
http://localhost:6333/dashboard
```

Run the full Docker smoke test:

```bash
python scripts/docker_smoke.py
```

Run locally without Docker:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Run verification:

```bash
python -m unittest discover -v
python scripts/evaluate_examples.py
python scripts/run_redteam.py
```

Current verified results:

- `36/36` unit and API tests pass.
- `10/10` labeled examples pass (category and field accuracy `1.0`).
- `8/8` red-team/security cases pass.
- Docker Compose starts the API plus Qdrant and successfully serves `/parse`, `/health`, `/metrics`, and `/docs`.

## API

### `POST /parse`

Request:

```json
{ "q": "טויוטה קורולה 2018-2021 עד 70 אלף שח צבע לבן" }
```

Response shape:

```json
{
  "category": "רכב",
  "params": {
    "יצרן": "טויוטה",
    "דגם": "קורולה",
    "שנה": { "min": 2018, "max": 2021 },
    "מחיר": { "max": 70000 },
    "צבע": "לבן"
  },
  "confidence": 0.915
}
```

### `GET /health`

Returns app status, taxonomy version, LLM fallback status, API-key mode, UI mode, and vector backend.

### `GET /metrics`

Returns Prometheus-style metrics for requests, latency, cache hit ratio, category distribution, model calls, token usage, and estimated cost.

Set `REQUIRE_API_KEY=true` and `SERVICE_API_KEY=<shared-secret>` to require an
`x-api-key` header for protected endpoints such as `/parse`, `/metrics`, and
`/cache/clear`. The demo keeps this off by default so reviewers can run it
locally without provisioning secrets.

## Architecture

```text
HTTP request
  -> request limits, API key check, abuse heuristics
  -> Hebrew and unit normalization
  -> normalized-query cache lookup
  -> taxonomy, alias, fuzzy, and vector candidate retrieval
  -> vertical-specific extraction
  -> confidence scoring
  -> optional OpenAI Structured Outputs fallback
  -> strict schema validation
  -> structured response, metrics, and redacted logs
```

Core modules:

- `app/main.py` - FastAPI app, UI, `/parse`, `/health`, `/metrics`.
- `app/parser.py` - category scoring, extraction, confidence, fallback orchestration.
- `app/taxonomy.py` - taxonomy loading and output-field validation.
- `app/normalizer.py` - Hebrew text, typo, unit, range, and amount normalization.
- `app/retrieval.py` - exact and fuzzy candidate retrieval.
- `app/embeddings.py` - pluggable embedding engine (`HashedEmbedder` default, optional `OpenAIEmbedder`).
- `app/vector_retrieval.py` - in-process vector retrieval and optional Qdrant retrieval over the embedding seam.
- `app/knowledge.py` - enrichment layer for locations, trims, multilingual aliases, and learned synonyms.
- `app/security.py` - input limits, rate limiting, prompt-injection checks, headers, redaction.
- `app/llm.py` - optional OpenAI fallback using schema-constrained output.
- `app/metrics.py` - Prometheus-style metrics.

## Why This Design

Yad2 free-text search is mostly a **structured understanding** problem, not a general chat problem. Common searches are short, high-volume, and map to known filters. A production solution should therefore optimize for precision, latency, and cost first, then use models selectively where they improve recall.

This implementation uses a hybrid approach:

- deterministic rules for stable, cheap extraction of known filters
- fuzzy matching for typos and compact alias drift
- vector retrieval for long-tail aliases, transliteration, and semantic candidate recall
- strict validation so neither fuzzy retrieval nor an LLM can invent unsupported output
- optional OpenAI fallback only for low-confidence or ambiguous queries
- fallback guardrails: default threshold `0.60`, no LLM call for clear category-only queries, minimal reasoning, `LLM_MAX_OUTPUT_TOKENS=700`, and rejection of broad taxonomy-echo outputs

The Docker Compose stack includes Qdrant because a real Yad2-scale system needs a retrieval layer for long-tail language, location names, vehicle trims, and second-hand marketplace phrases.

### ML Engine: Pluggable Embeddings

Semantic retrieval runs behind a model-agnostic `Embedder` contract (`app/embeddings.py`), with two backends:

- **`HashedEmbedder` (default)** - deterministic hashed character n-gram vectors. No model download, fully reproducible, strong on spelling/transliteration overlap. This keeps local runs, Docker, and tests self-contained.
- **`OpenAIEmbedder` (opt-in via `EMBEDDING_BACKEND=openai`)** - real learned embeddings (`text-embedding-3-small`), L2-normalized and LRU-cached in-process so taxonomy vectors and popular queries are embedded at most once.

Both implement `embed(text) -> vector` and expose `dim`, so the parser, the in-process `LocalVectorRetriever`, and the `QdrantVectorRetriever` are unchanged when the backend is swapped (the Qdrant collection size is taken from the embedder). In production this same seam can host a self-hosted multilingual/Hebrew encoder (E5 / bge-m3) served from an inference container - again with no parser changes.

The full rationale for every major choice, plus the production integration and rollout considerations, is in `docs/design_decisions.md`.

## Correctness & Robustness

Implemented:

- category classification across real estate, vehicles, and second-hand items
- vertical-specific extraction for price, rooms, city, property type, manufacturer, model, year, hand, km, fuel, gearbox, color, brand, storage, RAM, condition, region, and item type
- Hebrew typo normalization and unit normalization
- numeric amount handling such as `70 אלף`, `מליון`, `עד`, `מעל`, and ranges
- long-tail city, street, and neighborhood normalization through `data/enrichment_aliases.json`
- vehicle trim dictionaries, including submodel extraction outside the compact taxonomy
- second-hand phrases that are not present in the provided taxonomy
- multilingual aliases and transliteration, for example English vehicle and device queries
- learned-synonym support from anonymized query logs via `data/learned_synonyms.json`
- strict schema adherence through taxonomy validation before response
- bounded confidence scoring and low-confidence fallback path

Validation assets:

- `examples/sample_queries.json` - labeled example set
- `examples/redteam_queries.json` - adversarial/security cases
- `tests/` - parser, API, security, and vector retrieval tests

## Design Quality

Latency targets and how they are met:

| Path | Target | This implementation |
| --- | --- | --- |
| cache / rules only | p95 ≤ 150ms | in-process parse, no external call; measured ~0.4ms/parse warm on a dev laptop |
| model path (LLM fallback) | tail path, not hot-path SLA | one bounded OpenAI call on genuinely low-confidence queries only; minimal reasoning, output capped, cached, measured, and post-validated locally |

- the common path is in-process and does not call an external model
- taxonomy, enrichment, and retrieval indexes are loaded/built once at startup
- cache hits return immediately from the normalized-query cache
- external-model latency is provider-dependent, so fallback is treated as a
  controlled recall path rather than the common serving path
- throughput: a single instance comfortably exceeds the ≥12 QPS / ~1M-per-day target on the rules path; 10M/month is ~3.9 QPS average

Cost:

- default model spend is `$0` (LLM fallback disabled by default)
- OpenAI fallback is optional and can be gated by confidence, category, rate limits, and budget
- fallback routing is intentionally conservative: clear category-only queries such as "רכב אמריקאי גדול" stay on the rules path instead of spending model tokens
- token usage and estimated model cost are exposed in `/metrics`

### Cost Model (10M queries / month)

Assumptions: gpt-5-nano fallback pricing as configured in `app/llm.py` (`$0.05` / 1M input tokens, `$0.40` / 1M output tokens), a compact single-category taxonomy slice of ~2,500 input tokens and ~200 output tokens per fallback call (`≈ $0.000205` per call), and that the deterministic rules + cache path serves the vast majority of traffic at `$0` model cost.

| Scenario | Cache hit ratio | LLM fallback rate | LLM calls / month | Model cost / month | Blended cost / query |
| --- | --- | --- | --- | --- | --- |
| Default (fallback off) | ~40% | 0% | 0 | **$0** | **$0** |
| Conservative | ~40% | 5% of traffic | 500K | **~$103** | **~$0.00001** |
| Aggressive recall | ~30% | 20% of traffic | 2M | **~$410** | **~$0.00004** |

Even in the aggressive scenario the model cost is dominated by selective fallback and stays well under `$0.0001` per query; the rules/cache hot path is effectively free apart from commodity compute. Compute is trivial: 10M/month averages ~3.9 QPS, served by 1-2 small stateless instances. The credible path to "very low cost per query at Yad2 scale" is therefore: serve most traffic deterministically, cache aggressively, and spend tokens only on the ambiguous tail.

Scale:

- API instances are stateless except for local warm caches
- Qdrant is separated as retrieval infrastructure
- cache can be moved from local LRU to Redis, Dragonfly, or another shared cache
  or edge KV
- taxonomy/enrichment data can be versioned and promoted through offline evaluation

Trade-off:

- rules alone are cheap and precise but miss long-tail phrasing
- LLM-only parsing is flexible but expensive, slower, and harder to secure
- the chosen hybrid design keeps the hot path cheap while reserving AI/model use for cases where it is likely to add value

## Observability & Security

Metrics include:

- total requests and errors
- p50/p95 latency
- cache hit ratio
- requests by category
- requests by source: `rules`, `cache`, or `llm`
- model call success/failure
- input/output tokens
- estimated cost per request
- uptime

Logs are structured and include request ID, source, category, confidence, latency, cache status, extracted fields, model usage, and cost. Phone/email-like values are redacted before logging.

Security controls:

- max query length and request-size limits
- per-client token-bucket rate limiting
- stricter throttling for suspicious user agents
- optional API key protection with constant-time comparison
- prompt-injection and abuse-pattern detection
- LLM fallback disabled for suspicious inputs
- strict schema validation after both rules and LLM paths
- no hardcoded secrets
- security headers on browser responses
- configurable CORS
- red-team tests for prompt injection, unicode/control noise, SQL-like payloads, bot-like usage, and field confusion

The model, when enabled, is never trusted as the source of truth. It can only propose structured values, and the same validator decides what is allowed.

## Code Quality

The code is organized by responsibility rather than by framework layer. The parser, taxonomy validation, retrieval, vector backend, security, LLM fallback, and metrics are separately testable.

Included:

- Dockerfile
- Docker Compose with API + Qdrant
- unit/API/security/vector tests
- example evaluator
- red-team evaluator
- synonym-mining example from anonymized logs
- OpenAPI docs through FastAPI
- human-readable local UI for review
- additional design/security notes in `docs/`

Useful docs:

- `docs/design_decisions.md` - rationale for every major decision, production integration plan, and taxonomy scope/gap analysis
- `docs/security_model.md` - OWASP-based threat model and controls
- `docs/coverage_matrix.md` - assignment requirement → file mapping
- `docs/dashboard.json` + `docs/metrics_snapshot.txt` - observability snapshot

## Pragmatism: Low Cost at Yad2 Scale

The credible low-cost path is:

1. Use deterministic parsing, normalization, cache, and retrieval for the majority of traffic.
2. Use Qdrant or another retrieval service for candidate recall over aliases, locations, trims, and learned synonyms.
3. Call an LLM only when confidence is low, the query is not suspicious, and the request is within budget.
4. Cache normalized query results and high-quality fallback outputs.
5. Continuously mine anonymized query logs for synonyms, missed filters, and zero-result patterns.
6. Promote new aliases only after offline evals and human review.

This gives a path to very low marginal cost per query:

- common query: in-process parse plus cache/retrieval
- ambiguous query: selective LLM fallback
- repeated query: cache hit
- model cost: measured and bounded through metrics, fallback thresholds, and quotas

At Yad2 volume, this is more realistic than an always-LLM parser. It also creates a feedback loop where model usage can shrink over time as learned synonyms and normalization rules improve.

## OpenAI Usage

OpenAI is optional:

```env
ENABLE_LLM_FALLBACK=false
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5-nano
LLM_CONFIDENCE_THRESHOLD=0.60
LLM_MAX_OUTPUT_TOKENS=700
```

When enabled, the service uses schema-constrained output and validates the result against the same taxonomy rules. The fallback is intended for ambiguous or low-confidence queries, not for the common hot path. It is also budget-constrained: minimal reasoning, capped output, and parser-side rejection of broad taxonomy defaults.

## Known Limitations

- The included eval set is intentionally small for the home assignment and should be expanded with anonymized real query logs.
- Hebrew morphology is handled pragmatically through aliases and normalization, not a full morphological analyzer.
- The enrichment dictionaries are representative, not exhaustive.
- The current local embeddings are deterministic hashed vectors for reproducibility; production should evaluate learned Hebrew/multilingual embeddings.
