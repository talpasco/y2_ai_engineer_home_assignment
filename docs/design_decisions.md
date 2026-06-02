# Design Decisions & Rationale

This document is the "why" behind the implementation. Each decision lists the
problem, the options I considered, what I chose, and the trade-off.

---

## 1. Why not just wrap an LLM?

**Problem.** Convert untrusted Hebrew free-text into strict, taxonomy-valid
search filters, at Yad2 scale (tens of millions of searches/month), cheaply,
fast, and safely.

**Options.**
- (A) LLM-only: send every query to a model, parse JSON out.
- (B) Rules-only: regex/lexicon parsing.
- (C) Hybrid: deterministic retrieval + extraction first, LLM only as a gated
  fallback.

**Decision: (C) Hybrid, LLM disabled by default.**

**Why.**
- **Cost.** At 10M queries/month, an LLM call per query is wasteful when most
  queries are head traffic ("3 חדרים תל אביב", "טויוטה קורולה"). Serving those
  deterministically is effectively free. See the cost table in the README.
- **Latency.** The hot path is in-process (~0.4ms/parse warm). External model
  round-trips are provider-dependent and can move from hundreds of milliseconds
  to seconds, so they are gated, cached, and excluded from the common-path SLA.
- **Determinism & testability.** A deterministic core can be unit-tested and
  regression-tested; an LLM-only system is hard to pin down and to certify
  against the taxonomy contract.
- **Security.** The query is *untrusted input*. Not routing it to a model by
  default removes a whole class of prompt-injection risk. When the model is used,
  its output is still validated against the taxonomy allowlist.

**Trade-off.** Rules need maintenance and miss long-tail phrasing. I mitigate
this with (a) enrichment dictionaries, (b) hybrid lexical + vector retrieval,
(c) a learned-synonym mining pipeline from anonymized logs, and (d) the optional
LLM fallback for genuinely ambiguous, low-confidence cases.

**Fallback guardrail.** The fallback threshold is intentionally conservative
(`LLM_CONFIDENCE_THRESHOLD=0.60` by default), and clear category-only queries such
as `רכב אמריקאי גדול` remain on the deterministic vehicle path rather than
spending tokens to invent unsupported filters. When the model is used, output is capped
(`LLM_MAX_OUTPUT_TOKENS=700`), reasoning is minimal, and parser-side checks reject
responses that echo broad taxonomy defaults instead of user-constrained filters.

---

## 2. The pipeline shape

```
validate/sanitize -> normalize -> enrichment lookup -> lexical+fuzzy retrieval
  -> vector retrieval (ML) -> category scoring -> field extraction -> taxonomy
  validation -> confidence -> (optional) gated LLM fallback -> cache
```

**Why this order.** Cheap, high-precision signals run first; expensive/fuzzy
signals only contribute when needed. Validation is the last gate before output so
nothing leaves the service that is not in the taxonomy. This is "retrieval before
generation": we assemble candidate taxonomy values, then decide, rather than
letting a model free-form the answer.

---

## 3. Taxonomy as a hard output contract

**Decision.** `app/taxonomy.py` is the single source of truth. Every field and
value (and every LLM output) is filtered against the allowlist; unsupported
fields are dropped and noted.

**Why.** The downstream consumer is a search/filter backend, not a human. A
strict contract means a wrong/hallucinated field can never reach search. It also
makes the response safe to render and machine-consume.

**Trade-off.** The supplied taxonomy is compact, so some real marketplace filters
are not representable in this assignment. The design treats the taxonomy as data:
extend the metadata and the parser/validation contract follows. The gap vs. a full
marketplace filter surface is covered in the scope section below.

---

## 4. The ML engine: pluggable embeddings

**Problem.** Lexical/fuzzy matching misses semantic and transliterated long-tail
phrasing ("ג'יפ" vs "רכב שטח", "korola" vs "קורולה", slang).

**Decision.** A model-agnostic `Embedder` seam (`app/embeddings.py`) with two
backends behind one contract (`embed(text) -> vector`, `dim`):
- `HashedEmbedder` (default): deterministic hashed char n-grams. No model
  download, fully reproducible, good at spelling/transliteration overlap.
- `OpenAIEmbedder` (opt-in): real learned embeddings (`text-embedding-3-small`),
  L2-normalized and LRU-cached in-process.

Retrieval runs either in-process (`LocalVectorRetriever`) or against **Qdrant**
(`QdrantVectorRetriever`) via `docker-compose`. The vector size is taken from the
embedder, so swapping backends needs only a fresh collection.

**Why this design instead of shipping a heavy model by default.**
- Keeps the default image small, the build reproducible, and review frictionless.
- Embeddings are cheap and cacheable: taxonomy/alias vectors are embedded once at
  startup; popular queries hit the cache. So the "ML engine" adds recall without
  adding per-query cost in the common case.
- The same seam can host a self-hosted multilingual encoder (E5 / bge-m3 / a
  Hebrew-tuned model) served from an inference container — no parser changes.

**Trade-off.** Hashed embeddings are not true semantics; they are a baseline. The
honest production answer is a learned multilingual/Hebrew encoder behind this same
interface, which is exactly what the seam enables.

---

## 5. Category classification

**Decision.** Heuristic scoring over retrieval evidence (lexical + fuzzy + vector
signals, weighted) rather than a trained classifier.

**Why.** We were given a taxonomy and examples, not labeled traffic. A trained
classifier without representative labels would overfit and be hard to justify.
The scorer is transparent and debuggable (see `INCLUDE_DEBUG_NOTES`).

**Upgrade path.** Once anonymized labeled logs exist, the `_score_categories`
seam can be replaced by a lightweight learned classifier (fastText / logistic
regression over embeddings / a small transformer) with the existing eval harness
as the gate. This is the right time to add a model: when we can measure it.

---

## 6. Caching

**Decision.** TTL + LRU cache keyed on the *normalized* query (`app/cache.py`).

**Why.** Search traffic is heavy-tailed; the same head queries repeat constantly.
Normalizing before keying means "3 חדרים ת"א" and "3 חדרים תל אביב" share a hit.
Cache entries are deep-copied in/out so callers cannot mutate shared state.

**Trade-off / scale.** In-process LRU is per-instance. For a fleet, move to Redis/
Dragonfly (the interface is small) for a shared, higher hit-rate cache.

---

## 7. Security posture

**Decisions.** Input validation + control-char stripping, prompt-injection and
abuse-payload detection (flagged in `notes`), PII redaction in logs, optional API
key, per-client token-bucket rate limiting with stricter limits for suspicious
clients, request size caps, and strict security headers/CSP.

**Why.** The input is hostile by assumption. The model (when enabled) never gets
tools and never sees flagged inputs, and all output is allowlist-validated. We
limit the blast radius rather than pretending injection is "solved".

---

## 8. Observability

**Decision.** Prometheus `/metrics`: requests/errors, p50/p95 latency, cache-hit
ratio, category/source mix, model success/failure, token usage, and estimated
cost per request; structured per-request logs with a request id and redacted
query; a Grafana panel snapshot in `docs/dashboard.json`.

**Why.** To run this in production you must see quality and cost in real time —
especially cost-per-request once the LLM fallback is enabled, which is the metric
that protects the budget.

---

## 9. Response contract cleanliness

**Decision.** `/parse` returns `category`, `params`, `confidence`, optional
`notes`, and a compact `resolution` block. `resolution` is intentionally exposed
for reviewer and operator traceability: it shows whether the result came from
rules, cache, or LLM fallback, and includes token/cost data when a model was used.
Debug `evidence:` strings remain opt-in via `INCLUDE_DEBUG_NOTES`.

**Why.** The assignment asks for a structured parser, but a production AI system
also needs explainability and cost visibility. The response keeps business
filters separate from operational metadata, while logs and metrics still carry
the richer observability stream.

---

## Production Integration Path

This assignment is a local, Docker-runnable parser against the provided taxonomy.
In a real marketplace search stack, I would integrate it conservatively and prove
quality before it affects users. The credible path:

1. **Shadow mode.** Run the parser on real (anonymized) traffic without affecting
   results. Log parsed filters alongside what users actually clicked/edited.
2. **Offline eval set.** Build a versioned, sampled eval set from logs: head,
   tail, typo-heavy, transliterations, ambiguous verticals, zero-result, and
   injection attempts. Gate every change on `scripts/evaluate_examples.py`-style
   accuracy plus the red-team suite.
3. **Feedback loop.** Mine synonyms/aliases from query logs
   (`scripts/mine_synonyms.py`); promote them to enrichment only after eval +
   human review (prevents alias-driven false positives).
4. **A/B test.** Route a small percentage of eligible traffic to parser-assisted
   filtering; compare CTR, zero-result rate, filter-edit rate, and conversion
   against the existing search experience.
5. **Progressive rollout with guardrails.** Ramp by vertical and by confidence.
   Low-confidence or flagged queries fall back to the existing keyword search.
   Keep a kill switch and per-vertical quality alerts.
6. **Broaden only where measured quality wins.** Expand by vertical and query
   class after offline and online metrics prove that parser-assisted filtering is
   improving the existing experience.

**What full scale additionally needs (and where it plugs in):**
- Shared cache (Redis/Dragonfly) — swap `app/cache.py`.
- Hosted/learned multilingual-Hebrew embeddings — swap the `Embedder` backend.
- Managed Qdrant (or equivalent) cluster — already abstracted.
- A learned category classifier once labels exist — swap `_score_categories`.
- Autoscaled stateless API replicas behind the rate limiter and an LLM budget
  guard on the fallback.

The architecture is deliberately built as **swappable seams** (embedder, vector
backend, cache, classifier, fallback) so each component can graduate from a
review-friendly default to a production-grade implementation without rewriting
the pipeline.

---

## Scope vs. a Full Marketplace Filter Surface

I would not present the supplied taxonomy as a complete marketplace filter model.
It is the contract supplied for this assignment. A full production surface is wider:

- **Real estate:** transaction path (sale/rent/commercial), full location hierarchy
  (area/city/neighborhood/street), floor, sqm, and many amenity flags.
- **Vehicles:** vertical subtype (private/commercial/motorcycle/truck/boat), trim,
  engine volume/horsepower, and opportunity flags (0 km, price drop, financing).
- **Second-hand:** a much broader category tree (electronics, fashion, kids, home,
  tools, collectibles, ...) than the assignment's sample.

**How I'd close the gap in production (without rewriting the parser):**

1. **Filter metadata service.** Don't hardcode filters in parser code. Maintain a
   versioned metadata layer per vertical: `category → filter id → canonical value →
   label → aliases/typos → type (enum/bool/number/range/geo/date) → Yad2 URL/API
   parameter mapping`. The parser emits canonical internal filter ids; a thin
   downstream adapter translates them into real Yad2 search parameters.
2. **Clarification strategy.** Some queries should not be guessed (e.g. `אייפון פרו`,
   `דירה במרכז עד 4`). For low-confidence cases, return the partial parse plus
   `notes`, or — in a product UI — ask one short clarifying question instead of
   committing to a wrong filter.
3. **Online learning loop.** Use click / filter-edit / zero-result telemetry to
   train synonyms, calibrate confidence, and tune fallback routing over time, so
   model usage and error rate shrink as the system learns.

**Interview framing.** The repo is the runnable local service for the supplied
taxonomy. The production design does not stop at hand-written parsing; it adds a
filter metadata service, the candidate-retrieval/embedding seams already present
here, search-parameter mapping, and evals from real search logs. I deliberately
kept the submitted artifact self-contained and Docker-runnable, while leaving the
interfaces that would let production components replace the local defaults.
