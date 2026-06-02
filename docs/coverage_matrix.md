# Assignment Coverage Matrix

| Requirement | Covered By |
| --- | --- |
| Python service | `app/` |
| Dockerized local run | `Dockerfile` |
| Production-like Docker stack | `docker-compose.yml` with API + Qdrant |
| `POST /parse` | `app/main.py` |
| `GET /health` | `app/main.py` |
| `GET /metrics` | `app/main.py`, `app/metrics.py` |
| OpenAPI/Swagger | FastAPI `/docs` |
| Hebrew free-text parsing | `app/parser.py`, `app/normalizer.py` |
| Typo/slang tolerance | taxonomy typo maps, fuzzy retrieval, enrichment aliases |
| Marketplace category detection | category scoring in `app/parser.py` |
| Structured filter extraction | vertical extractors in `app/parser.py` |
| Strict category/field allowlist | `app/taxonomy.py` |
| Numeric ranges | price/year/km/size parsing |
| Normalized strings | city/location/brand/model normalization |
| Caching | `app/cache.py` |
| Token/cost tracking | `app/llm.py`, `app/metrics.py` |
| Monthly cost estimate (10M queries) | `README.md` "Cost Model (10M queries / month)" |
| Latency targets (p95 ≤ 600ms model / ≤ 150ms cache+rules) | `README.md` "Design Quality" |
| p50/p95 latency metrics | `app/metrics.py` |
| Cache hit ratio | `app/metrics.py` |
| Model call success/failure | `app/metrics.py` |
| Structured logs | `app/main.py` |
| Security headers and API abuse controls | `app/security.py`, `app/main.py` |
| Prompt-injection handling | `app/security.py`, `examples/redteam_queries.json` |
| Red-team tests | `scripts/run_redteam.py` |
| Example I/O | `examples/sample_queries.json` |
| Evaluation harness | `scripts/evaluate_examples.py` |
| Design doc | `README.md`, `docs/design_decisions.md` |
| Dashboard snapshot | `docs/dashboard.json`, `docs/metrics_snapshot.txt` |
| Optional LLM fallback | `app/llm.py` |
| Pluggable ML embedding engine | `app/embeddings.py` (`HashedEmbedder` default, `OpenAIEmbedder` opt-in) |
| Optional vector DB | `docker-compose.yml`, `app/vector_retrieval.py` |
| Design rationale / production integration path | `docs/design_decisions.md` |
| Long-tail enrichment | `data/enrichment_aliases.json`, `app/knowledge.py` |
| Learned synonym pipeline | `examples/anonymized_query_logs.jsonl`, `scripts/mine_synonyms.py` |

## Notes

The supplied taxonomy is compact. The implementation honors it as the output contract while adding production-style enrichment and retrieval layers around it.

The default Dockerfile runs a single API container. `docker-compose.yml` runs the more production-like profile with Qdrant as a vector retrieval service.
