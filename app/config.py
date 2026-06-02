from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    taxonomy_path: Path
    enrichment_path: Path | None = None
    learned_synonyms_path: Path | None = None
    enable_llm_fallback: bool = False
    openai_api_key: str | None = None
    openai_model: str = "gpt-5-nano"
    llm_confidence_threshold: float = 0.60
    llm_max_output_tokens: int = 700
    max_query_chars: int = 500
    cache_ttl_seconds: int = 3600
    cache_max_items: int = 10_000
    app_version: str = "1.0.0"
    service_api_key: str | None = None
    require_api_key: bool = False
    rate_limit_per_minute: int = 120
    rate_limit_burst: int = 40
    suspicious_rate_limit_per_minute: int = 12
    max_request_bytes: int = 4096
    enable_demo_ui: bool = True
    allowed_origins: tuple[str, ...] = ("*",)
    vector_backend: str = "local"
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "yad2_taxonomy_candidates"
    vector_top_k: int = 12
    embedding_backend: str = "hashed"
    openai_embedding_model: str = "text-embedding-3-small"
    include_debug_notes: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        repo_root = Path(__file__).resolve().parents[1]
        taxonomy_path = Path(
            os.getenv("YAD2_TAXONOMY_PATH", repo_root / "yad2_search_taxonomy.json")
        )
        enrichment_path = Path(
            os.getenv("YAD2_ENRICHMENT_PATH", repo_root / "data" / "enrichment_aliases.json")
        )
        learned_synonyms_path = Path(
            os.getenv(
                "YAD2_LEARNED_SYNONYMS_PATH",
                repo_root / "data" / "learned_synonyms.json",
            )
        )
        api_key = os.getenv("OPENAI_API_KEY")
        enable_llm = os.getenv("ENABLE_LLM_FALLBACK", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        return cls(
            taxonomy_path=taxonomy_path,
            enrichment_path=enrichment_path,
            learned_synonyms_path=learned_synonyms_path,
            enable_llm_fallback=enable_llm and bool(api_key),
            openai_api_key=api_key,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5-nano"),
            llm_confidence_threshold=float(
                os.getenv("LLM_CONFIDENCE_THRESHOLD", "0.60")
            ),
            llm_max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "700")),
            max_query_chars=int(os.getenv("MAX_QUERY_CHARS", "500")),
            cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600")),
            cache_max_items=int(os.getenv("CACHE_MAX_ITEMS", "10000")),
            app_version=os.getenv("APP_VERSION", "1.0.0"),
            service_api_key=os.getenv("SERVICE_API_KEY"),
            require_api_key=os.getenv("REQUIRE_API_KEY", "false").lower()
            in {"1", "true", "yes"},
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "120")),
            rate_limit_burst=int(os.getenv("RATE_LIMIT_BURST", "40")),
            suspicious_rate_limit_per_minute=int(
                os.getenv("SUSPICIOUS_RATE_LIMIT_PER_MINUTE", "12")
            ),
            max_request_bytes=int(os.getenv("MAX_REQUEST_BYTES", "4096")),
            enable_demo_ui=os.getenv("ENABLE_DEMO_UI", "true").lower()
            in {"1", "true", "yes"},
            allowed_origins=tuple(
                origin.strip()
                for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
                if origin.strip()
            ),
            vector_backend=os.getenv("VECTOR_BACKEND", "local").lower(),
            qdrant_url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
            qdrant_collection=os.getenv(
                "QDRANT_COLLECTION", "yad2_taxonomy_candidates"
            ),
            vector_top_k=int(os.getenv("VECTOR_TOP_K", "12")),
            embedding_backend=os.getenv("EMBEDDING_BACKEND", "hashed").lower(),
            openai_embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            include_debug_notes=os.getenv("INCLUDE_DEBUG_NOTES", "false").lower()
            in {"1", "true", "yes"},
        )
