from __future__ import annotations

from pathlib import Path

from app.cache import TTLCache
from app.knowledge import KnowledgeBase
from app.models import ParseResult
from app.parser import SearchParser
from app.taxonomy import Taxonomy


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = Taxonomy.from_file(ROOT / "yad2_search_taxonomy.json")
KNOWLEDGE = KnowledgeBase.from_files(
    ROOT / "data" / "enrichment_aliases.json",
    ROOT / "data" / "learned_synonyms.json",
)


def make_parser() -> SearchParser:
    return SearchParser(
        TAXONOMY,
        knowledge=KNOWLEDGE,
        cache=TTLCache[ParseResult](max_items=100, ttl_seconds=60),
    )
