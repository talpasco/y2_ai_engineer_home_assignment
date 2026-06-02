from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.cache import TTLCache
from app.knowledge import KnowledgeBase
from app.models import ParseResult
from app.parser import SearchParser
from app.taxonomy import Taxonomy


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    taxonomy = Taxonomy.from_file(ROOT / "yad2_search_taxonomy.json")
    knowledge = KnowledgeBase.from_files(
        ROOT / "data" / "enrichment_aliases.json",
        ROOT / "data" / "learned_synonyms.json",
    )
    parser = SearchParser(
        taxonomy,
        knowledge=knowledge,
        cache=TTLCache[ParseResult](max_items=100, ttl_seconds=60),
    )
    examples = json.loads((ROOT / "examples" / "sample_queries.json").read_text(encoding="utf-8"))

    category_correct = 0
    expected_fields = 0
    matched_fields = 0
    rows: list[dict[str, Any]] = []

    for item in examples:
        result = parser.parse(item["q"]).public_dict()
        expected = item["expected"]
        category_match = result["category"] == expected["category"]
        category_correct += int(category_match)

        field_matches = 0
        field_total = len(expected["params"])
        for key, expected_value in expected["params"].items():
            expected_fields += 1
            if result["params"].get(key) == expected_value:
                matched_fields += 1
                field_matches += 1

        rows.append(
            {
                "q": item["q"],
                "category_match": category_match,
                "field_matches": f"{field_matches}/{field_total}",
                "confidence": result["confidence"],
            }
        )

    report = {
        "examples": len(examples),
        "category_accuracy": round(category_correct / len(examples), 3),
        "expected_field_accuracy": round(matched_fields / expected_fields, 3),
        "rows": rows,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
