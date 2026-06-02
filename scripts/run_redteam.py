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
    cases = json.loads((ROOT / "examples" / "redteam_queries.json").read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for case in cases:
        result = parser.parse(case["q"]).public_dict()
        passed, reasons = evaluate_case(case, result)
        if not passed:
            failures.append(case["id"])
        rows.append(
            {
                "id": case["id"],
                "passed": passed,
                "reasons": reasons,
                "category": result["category"],
                "confidence": result["confidence"],
            }
        )

    report = {
        "cases": len(cases),
        "passed": len(cases) - len(failures),
        "failed": len(failures),
        "failure_ids": failures,
        "rows": rows,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def evaluate_case(case: dict[str, Any], result: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    params = result.get("params", {})
    notes = " ".join(result.get("notes", []))

    if result.get("category") != case.get("expected_category"):
        reasons.append(
            f"category expected {case.get('expected_category')}, got {result.get('category')}"
        )

    for field in case.get("must_not_include_fields", []):
        if field in params:
            reasons.append(f"forbidden field present: {field}")

    for key, value in case.get("must_include_params", {}).items():
        if params.get(key) != value:
            reasons.append(f"param {key!r} expected {value!r}, got {params.get(key)!r}")

    for key, value in case.get("must_not_include_params", {}).items():
        if params.get(key) == value:
            reasons.append(f"forbidden param value present: {key}={value!r}")

    fragment = case.get("must_include_note_fragment")
    if fragment and fragment not in notes:
        reasons.append(f"missing note fragment: {fragment}")

    return not reasons, reasons


if __name__ == "__main__":
    main()
