from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


STOPWORDS = {
    "עד",
    "עם",
    "יד",
    "של",
    "חדש",
    "חדשה",
    "כמו",
    "דירה",
    "חדרים",
    "שח",
    "אלף",
    "מליון",
    "מיליון",
}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    path = ROOT / "examples" / "anonymized_query_logs.jsonl"
    aliases = mine_aliases(path)
    report = {
        "source": str(path.relative_to(ROOT)),
        "aliases": aliases,
        "note": "Illustrative miner: production would use click/filter-edit statistics, thresholds, and human review.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def mine_aliases(path: Path) -> dict[str, list[str]]:
    aliases: dict[str, set[str]] = defaultdict(set)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tokens = tokenize(row["q"])
        for canonical in canonical_values(row.get("clicked_filters", {})):
            canonical_tokens = set(tokenize(canonical))
            for token in tokens:
                if token in STOPWORDS or token.isdigit() or token in canonical_tokens:
                    continue
                if looks_like_alias(token, canonical):
                    aliases[canonical].add(token)
    return {key: sorted(values) for key, values in sorted(aliases.items()) if values}


def canonical_values(filters: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key, value in filters.items():
        if key == "category":
            continue
        if isinstance(value, str):
            values.append(value)
    return values


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[a-zA-Zא-ת0-9׳״'\"]+", text)
        if len(token) >= 2
    ]


def looks_like_alias(token: str, canonical: str) -> bool:
    canonical_lower = canonical.lower()
    if token in canonical_lower or canonical_lower in token:
        return True
    if re.search(r"[a-zA-Z]", token) and re.search(r"[א-ת]", canonical):
        return True
    return False


if __name__ == "__main__":
    main()

