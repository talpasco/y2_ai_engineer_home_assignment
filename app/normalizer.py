from __future__ import annotations

import re
import unicodedata


NIKUD_RE = re.compile(r"[\u0591-\u05C7]")


UNIT_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("שח", "ש״ח"),
    ("ש\"ח", "ש״ח"),
    ("ש''ח", "ש״ח"),
    ("₪", "ש״ח"),
    ("מליון", "מיליון"),
    ("מיליון", "מיליון"),
    ("מטרים רבועים", "מ״ר"),
    ("מטר רבוע", "מ״ר"),
    ("מטר", "מטר"),
    ("מ׳", "מטר"),
    ("מ\"ר", "מ״ר"),
    ("מר ", "מ״ר "),
    ("קמ", "ק״מ"),
    ("ק\"מ", "ק״מ"),
)


def strip_nikud(text: str) -> str:
    return NIKUD_RE.sub("", unicodedata.normalize("NFKC", text))


def normalize_spacing(text: str) -> str:
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text(text: str, typo_map: dict[str, str] | None = None) -> str:
    normalized = strip_nikud(text).lower()
    normalized = normalized.replace("׳", "'")
    normalized = normalized.replace("״", '"')
    normalized = normalized.replace("–", "-").replace("—", "-")

    typo_map = typo_map or {}
    for source, target in sorted(typo_map.items(), key=lambda item: len(item[0]), reverse=True):
        normalized = re.sub(re.escape(source.lower()), target.lower(), normalized)

    for source, target in UNIT_REPLACEMENTS:
        normalized = re.sub(re.escape(source.lower()), target.lower(), normalized)

    normalized = re.sub(r"(?<=\d),(?=\d)", ".", normalized)
    normalized = re.sub(r"\s*-\s*", "-", normalized)
    return normalize_spacing(normalized)


def canonical_phrase(text: str) -> str:
    return normalize_text(text, {})

