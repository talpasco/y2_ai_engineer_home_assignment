from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class LocationMatch:
    city: str | None = None
    neighborhood: str | None = None
    street: str | None = None
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SecondHandPhrase:
    phrase: str
    sector: str
    subcategory: str
    brand: str | None = None
    aliases: tuple[str, ...] = ()


class KnowledgeBase:
    def __init__(self, data: dict[str, Any], learned: dict[str, Any] | None = None) -> None:
        self.data = data
        self.learned = learned or {}

    @classmethod
    def from_files(
        cls, enrichment_path: Path | None, learned_path: Path | None = None
    ) -> "KnowledgeBase":
        data: dict[str, Any] = {}
        learned: dict[str, Any] = {}
        if enrichment_path and enrichment_path.exists():
            data = json.loads(enrichment_path.read_text(encoding="utf-8"))
        if learned_path and learned_path.exists():
            learned = json.loads(learned_path.read_text(encoding="utf-8"))
        return cls(data, learned)

    @classmethod
    def empty(cls) -> "KnowledgeBase":
        return cls({})

    def city_aliases(self) -> dict[str, list[str]]:
        cities = self.data.get("locations", {}).get("cities", {})
        return {city: [city, *aliases] for city, aliases in cities.items()}

    def neighborhood_aliases(self) -> dict[str, LocationMatch]:
        neighborhoods = self.data.get("locations", {}).get("neighborhoods", {})
        return {
            name: LocationMatch(
                city=spec.get("city"),
                neighborhood=name,
                aliases=tuple([name, *spec.get("aliases", [])]),
            )
            for name, spec in neighborhoods.items()
        }

    def street_aliases(self) -> dict[str, LocationMatch]:
        streets = self.data.get("locations", {}).get("streets", {})
        return {
            name: LocationMatch(
                city=spec.get("city"),
                street=name,
                aliases=tuple([name, *spec.get("aliases", [])]),
            )
            for name, spec in streets.items()
        }

    def vehicle_trims(self) -> dict[str, Any]:
        return self.data.get("vehicle_trims", {})

    def second_hand_phrases(self) -> list[SecondHandPhrase]:
        phrases = self.data.get("second_hand_phrases", {})
        return [
            SecondHandPhrase(
                phrase=phrase,
                sector=spec["sector"],
                subcategory=spec["subcategory"],
                brand=spec.get("brand"),
                aliases=tuple([phrase, *spec.get("aliases", [])]),
            )
            for phrase, spec in phrases.items()
        ]

    def multilingual_aliases(self) -> dict[str, list[str]]:
        aliases = self.data.get("multilingual_aliases", {})
        learned_aliases = self.learned.get("aliases", {})
        merged: dict[str, list[str]] = {}
        for canonical, values in {**aliases, **learned_aliases}.items():
            merged.setdefault(canonical, [])
            for value in values:
                if value not in merged[canonical]:
                    merged[canonical].append(value)
        return merged

