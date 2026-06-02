from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REAL_ESTATE = "נדל״ן"
VEHICLES = "רכב"
SECOND_HAND = "יד_שנייה"


@dataclass(frozen=True, slots=True)
class SecondHandSubcategory:
    sector: str
    subcategory: str
    spec: dict[str, Any]


class Taxonomy:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.categories: dict[str, Any] = data["קטגוריות"]

    @classmethod
    def from_file(cls, path: Path) -> "Taxonomy":
        with path.open("r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def category_names(self) -> list[str]:
        return list(self.categories.keys())

    def typo_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for category in self.categories.values():
            mapping.update(category.get("מיפוי_מילות_שגיאה", {}))
        return mapping

    def real_estate_property_types(self) -> list[str]:
        return list(self.categories[REAL_ESTATE]["סוגי_נכס"])

    def real_estate_transaction_types(self) -> list[str]:
        return list(self.categories[REAL_ESTATE]["מצבי_עסקה"])

    def real_estate_city_examples(self) -> list[str]:
        examples = self.categories[REAL_ESTATE]["מאפיינים_כלליים"]["עיר"].get(
            "דוגמאות", []
        )
        return list(examples)

    def vehicle_manufacturers(self) -> dict[str, Any]:
        return dict(self.categories[VEHICLES]["יצרנים"])

    def vehicle_models(self) -> dict[str, str]:
        models: dict[str, str] = {}
        for manufacturer, spec in self.vehicle_manufacturers().items():
            for model in spec.get("דגמים", []):
                models[str(model)] = manufacturer
        return models

    def vehicle_colors(self) -> list[str]:
        return list(self.categories[VEHICLES]["מאפיינים_כלליים"]["צבע"])

    def second_hand_subcategories(self) -> list[SecondHandSubcategory]:
        output: list[SecondHandSubcategory] = []
        sectors = self.categories[SECOND_HAND]["סקטורים"]
        for sector, sector_spec in sectors.items():
            for subcategory, spec in sector_spec["תתי_קטגוריות"].items():
                output.append(SecondHandSubcategory(sector, subcategory, dict(spec)))
        return output

    def second_hand_brands(self) -> dict[str, tuple[str, str]]:
        brands: dict[str, tuple[str, str]] = {}
        for item in self.second_hand_subcategories():
            for brand in item.spec.get("מותגים", []):
                brands[str(brand)] = (item.sector, item.subcategory)
        return brands

    def city_aliases(self) -> dict[str, list[str]]:
        return {
            "תל אביב-יפו": ["תל אביב-יפו", "תל אביב", "תל-אביב", "תל אביב יפו"],
            "ירושלים": ["ירושלים", "ירושליים"],
            "חיפה": ["חיפה"],
            "רמת גן": ["רמת גן", "רמתגן"],
            "באר שבע": ["באר שבע", "בארשבע"],
        }

    def allowed_fields(self, category: str, params: dict[str, Any] | None = None) -> set[str]:
        params = params or {}
        if category == REAL_ESTATE:
            fields = set(self.categories[REAL_ESTATE]["מאפיינים_כלליים"].keys())
            fields.update({"סוגי_נכס", "מצבי_עסקה"})
            return fields
        if category == VEHICLES:
            fields = set(self.categories[VEHICLES]["מאפיינים_כלליים"].keys())
            fields.update({"סוגי_רכב", "יצרן", "דגם", "תת_דגם"})
            return fields
        if category == SECOND_HAND:
            fields = set(self.categories[SECOND_HAND]["מאפיינים_כלליים"].keys())
            fields.update({"סקטור", "תת_קטגוריה", "דגם"})
            for item in self.second_hand_subcategories():
                fields.update(item.spec.keys())
                if "מותגים" in fields:
                    fields.discard("מותגים")
                    fields.add("מותג")
            return fields
        return set()

    def validate_params(
        self, category: str, params: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        if category not in self.categories:
            return {}, [f"invalid category: {category}"]
        allowed = self.allowed_fields(category, params)
        notes: list[str] = []
        filtered: dict[str, Any] = {}
        for key, value in params.items():
            if key not in allowed:
                notes.append(f"removed unsupported field: {key}")
                continue
            if value is None or value == [] or value == {}:
                continue
            spec = self._spec_for_param(category, key, params)
            cleaned = self._validate_value(key, value, spec)
            if cleaned is None:
                notes.append(f"removed invalid value for field: {key}")
                continue
            filtered[key] = cleaned
        return filtered, notes

    def compact_slice_for_llm(self, category: str | None = None) -> dict[str, Any]:
        if category:
            return {"קטגוריות": {category: self.categories[category]}}
        return {
            "קטגוריות": self.categories,
            "כללי": self.data.get("כללי", {}),
        }

    def _spec_for_param(
        self, category: str, key: str, params: dict[str, Any]
    ) -> Any:
        if category == REAL_ESTATE:
            if key == "סוגי_נכס":
                return self.categories[REAL_ESTATE]["סוגי_נכס"]
            if key == "מצבי_עסקה":
                return self.categories[REAL_ESTATE]["מצבי_עסקה"]
            return self.categories[REAL_ESTATE]["מאפיינים_כלליים"].get(key)

        if category == VEHICLES:
            if key == "סוגי_רכב":
                return self.categories[VEHICLES]["סוגי_רכב"]
            if key == "יצרן":
                return list(self.vehicle_manufacturers().keys())
            if key == "דגם":
                manufacturer = params.get("יצרן")
                if manufacturer in self.vehicle_manufacturers():
                    return self.vehicle_manufacturers()[manufacturer].get("דגמים", [])
                return list(self.vehicle_models().keys())
            if key == "תת_דגם":
                manufacturer = params.get("יצרן")
                model = params.get("דגם")
                if manufacturer in self.vehicle_manufacturers():
                    values = (
                        self.vehicle_manufacturers()[manufacturer]
                        .get("תתי_דגמים", {})
                        .get(model, [])
                    )
                    return values or None
                return None
            return self.categories[VEHICLES]["מאפיינים_כלליים"].get(key)

        if category == SECOND_HAND:
            sectors = self.categories[SECOND_HAND]["סקטורים"]
            if key == "סקטור":
                return list(sectors.keys())
            if key == "תת_קטגוריה":
                sector = params.get("סקטור")
                if sector in sectors:
                    return list(sectors[sector]["תתי_קטגוריות"].keys())
                return [item.subcategory for item in self.second_hand_subcategories()]
            if key == "מותג":
                brands: list[str] = []
                for item in self.second_hand_subcategories():
                    if params.get("תת_קטגוריה") and item.subcategory != params["תת_קטגוריה"]:
                        continue
                    brands.extend(str(value) for value in item.spec.get("מותגים", []))
                return brands or None
            if key == "דגם":
                return {"טיפוס": "מחרוזת"}
            general = self.categories[SECOND_HAND]["מאפיינים_כלליים"].get(key)
            if general is not None:
                return general
            for item in self.second_hand_subcategories():
                if params.get("תת_קטגוריה") and item.subcategory != params["תת_קטגוריה"]:
                    continue
                if key in item.spec:
                    return item.spec[key]
        return None

    @staticmethod
    def _validate_value(key: str, value: Any, spec: Any) -> Any:
        if spec is None:
            return value
        if isinstance(spec, list):
            if isinstance(value, list):
                valid = [item for item in value if item in spec]
                return valid or None
            return value if value in spec else None
        if not isinstance(spec, dict):
            return value

        value_type = spec.get("טיפוס")
        if value_type == "מספר":
            return _validate_numeric(value)
        if value_type == "בוליאני":
            return value if isinstance(value, bool) else None
        if value_type in {"מחרוזת", "תאריך", "תאריך_יחסי_או_מדויק"}:
            return value if isinstance(value, str) and value.strip() else None
        return value


def _validate_numeric(value: Any) -> Any:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, dict):
        cleaned: dict[str, int | float] = {}
        for key in ("min", "max"):
            if key in value:
                number = value[key]
                if isinstance(number, bool) or not isinstance(number, (int, float)):
                    return None
                cleaned[key] = number
        return cleaned or None
    return None
