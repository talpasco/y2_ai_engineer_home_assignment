from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from app.cache import TTLCache, cache_hit_result
from app.knowledge import KnowledgeBase
from app.models import ParseResult, QueryValidationError
from app.normalizer import normalize_text
from app.retrieval import (
    LocalCandidateRetriever,
    RetrievalEntry,
    any_phrase,
    any_phrase_fuzzy,
    fuzzy_phrase_in_text,
    phrase_in_text,
)
from app.security import detect_security_flags, validate_query_shape
from app.taxonomy import REAL_ESTATE, SECOND_HAND, VEHICLES, Taxonomy
from app.vector_retrieval import LocalVectorRetriever, QdrantVectorRetriever, VectorRetriever


NUMBER = r"(?P<num>\d+(?:[.,]\d+)?|מיליון)"
AMOUNT = r"(?P<num>\d+(?:[.,]\d+)?|מיליון)(?:\s*(?P<unit>אלף|אלפים|מיליון|k))?"
MAX_WORDS = {"עד", "מקסימום", "מקס", "מתחת", "פחות"}
MIN_WORDS = {"מעל", "לפחות", "מינימום", "יותר"}


REAL_ESTATE_ALIASES: dict[str, list[str]] = {
    "דירה": ["דירה", "דירת"],
    "דירת גן": ["דירת גן"],
    "דופלקס": ["דופלקס"],
    "פנטהאוז": ["פנטהאוז"],
    "מיני פנטהאוז": ["מיני פנטהאוז"],
    "דירת סטודיו": ["סטודיו", "דירת סטודיו"],
    "קוטג׳": ["קוטג", "קוטג'"],
    "בית פרטי/וילה": ["בית פרטי", "וילה", "בית"],
    "יחידת דיור": ["יחידת דיור", "יחידה"],
}

TRANSACTION_ALIASES: dict[str, list[str]] = {
    "מכירה": ["מכירה", "למכירה", "קניה", "קנייה"],
    "השכרה": ["השכרה", "להשכרה", "שכירות", "שכר דירה"],
    "שותפים": ["שותפים", "שותף", "שותפה"],
    "מסחרי": ["מסחרי", "משרד", "חנות"],
    "מגרשים": ["מגרש", "מגרשים"],
}

VEHICLE_MANUFACTURER_ALIASES: dict[str, list[str]] = {
    "טויוטה": ["טויוטה", "toyota"],
    "יונדאי": ["יונדאי", "hyundai"],
    "קיה": ["קיה", "kia"],
    "מאזדה": ["מאזדה", "mazda"],
    "מרצדס": ["מרצדס", "mercedes"],
    "ב.מ.וו": ["ב.מ.וו", "במוו", "bmw"],
    "אאודי": ["אאודי", "audi"],
    "שברולט": ["שברולט", "chevrolet"],
    "סוזוקי": ["סוזוקי", "suzuki"],
    "MG": ["mg"],
    "BYD": ["byd"],
    "טסלה": ["טסלה", "tesla"],
}

# Generic vehicle nouns that signal the Vehicle category even without a specific
# manufacturer/model (e.g. "רכב אמריקאי יחסית חדש"). Hebrew word-boundary matching
# means only the standalone word matches, so prefixed forms like "לרכב" ("for the
# car", i.e. an accessory) do not trigger a vehicle classification.
VEHICLE_GENERIC_ALIASES: list[str] = [
    "רכב",
    "רכבים",
    "אוטו",
    "מכונית",
    "מכוניות",
    "ג'יפ",
    "טנדר",
]

SECOND_HAND_ALIASES: dict[tuple[str, str], list[str]] = {
    ("אלקטרוניקה", "טלפונים_סלולריים"): [
        "אייפון",
        "iphone",
        "גלקסי",
        "galaxy",
        "טלפון",
        "סלולרי",
        "סמארטפון",
    ],
    ("אלקטרוניקה", "מחשבים_ניידים"): [
        "מחשב נייד",
        "לפטופ",
        "מקבוק",
        "macbook",
        "notebook",
    ],
    ("אלקטרוניקה", "טלוויזיות"): ["טלוויזיה", "טלויזיה", "מסך", "oled", "qled"],
    ("ריהוט", "סלון"): ["ספה", "כורסה", "שולחן קפה", "מזנון", "סלון"],
    ("ריהוט", "חדר_שינה"): ["מיטה", "ארון", "שידות", "מזרן", "חדר שינה"],
    ("ריהוט", "פינת_אוכל"): ["פינת אוכל", "שולחן אוכל", "כיסאות", "ויטרינה"],
    ("ספורט_וקמפינג", "אופניים"): ["אופניים", "אופני", "אופני כביש", "אופני שטח"],
    ("ספורט_וקמפינג", "ציוד_כושר"): [
        "הליכון",
        "אליפטיקל",
        "משקולות",
        "ציוד כושר",
    ],
    ("לתינוקות_וסופגנים", "עגלות"): ["עגלה", "עגלת", "עגלות", "יויו"],
    ("לתינוקות_וסופגנים", "מוצצים_ובקבוקים"): ["מוצץ", "מוצצים", "בקבוק"],
    ("מוסיקה_וכלים", "גיטרות"): ["גיטרה", "גיטרות", "בס", "אקוסטית"],
    ("מוסיקה_וכלים", "קלידים"): ["קלידים", "סינתיסייזר", "פסנתר דיגיטלי"],
}

BRAND_ALIASES: dict[str, str] = {
    "אייפון": "אפל",
    "iphone": "אפל",
    "מקבוק": "אפל",
    "macbook": "אפל",
    "גלקסי": "סמסונג",
    "galaxy": "סמסונג",
    "אל ג'י": "LG",
    "אלגי": "LG",
    "יויו": "YOYO",
}

COLOR_ALIASES: dict[str, str] = {
    "לבן": "לבן",
    "לבנה": "לבן",
    "שחור": "שחור",
    "שחורה": "שחור",
    "אפור": "אפור",
    "אפורה": "אפור",
    "כסוף": "כסוף",
    "כסופה": "כסוף",
    "כחול": "כחול",
    "כחולה": "כחול",
    "אדום": "אדום",
    "אדומה": "אדום",
    "ירוק": "ירוק",
    "ירוקה": "ירוק",
    "צהוב": "צהוב",
    "צהובה": "צהוב",
    "זהב": "זהב",
    "סגול": "סגול",
    "סגולה": "סגול",
}


def _retrieval_entries(taxonomy: Taxonomy, knowledge: KnowledgeBase) -> list[RetrievalEntry]:
    entries: list[RetrievalEntry] = []
    for value, aliases in REAL_ESTATE_ALIASES.items():
        entries.append(
            RetrievalEntry(
                REAL_ESTATE,
                "סוגי_נכס",
                value,
                tuple([*aliases, *knowledge.multilingual_aliases().get(value, [])]),
                4.0,
            )
        )
    for value, aliases in TRANSACTION_ALIASES.items():
        entries.append(RetrievalEntry(REAL_ESTATE, "מצבי_עסקה", value, tuple(aliases), 2.0))
    for city, aliases in taxonomy.city_aliases().items():
        entries.append(RetrievalEntry(REAL_ESTATE, "עיר", city, tuple(aliases), 1.4))
    for city, aliases in knowledge.city_aliases().items():
        entries.append(RetrievalEntry(REAL_ESTATE, "עיר", city, tuple(aliases), 1.6))
    for neighborhood, match in knowledge.neighborhood_aliases().items():
        entries.append(
            RetrievalEntry(
                REAL_ESTATE, "שכונה", neighborhood, match.aliases, 2.2
            )
        )
    for street, match in knowledge.street_aliases().items():
        entries.append(RetrievalEntry(REAL_ESTATE, "רחוב", street, match.aliases, 2.0))

    for value, aliases in VEHICLE_MANUFACTURER_ALIASES.items():
        entries.append(
            RetrievalEntry(
                VEHICLES,
                "יצרן",
                value,
                tuple([*aliases, *knowledge.multilingual_aliases().get(value, [])]),
                5.0,
            )
        )
    for model, manufacturer in taxonomy.vehicle_models().items():
        entries.append(
            RetrievalEntry(
                VEHICLES,
                "דגם",
                model,
                tuple([f"{manufacturer} {model}", *knowledge.multilingual_aliases().get(model, [])]),
                4.0,
            )
        )
    for manufacturer, models in knowledge.vehicle_trims().items():
        for model, trims in models.items():
            for trim, aliases in trims.items():
                entries.append(
                    RetrievalEntry(
                        VEHICLES,
                        "תת_דגם",
                        trim,
                        tuple([trim, *aliases, f"{manufacturer} {model} {trim}"]),
                        2.8,
                    )
                )

    for (sector, subcategory), aliases in SECOND_HAND_ALIASES.items():
        entries.append(
            RetrievalEntry(SECOND_HAND, "תת_קטגוריה", subcategory, tuple(aliases), 5.0)
        )
        entries.append(RetrievalEntry(SECOND_HAND, "סקטור", sector, (sector,), 1.0))
    for brand, (sector, subcategory) in taxonomy.second_hand_brands().items():
        entries.append(
            RetrievalEntry(
                SECOND_HAND,
                "מותג",
                brand,
                tuple([brand, *knowledge.multilingual_aliases().get(brand, [])]),
                3.0,
            )
        )
        entries.append(RetrievalEntry(SECOND_HAND, "תת_קטגוריה", subcategory, (brand,), 2.0))
    for phrase in knowledge.second_hand_phrases():
        entries.append(
            RetrievalEntry(
                SECOND_HAND,
                "תת_קטגוריה",
                phrase.subcategory,
                phrase.aliases,
                4.5,
            )
        )
        if phrase.brand:
            entries.append(
                RetrievalEntry(
                    SECOND_HAND,
                    "מותג",
                    phrase.brand,
                    phrase.aliases,
                    2.5,
                )
            )

    return entries


class SearchParser:
    def __init__(
        self,
        taxonomy: Taxonomy,
        *,
        knowledge: KnowledgeBase | None = None,
        max_query_chars: int = 500,
        cache: TTLCache[ParseResult] | None = None,
        llm_fallback: Any | None = None,
        llm_confidence_threshold: float = 0.60,
        vector_backend: str = "local",
        qdrant_url: str = "http://qdrant:6333",
        qdrant_collection: str = "yad2_taxonomy_candidates",
        vector_top_k: int = 12,
        embedder: Any | None = None,
        include_debug_notes: bool = False,
    ) -> None:
        self.taxonomy = taxonomy
        self.knowledge = knowledge or KnowledgeBase.empty()
        self.max_query_chars = max_query_chars
        self.cache = cache or TTLCache[ParseResult]()
        self.llm_fallback = llm_fallback
        self.llm_confidence_threshold = llm_confidence_threshold
        self.vector_top_k = vector_top_k
        self.include_debug_notes = include_debug_notes
        entries = _retrieval_entries(taxonomy, self.knowledge)
        self.retriever = LocalCandidateRetriever(entries)
        self.vector_retriever = self._build_vector_retriever(
            entries,
            vector_backend=vector_backend,
            qdrant_url=qdrant_url,
            qdrant_collection=qdrant_collection,
            embedder=embedder,
        )

    def parse(self, query: str) -> ParseResult:
        raw = validate_query_shape(query, self.max_query_chars)
        security_flags = detect_security_flags(raw)
        normalized = normalize_text(raw, self.taxonomy.typo_map())
        cached = self.cache.get(normalized)
        if cached:
            return cache_hit_result(cached)

        result = self._parse_rules(normalized, security_flags)
        if self._should_try_llm(result, security_flags):
            result = self._try_llm(normalized, result)

        params, validation_notes = self.taxonomy.validate_params(result.category, result.params)
        result.params = params
        result.notes.extend(validation_notes)
        result.notes = _dedupe(result.notes)
        self.cache.set(normalized, result)
        return result

    def _should_try_llm(self, result: ParseResult, security_flags: list[str]) -> bool:
        if self.llm_fallback is None or security_flags:
            return False
        if result.confidence >= self.llm_confidence_threshold:
            return False
        if result.category in self.taxonomy.category_names() and not result.params:
            return result.confidence < 0.50
        return True

    @staticmethod
    def _build_vector_retriever(
        entries: list[RetrievalEntry],
        *,
        vector_backend: str,
        qdrant_url: str,
        qdrant_collection: str,
        embedder: Any | None = None,
    ) -> VectorRetriever:
        if vector_backend == "none":
            return LocalVectorRetriever([], embedder=embedder)
        if vector_backend == "qdrant":
            qdrant = QdrantVectorRetriever(
                entries,
                url=qdrant_url,
                collection=qdrant_collection,
                embedder=embedder,
            )
            if qdrant.ready:
                return qdrant
        return LocalVectorRetriever(entries, embedder=embedder)

    def _try_llm(self, normalized: str, rule_result: ParseResult) -> ParseResult:
        try:
            llm_result = self.llm_fallback.parse(
                normalized,
                taxonomy=self.taxonomy,
                preferred_category=rule_result.category,
            )
        except Exception as exc:  # pragma: no cover - defensive integration path
            rule_result.notes.append(f"llm fallback failed: {exc.__class__.__name__}")
            rule_result.model_call_failure = True
            return rule_result

        if llm_result.category not in self.taxonomy.category_names():
            rule_result.notes.append("llm fallback returned invalid category")
            rule_result.model_call_failure = True
            return rule_result

        if _looks_like_taxonomy_echo(llm_result.params):
            rule_result.notes.append("kept rules result over llm fallback: broad taxonomy echo")
            rule_result.model_name = llm_result.model_name
            rule_result.model_call_failure = True
            rule_result.usage = llm_result.usage
            return rule_result

        params, validation_notes = self.taxonomy.validate_params(
            llm_result.category, llm_result.params
        )
        llm_result.params = params
        llm_result.notes.extend(validation_notes)
        required_gain = 0.05 if llm_result.category == rule_result.category else 0.0
        if (
            llm_result.confidence >= rule_result.confidence + required_gain
            and llm_result.params
        ):
            return llm_result
        rule_result.notes.append("kept rules result over llm fallback")
        rule_result.model_name = llm_result.model_name
        rule_result.model_call_success = True
        rule_result.usage = llm_result.usage
        return rule_result

    def _parse_rules(self, text: str, security_flags: list[str]) -> ParseResult:
        scores, evidence = self._score_categories(text)
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        category, score = ordered[0]
        margin = score - ordered[1][1] if len(ordered) > 1 else score

        if score <= 0:
            category = SECOND_HAND
            evidence[category].append("default_low_information")

        if category == REAL_ESTATE:
            params = self._extract_real_estate(text)
        elif category == VEHICLES:
            params = self._extract_vehicle(text)
        else:
            params = self._extract_second_hand(text)

        notes = [f"security: {flag}" for flag in security_flags]
        if score <= 0:
            notes.append("low confidence: no strong taxonomy signal")
        if self.include_debug_notes and evidence.get(category):
            notes.append("evidence: " + ", ".join(evidence[category][:5]))

        confidence = self._confidence(score, margin, len(params), bool(security_flags))
        return ParseResult(
            category=category,
            params=params,
            confidence=confidence,
            notes=notes,
            source="rules",
        )

    def _score_categories(self, text: str) -> tuple[dict[str, float], dict[str, list[str]]]:
        scores = {REAL_ESTATE: 0.0, VEHICLES: 0.0, SECOND_HAND: 0.0}
        evidence = {REAL_ESTATE: [], VEHICLES: [], SECOND_HAND: []}

        for candidate in self.retriever.retrieve(text, limit=24):
            scores[candidate.category] += candidate.score * 0.25
            evidence[candidate.category].append(
                f"{candidate.field}:{candidate.value}:{candidate.match_type}"
            )
        for candidate in self.vector_retriever.retrieve(text, limit=self.vector_top_k):
            scores[candidate.category] += candidate.score * 0.18
            evidence[candidate.category].append(
                f"{candidate.field}:{candidate.value}:{candidate.match_type}"
            )

        for value, aliases in REAL_ESTATE_ALIASES.items():
            if _any_phrase(text, aliases) or _any_phrase_fuzzy(text, aliases):
                scores[REAL_ESTATE] += 4
                evidence[REAL_ESTATE].append(value)
        if re.search(r"\d+(?:[.,]\d+)?\s*(?:חדר|חדרים|חד)", text):
            scores[REAL_ESTATE] += 3
            evidence[REAL_ESTATE].append("rooms")
        if re.search(r"\bקומה\b|מ״ר|ממד|ממ\"ד|מרפסת|מעלית", text):
            scores[REAL_ESTATE] += 2
            evidence[REAL_ESTATE].append("real_estate_features")
        for city, aliases in self.taxonomy.city_aliases().items():
            if _any_phrase(text, aliases) or _any_phrase_fuzzy(text, aliases):
                scores[REAL_ESTATE] += 1
                scores[SECOND_HAND] += 0.3
                evidence[REAL_ESTATE].append(city)

        for manufacturer, aliases in VEHICLE_MANUFACTURER_ALIASES.items():
            if _any_phrase(text, aliases) or _any_phrase_fuzzy(text, aliases):
                scores[VEHICLES] += 5
                evidence[VEHICLES].append(manufacturer)
        for model, manufacturer in self.taxonomy.vehicle_models().items():
            if str(model).isdigit() and not _phrase_in_text(text, f"{manufacturer} {model}"):
                continue
            model_aliases = self.knowledge.multilingual_aliases().get(str(model), [])
            if (
                _phrase_in_text(text, model)
                or _fuzzy_phrase_in_text(text, str(model))
                or _any_phrase(text, model_aliases)
                or _any_phrase_fuzzy(text, model_aliases)
            ):
                scores[VEHICLES] += 4
                evidence[VEHICLES].append(f"{manufacturer} {model}")
        if re.search(r"\bק״מ\b|קילומטר|יד\s*\d+|אוטומט|ידנית|בנזין|דיזל|היברידי|חשמלי", text):
            scores[VEHICLES] += 2
            evidence[VEHICLES].append("vehicle_features")
        if _any_phrase(text, VEHICLE_GENERIC_ALIASES):
            scores[VEHICLES] += 4
            evidence[VEHICLES].append("vehicle_category_term")

        subcategory_match = self._best_second_hand_candidate(text)
        if subcategory_match:
            scores[SECOND_HAND] += subcategory_match[2]
            evidence[SECOND_HAND].append(subcategory_match[1])
        for brand, (sector, subcategory) in self.taxonomy.second_hand_brands().items():
            if _phrase_in_text(text, brand):
                scores[SECOND_HAND] += 3
                evidence[SECOND_HAND].append(f"{sector}/{subcategory}/{brand}")
        if re.search(r"כמו חדש|לחלפים|משומש|חדש|ג׳יגה|גיגה|gb|אינץ|ram|ראם", text):
            scores[SECOND_HAND] += 1.5
            evidence[SECOND_HAND].append("second_hand_specs")

        return scores, evidence

    def _extract_real_estate(self, text: str) -> dict[str, Any]:
        params: dict[str, Any] = {}

        property_types: list[str] = []
        for property_type, aliases in sorted(
            REAL_ESTATE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if property_type in self.taxonomy.real_estate_property_types() and (
                _any_phrase(text, aliases) or _any_phrase_fuzzy(text, aliases)
            ):
                property_types.append(property_type)
        if property_types:
            params["סוגי_נכס"] = _dedupe(property_types)

        transactions = [
            transaction
            for transaction, aliases in TRANSACTION_ALIASES.items()
            if _any_phrase(text, aliases)
        ]
        if transactions:
            params["מצבי_עסקה"] = _dedupe(transactions)

        city = self._extract_city(text)
        if city:
            params["עיר"] = city

        neighborhood = self._extract_neighborhood(text)
        if neighborhood:
            params["שכונה"] = neighborhood
            if "עיר" not in params:
                city_from_neighborhood = self._city_for_neighborhood(neighborhood)
                if city_from_neighborhood:
                    params["עיר"] = city_from_neighborhood

        street = self._extract_street(text)
        if street:
            params["רחוב"] = street
            if "עיר" not in params:
                city_from_street = self._city_for_street(street)
                if city_from_street:
                    params["עיר"] = city_from_street

        rooms = _number_after_or_before(text, r"חדרים?|חד")
        if rooms is not None and 1 <= rooms <= 12:
            params["מס׳_חדרים"] = _int_if_whole(rooms)

        floor_match = re.search(r"(?:קומה|בקומה)\s*(\d{1,2})", text)
        if floor_match:
            params["קומה"] = int(floor_match.group(1))
        elif "קומת קרקע" in text or "קרקע" in text:
            params["קומה"] = 0

        sqm_match = re.search(r"(\d{2,4})\s*(?:מ״ר|מטר(?:ים)?)", text)
        if sqm_match:
            params["מ״ר_בנוי"] = int(sqm_match.group(1))

        balcony_match = re.search(r"(\d+)\s*מרפס", text)
        if balcony_match:
            params["מרפסות"] = int(balcony_match.group(1))
        elif "מרפסת" in text:
            params["מרפסות"] = 1
        if "מרפסת שמש" in text:
            params["מרפסת_שמש"] = True

        parking_match = re.search(r"(\d+)\s*חני", text)
        if parking_match:
            params["חניה"] = int(parking_match.group(1))
        elif "חניה" in text or "חנייה" in text:
            params["חניה"] = 1

        for phrase, field in (
            ("מעלית", "מעלית"),
            ("מחסן", "מחסן"),
            ("מיזוג", "מיזוג"),
            ("מזגן", "מיזוג"),
            ("ממד", "ממ״ד"),
            ('ממ"ד', "ממ״ד"),
            ("גישה לנכים", "גישה_לנכים"),
            ("חיות מחמד", "חיות_מחמד"),
        ):
            if phrase in text:
                params[field] = True

        for condition in ("דורש שיפוץ", "משופץ", "שמור", "חדש"):
            if condition in text:
                params["מצב_נכס"] = condition
                break

        if "ללא ריהוט" in text:
            params["ריהוט"] = "ללא"
        elif "מרוהט" in text or "ריהוט מלא" in text:
            params["ריהוט"] = "מלא"
        elif "ריהוט חלקי" in text:
            params["ריהוט"] = "חלקי"

        proximity = [
            value
            for value in self.taxonomy.categories[REAL_ESTATE]["מאפיינים_כלליים"]["קרבה"]
            if _phrase_in_text(text, value)
        ]
        if proximity:
            params["קרבה"] = proximity

        price = extract_price(text, implicit=True, category=REAL_ESTATE)
        if price:
            params["מחיר"] = price

        if "מיידי" in text or "כניסה מיידית" in text:
            params["תאריך_כניסה"] = "מיידי"

        return params

    def _extract_vehicle(self, text: str) -> dict[str, Any]:
        params: dict[str, Any] = {}
        manufacturer = None
        for candidate, aliases in VEHICLE_MANUFACTURER_ALIASES.items():
            if _any_phrase(text, aliases) or _any_phrase_fuzzy(text, aliases):
                manufacturer = candidate
                params["יצרן"] = candidate
                break

        model_owner = None
        for model, owner in sorted(
            self.taxonomy.vehicle_models().items(), key=lambda item: len(item[0]), reverse=True
        ):
            if str(model).isdigit() and not _phrase_in_text(text, f"{owner} {model}"):
                continue
            model_aliases = self.knowledge.multilingual_aliases().get(str(model), [])
            if (
                _phrase_in_text(text, model)
                or _fuzzy_phrase_in_text(text, str(model))
                or _any_phrase(text, model_aliases)
                or _any_phrase_fuzzy(text, model_aliases)
            ):
                params["דגם"] = model
                model_owner = owner
                if not manufacturer:
                    params["יצרן"] = owner
                break

        if model_owner and model_owner in self.taxonomy.vehicle_manufacturers():
            submodels = (
                self.taxonomy.vehicle_manufacturers()
                .get(model_owner, {})
                .get("תתי_דגמים", {})
                .get(params.get("דגם"), [])
            )
            for submodel in submodels:
                if _phrase_in_text(text, submodel):
                    params["תת_דגם"] = submodel
                    break
        trim = self._extract_vehicle_trim(
            text,
            params.get("יצרן"),
            params.get("דגם"),
        )
        if trim:
            params["תת_דגם"] = trim

        years = _extract_years(text)
        if len(years) >= 2:
            params["שנה"] = {"min": min(years), "max": max(years)}
        elif len(years) == 1:
            params["שנה"] = years[0]

        hand_match = re.search(r"\bיד\s*(\d{1,2})\b", text)
        if hand_match:
            params["יד"] = int(hand_match.group(1))

        km_range = extract_metric_range(text, ("ק״מ", "קילומטר"), default_multiplier=True)
        if km_range:
            params["ק״מ"] = km_range

        engine_match = re.search(r"(?:נפח(?:\s*מנוע)?\s*)?(\d{3,4})\s*(?:סמ״ק|סמק)", text)
        if engine_match:
            params["נפח_מנוע_סמ״ק"] = int(engine_match.group(1))

        hp_match = re.search(r"(\d{2,4})\s*(?:כ״ס|כס|כוח סוס)", text)
        if hp_match:
            params["הספק_כ״ס"] = int(hp_match.group(1))

        if "אוטומט" in text:
            params["תיבת_הילוכים"] = "אוטומטית"
        elif "ידנית" in text:
            params["תיבת_הילוכים"] = "ידנית"
        elif "רובוטית" in text:
            params["תיבת_הילוכים"] = "רובוטית"
        elif "cvt" in text:
            params["תיבת_הילוכים"] = "CVT"

        for fuel in ("היברידי נטען", "היברידי", "חשמלי", "בנזין", "דיזל", "גז"):
            if fuel in text or (fuel == "חשמלי" and "חשמלית" in text):
                params["סוג_דלק"] = fuel
                break

        if "חשמלי" in text or "היברידי" in text:
            params["סוגי_רכב"] = ["רכב היברידי/חשמלי"]
        else:
            for vehicle_type in self.taxonomy.categories[VEHICLES]["סוגי_רכב"]:
                if _phrase_in_text(text, vehicle_type):
                    params["סוגי_רכב"] = [vehicle_type]
                    break

        color = _extract_color(text, allowed=set(self.taxonomy.vehicle_colors()))
        if color:
            params["צבע"] = color

        for owner in self.taxonomy.categories[VEHICLES]["מאפיינים_כלליים"]["בעלות"]:
            if _phrase_in_text(text, owner):
                params["בעלות"] = owner
                break

        safety = [
            value
            for value in self.taxonomy.categories[VEHICLES]["מאפיינים_כלליים"][
                "מערכות_בטיחות"
            ]
            if _phrase_in_text(text, value)
        ]
        if safety:
            params["מערכות_בטיחות"] = safety

        accessories = [
            value
            for value in self.taxonomy.categories[VEHICLES]["מאפיינים_כלליים"]["אבזור"]
            if _phrase_in_text(text, value)
        ]
        if accessories:
            params["אבזור"] = accessories

        price = extract_price(text, implicit="ק״מ" not in text, category=VEHICLES)
        if price:
            params["מחיר"] = price
        return params

    def _extract_second_hand(self, text: str) -> dict[str, Any]:
        params: dict[str, Any] = {}
        candidate = self._best_second_hand_candidate(text)
        phrase_match = self._best_second_hand_phrase(text)
        subcategory_spec: dict[str, Any] = {}
        if phrase_match:
            params["סקטור"] = phrase_match.sector
            params["תת_קטגוריה"] = phrase_match.subcategory
            if phrase_match.brand:
                params["מותג"] = phrase_match.brand
        elif candidate:
            sector, subcategory, _score = candidate
            params["סקטור"] = sector
            params["תת_קטגוריה"] = subcategory
            for item in self.taxonomy.second_hand_subcategories():
                if item.sector == sector and item.subcategory == subcategory:
                    subcategory_spec = item.spec
                    break

        brand = self._extract_second_hand_brand(text, subcategory_spec)
        if brand:
            params["מותג"] = brand

        model = _extract_phone_or_laptop_model(text)
        if model:
            params["דגם"] = model

        condition = _extract_condition(text)
        if condition:
            params["מצב"] = condition

        color = _extract_color(text)
        if color:
            params["צבע"] = color

        storage = _extract_storage(text)
        if storage and params.get("תת_קטגוריה") == "טלפונים_סלולריים":
            params["נפח_אחסון"] = storage
        elif storage and params.get("תת_קטגוריה") == "מחשבים_ניידים":
            params["אחסון_GB"] = 1024 if storage == "1TB" else int(storage.replace("GB", ""))

        ram = re.search(r"(\d{1,3})\s*(?:gb|גיגה)?\s*(?:ram|ראם|זיכרון)", text)
        if ram and params.get("תת_קטגוריה") == "מחשבים_ניידים":
            params["זיכרון_RAM"] = int(ram.group(1))

        for processor in ("i5", "i7", "i9", "ryzen 5", "ryzen 7"):
            if processor in text:
                params["מעבד"] = processor.title() if processor.startswith("ryzen") else processor
                break

        size = re.search(r"(\d{2,3})\s*(?:אינץ|אינצ|inch|\")", text)
        if size and params.get("תת_קטגוריה") == "טלוויזיות":
            params["גודל_אינצ׳"] = int(size.group(1))

        for field in ("טכנולוגיה", "רזולוציה", "סוג", "גודל_גלגל", "חומר"):
            values = subcategory_spec.get(field)
            if isinstance(values, list):
                for value in values:
                    if _phrase_in_text(text, value) or _fuzzy_phrase_in_text(text, str(value)):
                        params[field] = value
                        break
        if (
            params.get("תת_קטגוריה") == "אופניים"
            and "סוג" not in params
            and ("חשמלי" in text or "electric" in text)
        ):
            params["סוג"] = "חשמליים"

        for field in ("סוגי_רהיט",):
            values = subcategory_spec.get(field)
            if isinstance(values, list):
                matches = [value for value in values if _phrase_in_text(text, value)]
                if matches:
                    params[field] = matches

        region = _extract_region(text)
        if region:
            params["אזור"] = region
        city = self._extract_city(text)
        if city:
            params["עיר"] = city

        years = _extract_years(text)
        if len(years) == 1 and params.get("תת_קטגוריה") != "טלפונים_סלולריים":
            params["שנת_ייצור"] = years[0]

        price = extract_price(text, implicit=True, category=SECOND_HAND)
        if price:
            params["מחיר"] = price
        return params

    def _extract_city(self, text: str) -> str | None:
        aliases_by_city = {
            **self.taxonomy.city_aliases(),
            **self.knowledge.city_aliases(),
        }
        for city, aliases in aliases_by_city.items():
            prefixed_aliases = ["ב" + alias for alias in aliases]
            if (
                _any_phrase(text, aliases)
                or _any_phrase(text, prefixed_aliases)
                or _any_phrase_fuzzy(text, aliases)
                or _any_phrase_fuzzy(text, prefixed_aliases)
            ):
                return city
        return None

    def _extract_neighborhood(self, text: str) -> str | None:
        for neighborhood, match in self.knowledge.neighborhood_aliases().items():
            aliases = list(match.aliases)
            if _any_phrase(text, aliases) or _any_phrase_fuzzy(text, aliases):
                return neighborhood
        return None

    def _extract_street(self, text: str) -> str | None:
        for street, match in self.knowledge.street_aliases().items():
            aliases = list(match.aliases)
            if _any_phrase(text, aliases) or _any_phrase_fuzzy(text, aliases):
                return street
        return None

    def _city_for_neighborhood(self, neighborhood: str) -> str | None:
        match = self.knowledge.neighborhood_aliases().get(neighborhood)
        return match.city if match else None

    def _city_for_street(self, street: str) -> str | None:
        match = self.knowledge.street_aliases().get(street)
        return match.city if match else None

    def _extract_vehicle_trim(
        self, text: str, manufacturer: str | None, model: str | None
    ) -> str | None:
        trims_by_manufacturer = self.knowledge.vehicle_trims()
        manufacturers = [manufacturer] if manufacturer else list(trims_by_manufacturer.keys())
        for current_manufacturer in manufacturers:
            models = trims_by_manufacturer.get(current_manufacturer or "", {})
            candidate_models = [model] if model else list(models.keys())
            for current_model in candidate_models:
                for trim, aliases in models.get(current_model or "", {}).items():
                    phrases = [trim, *aliases]
                    if _any_phrase(text, phrases) or _any_phrase_fuzzy(text, phrases):
                        return trim
        return None

    def _best_second_hand_phrase(self, text: str):
        for phrase in self.knowledge.second_hand_phrases():
            if _any_phrase(text, list(phrase.aliases)) or _any_phrase_fuzzy(
                text, list(phrase.aliases)
            ):
                return phrase
        return None

    def _best_second_hand_candidate(self, text: str) -> tuple[str, str, float] | None:
        best: tuple[str, str, float] | None = None
        for item in self.taxonomy.second_hand_subcategories():
            score = 0.0
            subcategory_phrase = item.subcategory.replace("_", " ")
            if _phrase_in_text(text, item.sector):
                score += 1
            if _phrase_in_text(text, subcategory_phrase):
                score += 4
            for alias in SECOND_HAND_ALIASES.get((item.sector, item.subcategory), []):
                if _phrase_in_text(text, alias) or _fuzzy_phrase_in_text(text, alias):
                    score += 5
            for key, values in item.spec.items():
                if not isinstance(values, list):
                    continue
                for value in values:
                    if _phrase_in_text(text, str(value)):
                        score += 2 if key == "מותגים" else 1
            if score > 0 and (best is None or score > best[2]):
                best = (item.sector, item.subcategory, score)
        return best

    def _extract_second_hand_brand(
        self, text: str, subcategory_spec: dict[str, Any]
    ) -> str | None:
        for alias, brand in BRAND_ALIASES.items():
            if _phrase_in_text(text, alias):
                return brand
        candidates: list[str] = []
        if subcategory_spec:
            candidates.extend(str(value) for value in subcategory_spec.get("מותגים", []))
        candidates.extend(self.taxonomy.second_hand_brands().keys())
        for brand in sorted(set(candidates), key=len, reverse=True):
            if _phrase_in_text(text, brand):
                return brand
        return None

    @staticmethod
    def _confidence(score: float, margin: float, param_count: int, security: bool) -> float:
        if score <= 0:
            return 0.25
        confidence = 0.42 + min(0.32, score * 0.035) + min(0.18, param_count * 0.035)
        if margin < 2:
            confidence -= 0.12
        if security:
            confidence = min(confidence, 0.84)
        return max(0.05, min(0.96, confidence))


def extract_price(text: str, *, implicit: bool, category: str) -> dict[str, int] | None:
    between = re.search(
        rf"בין\s+{AMOUNT}\s+(?:ל|לבין|-|עד)\s+"
        rf"(?P<num2>\d+(?:[.,]\d+)?|מיליון)(?:\s*(?P<unit2>אלף|אלפים|מיליון|k))?"
        r"(?:\s*ש״ח)?",
        text,
    )
    if between:
        first = _amount_value(between.group("num"), between.group("unit"))
        second = _amount_value(between.group("num2"), between.group("unit2"))
        if first is not None and second is not None and _price_context_ok(text, between, category):
            return {"min": min(first, second), "max": max(first, second)}

    directive = re.search(
        rf"(?P<op>עד|מקסימום|מקס|מתחת|פחות|מעל|לפחות|מינימום|יותר)"
        rf"\s*(?:ל|מ)?\s*{AMOUNT}(?:\s*ש״ח)?",
        text,
    )
    if directive and (implicit or "ש״ח" in directive.group(0)):
        if not _price_context_ok(text, directive, category):
            return None
        value = _amount_value(directive.group("num"), directive.group("unit"))
        if value is None:
            return None
        op = directive.group("op")
        if op in MAX_WORDS:
            return {"max": value}
        return {"min": value}

    explicit = re.search(rf"{AMOUNT}\s*ש״ח", text)
    if explicit:
        value = _amount_value(explicit.group("num"), explicit.group("unit"))
        if value is not None:
            return {"max": value}
    return None


def extract_metric_range(
    text: str, units: tuple[str, ...], *, default_multiplier: bool = False
) -> dict[str, int] | int | None:
    unit_pattern = "|".join(re.escape(unit) for unit in units)
    directive = re.search(
        rf"(?P<op>עד|מתחת|פחות|מעל|לפחות|מינימום|יותר)?\s*{AMOUNT}\s*(?:{unit_pattern})",
        text,
    )
    if not directive:
        return None
    value = _amount_value(directive.group("num"), directive.group("unit"))
    if value is None:
        return None
    op = directive.group("op")
    if op in MAX_WORDS or op is None:
        return {"max": value}
    return {"min": value}


def _amount_value(num: str | None, unit: str | None) -> int | None:
    if not num:
        return None
    if num == "מיליון":
        value = 1_000_000.0
    else:
        value = float(num.replace(",", "."))
    unit = unit or ""
    if unit in {"אלף", "אלפים", "k"}:
        value *= 1_000
    elif unit == "מיליון":
        value *= 1_000_000
    return int(value)


def _price_context_ok(text: str, match: re.Match[str], category: str) -> bool:
    after = text[match.end() : match.end() + 12]
    if category == VEHICLES and ("ק״מ" in after or "קילומטר" in after):
        return False
    return True


def _extract_years(text: str) -> list[int]:
    years = [int(value) for value in re.findall(r"\b(19[8-9]\d|20[0-2]\d)\b", text)]
    return [year for year in years if 1980 <= year <= 2026]


def _extract_color(text: str, allowed: set[str] | None = None) -> str | None:
    for alias, color in COLOR_ALIASES.items():
        if _phrase_in_text(text, alias) and (allowed is None or color in allowed):
            return color
    return None


def _extract_condition(text: str) -> str | None:
    for condition in ("כמו חדש", "לחלפים", "משומש", "חדש"):
        if condition in text:
            return condition
    return None


def _extract_storage(text: str) -> str | None:
    if re.search(r"\b1\s*(?:tb|טרה)\b", text):
        return "1TB"
    match = re.search(r"\b(64|128|256|512)\s*(?:gb|גיגה|ג'יגה|גיגהבייט)\b", text)
    if match:
        return f"{match.group(1)}GB"
    return None


def _extract_phone_or_laptop_model(text: str) -> str | None:
    iphone = re.search(
        r"(?:אייפון|iphone)\s*(?P<num>\d{1,2})\s*(?P<variant>פרו מקס|פרו|מקס|plus|פלוס|mini)?",
        text,
    )
    if iphone:
        variant = (iphone.group("variant") or "").strip()
        mapping = {
            "פרו": "Pro",
            "פרו מקס": "Pro Max",
            "מקס": "Max",
            "plus": "Plus",
            "פלוס": "Plus",
            "mini": "Mini",
        }
        suffix = mapping.get(variant, "")
        return f"iPhone {iphone.group('num')}{(' ' + suffix) if suffix else ''}"

    galaxy = re.search(r"(?:גלקסי|galaxy)\s*(?P<model>[a-z]?\d{1,2}(?:\s*ultra|\s*plus)?)", text)
    if galaxy:
        return "Galaxy " + galaxy.group("model").upper()

    macbook = re.search(r"(?:מקבוק|macbook)\s*(air|pro|אייר|פרו)?", text)
    if macbook:
        variant = macbook.group(1)
        mapping = {"air": "Air", "pro": "Pro", "אייר": "Air", "פרו": "Pro"}
        return "MacBook" + (f" {mapping.get(variant, variant.title())}" if variant else "")
    return None


def _extract_region(text: str) -> str | None:
    for region in ("ירושלים", "חיפה", "שרון", "שפלה", "מרכז", "צפון", "דרום"):
        if _phrase_in_text(text, region) or _phrase_in_text(text, "ב" + region):
            return region
    return None


def _number_after_or_before(text: str, noun_pattern: str) -> float | None:
    before = re.search(rf"(\d+(?:[.,]\d+)?)\s*{noun_pattern}", text)
    if before:
        return float(before.group(1).replace(",", "."))
    after = re.search(rf"{noun_pattern}\s*(\d+(?:[.,]\d+)?)", text)
    if after:
        return float(after.group(1).replace(",", "."))
    return None


def _int_if_whole(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _phrase_in_text(text: str, phrase: str) -> bool:
    return phrase_in_text(text, phrase)


def _any_phrase(text: str, phrases: list[str] | tuple[str, ...]) -> bool:
    return any_phrase(text, phrases)


def _any_phrase_fuzzy(
    text: str, phrases: list[str] | tuple[str, ...], *, threshold: float = 0.88
) -> bool:
    return any_phrase_fuzzy(text, phrases, threshold=threshold)


def _fuzzy_phrase_in_text(text: str, phrase: str, *, threshold: float = 0.88) -> bool:
    return fuzzy_phrase_in_text(text, phrase, threshold=threshold)


def _dedupe(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    output: list[Any] = []
    for value in values:
        marker = repr(value)
        if marker not in seen:
            seen.add(marker)
            output.append(value)
    return output


def _looks_like_taxonomy_echo(params: dict[str, Any]) -> bool:
    """Detect model outputs that copied broad taxonomy defaults."""
    if len(params) > 8:
        return True
    broad_fields = 0
    for value in params.values():
        if isinstance(value, list) and len(value) > 4:
            broad_fields += 1
        elif (
            isinstance(value, dict)
            and set(value) == {"min", "max"}
            and _is_broad_range(value.get("min"), value.get("max"))
        ):
            broad_fields += 1
    return broad_fields >= 3


def _is_broad_range(minimum: Any, maximum: Any) -> bool:
    if not isinstance(minimum, (int, float)) or not isinstance(maximum, (int, float)):
        return False
    return minimum <= 0 and maximum >= 100_000
