from __future__ import annotations

import unittest

from app.models import QueryValidationError
from app.parser import SearchParser, _looks_like_taxonomy_echo
from app.taxonomy import REAL_ESTATE, SECOND_HAND, VEHICLES
from tests.helpers import TAXONOMY, make_parser


class ParserTests(unittest.TestCase):
    def test_real_estate_example(self) -> None:
        result = make_parser().parse("דירת 3 חדרים בירושלים עד מליון שח")

        self.assertEqual(result.category, REAL_ESTATE)
        self.assertEqual(result.params["עיר"], "ירושלים")
        self.assertEqual(result.params["מס׳_חדרים"], 3)
        self.assertEqual(result.params["מחיר"], {"max": 1_000_000})
        self.assertIn("דירה", result.params["סוגי_נכס"])
        self.assertGreaterEqual(result.confidence, 0.7)

    def test_vehicle_example(self) -> None:
        result = make_parser().parse("טויוטה קורולה 2018-2021 עד 70 אלף שח צבע לבן")

        self.assertEqual(result.category, VEHICLES)
        self.assertEqual(result.params["יצרן"], "טויוטה")
        self.assertEqual(result.params["דגם"], "קורולה")
        self.assertEqual(result.params["שנה"], {"min": 2018, "max": 2021})
        self.assertEqual(result.params["מחיר"], {"max": 70_000})
        self.assertEqual(result.params["צבע"], "לבן")

    def test_second_hand_iphone_example(self) -> None:
        result = make_parser().parse("אייפון 13 פרו 256 ג׳יגה כחול כמו חדש עד 2500")

        self.assertEqual(result.category, SECOND_HAND)
        self.assertEqual(result.params["סקטור"], "אלקטרוניקה")
        self.assertEqual(result.params["תת_קטגוריה"], "טלפונים_סלולריים")
        self.assertEqual(result.params["מותג"], "אפל")
        self.assertEqual(result.params["דגם"], "iPhone 13 Pro")
        self.assertEqual(result.params["נפח_אחסון"], "256GB")
        self.assertEqual(result.params["מצב"], "כמו חדש")
        self.assertEqual(result.params["מחיר"], {"max": 2500})

    def test_vehicle_km_is_not_price(self) -> None:
        result = make_parser().parse("יונדאי טוסון יד 2 עד 90 אלף קמ אוטומטית")

        self.assertEqual(result.category, VEHICLES)
        self.assertEqual(result.params["יצרן"], "יונדאי")
        self.assertEqual(result.params["דגם"], "טוסון")
        self.assertEqual(result.params["ק״מ"], {"max": 90_000})
        self.assertNotIn("מחיר", result.params)

    def test_furniture_query(self) -> None:
        result = make_parser().parse("ספה עור לסלון במרכז עד 1200")

        self.assertEqual(result.category, SECOND_HAND)
        self.assertEqual(result.params["סקטור"], "ריהוט")
        self.assertEqual(result.params["תת_קטגוריה"], "סלון")
        self.assertEqual(result.params["סוגי_רהיט"], ["ספה"])
        self.assertEqual(result.params["חומר"], "עור")
        self.assertEqual(result.params["מחיר"], {"max": 1200})

    def test_typo_normalization(self) -> None:
        result = make_parser().parse("דירת 4 חדרים בתלאביב עם דופליקס עד 3 מיליון שח")

        self.assertEqual(result.category, REAL_ESTATE)
        self.assertEqual(result.params["עיר"], "תל אביב-יפו")
        self.assertIn("דופלקס", result.params["סוגי_נכס"])
        self.assertEqual(result.params["מחיר"], {"max": 3_000_000})

    def test_fuzzy_candidate_matching(self) -> None:
        result = make_parser().parse("טויוטא קורולהה 2020 עד 80 אלף שח")

        self.assertEqual(result.category, VEHICLES)
        self.assertEqual(result.params["יצרן"], "טויוטה")
        self.assertEqual(result.params["דגם"], "קורולה")
        self.assertEqual(result.params["שנה"], 2020)
        self.assertEqual(result.params["מחיר"], {"max": 80_000})

    def test_fuzzy_city_matching(self) -> None:
        result = make_parser().parse("דירת 2 חדרים בתל אבייב עד 2 מיליון שח")

        self.assertEqual(result.category, REAL_ESTATE)
        self.assertEqual(result.params["עיר"], "תל אביב-יפו")

    def test_long_tail_neighborhood_and_street_normalization(self) -> None:
        result = make_parser().parse("דירת 2 חדרים בפלורנטין ליד דיזנגוף עד 2 מיליון")

        self.assertEqual(result.category, REAL_ESTATE)
        self.assertEqual(result.params["עיר"], "תל אביב-יפו")
        self.assertEqual(result.params["שכונה"], "פלורנטין")
        self.assertEqual(result.params["רחוב"], "דיזנגוף")

    def test_vehicle_trim_dictionary(self) -> None:
        result = make_parser().parse("יונדאי טוסון פרימיום 2021 יד 2")

        self.assertEqual(result.category, VEHICLES)
        self.assertEqual(result.params["יצרן"], "יונדאי")
        self.assertEqual(result.params["דגם"], "טוסון")
        self.assertEqual(result.params["תת_דגם"], "Premium")

    def test_multilingual_aliases_and_transliteration(self) -> None:
        result = make_parser().parse("toyota korola hybrid 2019 עד 75000")

        self.assertEqual(result.category, VEHICLES)
        self.assertEqual(result.params["יצרן"], "טויוטה")
        self.assertEqual(result.params["דגם"], "קורולה")
        self.assertEqual(result.params["תת_דגם"], "Hybrid")

    def test_generic_vehicle_word_classifies_as_vehicle(self) -> None:
        for query in ("רכב חזק יחסית חדש", "רכב אמריקאי יחסית חדש", "מכונית משפחתית"):
            result = make_parser().parse(query)
            self.assertEqual(result.category, VEHICLES, query)

    def test_car_accessory_is_not_a_vehicle(self) -> None:
        result = make_parser().parse("מזגן חדש לרכב עד 1500")
        self.assertEqual(result.category, SECOND_HAND)

    def test_second_hand_phrase_outside_taxonomy(self) -> None:
        result = make_parser().parse("קורקינט חשמלי משומש עד 1800")

        self.assertEqual(result.category, SECOND_HAND)
        self.assertEqual(result.params["סקטור"], "ספורט_וקמפינג")
        self.assertEqual(result.params["תת_קטגוריה"], "אופניים")
        self.assertEqual(result.params["סוג"], "חשמליים")

    def test_empty_query_rejected(self) -> None:
        with self.assertRaises(QueryValidationError):
            make_parser().parse("   ")

    def test_all_output_fields_are_allowed(self) -> None:
        parser = make_parser()
        queries = [
            "דירת 3 חדרים בירושלים עד מליון שח",
            "טויוטה קורולה 2018-2021 עד 70 אלף שח צבע לבן",
            "אייפון 13 פרו 256 גיגה כחול כמו חדש עד 2500",
            "ספה עור לסלון במרכז עד 1200",
        ]
        for query in queries:
            result = parser.parse(query)
            allowed = TAXONOMY.allowed_fields(result.category, result.params)
            self.assertFalse(set(result.params) - allowed, query)

    def test_confident_generic_vehicle_query_does_not_call_llm(self) -> None:
        class Fallback:
            called = False

            def parse(self, *args, **kwargs):
                self.called = True
                raise AssertionError("LLM fallback should not be called")

        fallback = Fallback()
        parser = SearchParser(
            TAXONOMY,
            llm_fallback=fallback,
            llm_confidence_threshold=0.60,
        )

        result = parser.parse(
            "\u05e8\u05db\u05d1 \u05d0\u05de\u05e8\u05d9\u05e7\u05d0\u05d9 \u05d2\u05d3\u05d5\u05dc"
        )

        self.assertEqual(result.category, VEHICLES)
        self.assertFalse(fallback.called)

    def test_taxonomy_echo_detection_rejects_broad_llm_defaults(self) -> None:
        self.assertTrue(
            _looks_like_taxonomy_echo(
                {
                    "year": {"min": 1980, "max": 2025},
                    "km": {"min": 0, "max": 1_000_000},
                    "price": {"min": 0, "max": 5_000_000},
                    "color": ["white", "black", "gray", "blue", "red"],
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
