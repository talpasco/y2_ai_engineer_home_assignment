from __future__ import annotations

import unittest


try:
    from fastapi.testclient import TestClient

    from app.main import app
except ModuleNotFoundError:  # pragma: no cover - local stdlib-only verification path
    TestClient = None
    app = None


@unittest.skipIf(TestClient is None, "FastAPI dependencies are not installed")
class ApiTests(unittest.TestCase):
    def test_parse_endpoint(self) -> None:
        client = TestClient(app)
        response = client.post("/parse", json={"q": "דירת 3 חדרים בירושלים עד מליון שח"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["category"], "נדל״ן")
        self.assertEqual(body["params"]["מחיר"], {"max": 1_000_000})

    def test_metrics_endpoint(self) -> None:
        client = TestClient(app)
        client.post("/parse", json={"q": "אייפון 13 פרו 256 גיגה עד 2500"})
        response = client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertIn("yad2_requests_total", response.text)

    def test_security_headers_are_present(self) -> None:
        client = TestClient(app)
        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        self.assertIn("x-request-id", response.headers)


if __name__ == "__main__":
    unittest.main()
