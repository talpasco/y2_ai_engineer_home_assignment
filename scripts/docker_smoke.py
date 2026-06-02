from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request


BASE_URL = "http://localhost:8000"


def main() -> int:
    run(["docker", "info"], quiet=True)

    print("Building and starting Docker Compose stack...")
    run(["docker", "compose", "up", "--build", "-d"])

    print("Waiting for API health...")
    wait_for_health()

    print("Running parse smoke request...")
    result = request_json(
        "POST",
        f"{BASE_URL}/parse",
        {"q": "דירת 3 חדרים בירושלים עד מליון שח"},
    )
    if not result.get("category"):
        raise RuntimeError(f"Parse response did not include a category: {result}")
    if float(result.get("confidence", 0.0)) < 0.75:
        raise RuntimeError(f"Parse confidence is unexpectedly low: {result}")

    print("Checking cost guardrail query...")
    guarded = request_json("POST", f"{BASE_URL}/parse", {"q": "רכב אמריקאי גדול"})
    resolution = guarded.get("resolution") or {}
    if resolution.get("source") != "rules":
        raise RuntimeError(f"Expected rules path for guardrail query: {guarded}")
    if resolution.get("input_tokens", 0) or resolution.get("output_tokens", 0):
        raise RuntimeError(f"Guardrail query spent model tokens: {guarded}")

    print("Checking metrics...")
    metrics = request_text(f"{BASE_URL}/metrics", accept="text/plain")
    if "yad2_requests_total" not in metrics:
        raise RuntimeError("Metrics endpoint did not include yad2_requests_total")

    print("Docker smoke test passed.")
    print("Open http://localhost:8000 for the dashboard, /docs, /health, and /metrics.")
    return 0


def run(command: list[str], *, quiet: bool = False) -> None:
    kwargs = {
        "check": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
    subprocess.run(command, **kwargs)


def wait_for_health() -> None:
    deadline = time.monotonic() + 90
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            payload = request_json("GET", f"{BASE_URL}/health")
            if payload.get("status") == "ok":
                return
        except Exception as exc:  # noqa: BLE001 - surfaced after timeout
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"API did not become healthy: {last_error}")


def request_json(method: str, url: str, payload: dict | None = None) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with {exc.code}: {body}") from exc


def request_text(url: str, *, accept: str) -> str:
    request = urllib.request.Request(url, headers={"Accept": accept})
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read().decode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
