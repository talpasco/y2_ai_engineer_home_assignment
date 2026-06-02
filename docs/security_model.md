# Security Model

This service is designed around OWASP API Security and OWASP LLM Application risks:

- OWASP API4:2023 - unrestricted resource consumption
- OWASP automated threat patterns such as scraping and high-volume automated access
- OWASP LLM01 - prompt injection
- OWASP LLM02 / LLM05 style risks around insecure or improper output handling
- LLM unbounded consumption / cost abuse

References:

- https://owasp.org/API-Security/editions/2023/en/0x11-t10/
- https://owasp.org/www-project-top-10-for-large-language-model-applications
- https://owasp.org/www-project-automated-threats-to-web-applications/assets/oats/EN/OAT-011_Scraping

## Trust Boundaries

```text
Internet / client
  -> FastAPI middleware
  -> parser pipeline
  -> optional OpenAI API
  -> response validator
  -> client
```

The user query is always untrusted. Model output is also untrusted.

## Implemented Controls

### API Abuse Controls

- Maximum request body size via `MAX_REQUEST_BYTES`.
- Per-client token-bucket rate limiting.
- Stricter throttling for suspicious user agents.
- Optional `x-api-key` auth via `REQUIRE_API_KEY=true`.
- Request IDs for traceability.
- Security headers:
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: DENY`
  - `Referrer-Policy: no-referrer`
  - `Permissions-Policy`
  - Content Security Policy

### Anti-Bot / Automation Controls

The local service implements server-side controls suitable for an API:

- suspicious user-agent detection
- lower quota for suspicious clients
- request size caps
- structured security logs

Production deployment should add edge controls:

- edge WAF / bot-management controls
- Turnstile or equivalent challenge for public demo pages
- per-account and per-token quotas
- IP reputation
- anomaly detection on request velocity and query entropy
- circuit breakers for model fallback

### LLM Controls

- LLM fallback is off by default.
- LLM fallback requires `OPENAI_API_KEY`.
- Suspicious/prompt-injection text disables LLM fallback.
- The model receives a compact taxonomy slice, not broad system context.
- The model has no tools and no external actions.
- Model output is post-validated against category and field allowlists.
- Token usage and estimated cost are tracked.

### Logging Controls

- Logs are JSON-structured.
- Phone numbers and emails are redacted from query logs.
- Long queries are truncated in logs.
- No secrets are logged.

## Residual Risks

- In-memory rate limits reset on process restart and are per-instance. Production should use Redis or edge quotas.
- Suspicious user-agent detection is not sufficient against advanced bots. It is a local demonstration control.
- Prompt injection is mitigated by limiting model authority, but not eliminated.
- A real Yad2 deployment should use service-to-service auth and private networking for backend-only APIs.

## Security Tests

Covered by tests:

- prompt-injection strings are treated as search text
- suspicious payload text is flagged
- unicode/control-character noise is stripped
- forbidden fields are not emitted
- query logs redact email and phone-like values
- API key comparison uses constant-time comparison
- token bucket rate limiting denies excess requests
- security headers are present

Run:

```bash
python -m unittest discover -v
python scripts/run_redteam.py
```
