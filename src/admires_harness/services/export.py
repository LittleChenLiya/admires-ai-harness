"""Bounded, redacted JSON export of a run's persisted events.

The export is a pure read over the event store: it never invokes a model and
never mutates run state. The document is versioned via ``schema_version`` so
consumers can pin a stable serialization contract.

Redaction policy (applied recursively to every event payload, including
nested objects and lists):

* Dict keys whose normalized name (lowercased, ``-``/space -> ``_``) exactly
  matches a known sensitive name (``api_key``, ``token``, ``secret``,
  ``password``, ``authorization``, ``cookie``, ``private_key``, ...) have
  their entire value replaced with ``[redacted]``.
* String values are scanned for common secret shapes (``Bearer ...``
  headers, ``sk-...`` API keys, PEM private-key blocks) and the matching
  substring is replaced with ``[redacted]``.

Exact key matching keeps operational fields such as ``input_tokens`` /
``output_tokens`` intact while still catching credentials.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from admires_harness.contracts import JsonValue, RunEvent
from admires_harness.observability import RunView

TRACE_EXPORT_SCHEMA_VERSION = "1"
TRACE_EXPORT_KIND = "admires-harness/trace-export"
REDACTED = "[redacted]"
DEFAULT_EXPORT_LIMIT = 1_000
MAX_EXPORT_LIMIT = 5_000

_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "id_token",
        "token",
        "secret",
        "client_secret",
        "password",
        "passwd",
        "authorization",
        "proxy_authorization",
        "credential",
        "credentials",
        "private_key",
        "session_token",
        "cookie",
        "set_cookie",
        "x_api_key",
    }
)

_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN[A-Z ]*PRIVATE KEY-----.*?-----END[A-Z ]*PRIVATE KEY-----", re.DOTALL),
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
)


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace("-", "_").replace(" ", "_")


def redact_value(value: JsonValue) -> JsonValue:
    """Recursively redact sensitive keys and secret-looking strings."""

    if isinstance(value, dict):
        return {
            key: REDACTED if _normalize_key(key) in _SENSITIVE_KEYS else redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, str):
        for pattern in _SECRET_PATTERNS:
            value = pattern.sub(REDACTED, value)
        return value
    return value


def build_trace_export(
    view: RunView,
    events: list[RunEvent],
    *,
    limit: int = DEFAULT_EXPORT_LIMIT,
) -> dict[str, JsonValue]:
    """Serialize a run's events in persisted order with a bounded, redacted payload."""

    if not 1 <= limit <= MAX_EXPORT_LIMIT:
        raise ValueError(f"export limit must be between 1 and {MAX_EXPORT_LIMIT}")
    generated_at = datetime.now(UTC)
    ordered = sorted(events, key=lambda event: event.sequence)
    selected = ordered[:limit]
    serialized: list[JsonValue] = []
    for event in selected:
        document = event.model_dump(mode="json")
        document["payload"] = redact_value(document.get("payload", {}))
        serialized.append(document)
    return {
        "schema_version": TRACE_EXPORT_SCHEMA_VERSION,
        "kind": TRACE_EXPORT_KIND,
        "generated_at": generated_at.isoformat(),
        "run": {
            "id": str(view.run_id),
            "demo": view.demo,
            "status": view.status.value,
            "created_at": view.started_at.astimezone(UTC).isoformat(),
            "updated_at": (view.ended_at or generated_at).astimezone(UTC).isoformat(),
            "error": view.error,
        },
        "redaction": {
            "placeholder": REDACTED,
            "strategy": "exact sensitive key match plus secret pattern scan, applied recursively",
        },
        "event_count": len(selected),
        "total_events": len(ordered),
        "truncated": len(selected) < len(ordered),
        "events": serialized,
    }
