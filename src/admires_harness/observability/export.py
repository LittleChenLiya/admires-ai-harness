"""Bounded, redacted trace export built from persisted run events.

Export is read-only: it never invokes a model, runs a tool, or mutates a run.
Redaction is key-based and recursive. Any mapping key (at any depth) whose
normalized name contains a sensitive marker has its entire value replaced with
``[REDACTED]`` before serialization, so credentials nested inside tool
arguments, approval payloads, or model metadata cannot leak into an export.
"""

from __future__ import annotations

import re
from typing import Final

from admires_harness.contracts import JsonValue, RunEvent, TraceExport
from admires_harness.observability.projections import RunView

REDACTED: Final = "[REDACTED]"

_SENSITIVE_MARKERS: Final = (
    "secret",
    "token",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "private_key",
    "cookie",
    "session",
)

_MARKER_PATTERN: Final = re.compile("|".join(_SENSITIVE_MARKERS))

DEFAULT_MAX_EVENTS: Final = 1_000
MAX_EVENTS_LIMIT: Final = 5_000


def _is_sensitive_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_").replace(" ", "_")
    return _MARKER_PATTERN.search(normalized) is not None


def redact_value(value: JsonValue, *, _key: str | None = None) -> JsonValue:
    """Recursively redact sensitive mapping values at any nesting depth."""

    if _key is not None and _is_sensitive_key(_key):
        return REDACTED
    if isinstance(value, dict):
        return {key: redact_value(item, _key=key) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    return value


def redact_event(event: RunEvent) -> RunEvent:
    redacted = redact_value(event.payload)
    return event.model_copy(update={"payload": redacted if isinstance(redacted, dict) else {}})


def build_trace_export(
    view: RunView,
    events: list[RunEvent],
    *,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> TraceExport:
    """Assemble a versioned, bounded, redacted export preserving event order."""

    if not 1 <= max_events <= MAX_EVENTS_LIMIT:
        raise ValueError(f"max_events must be between 1 and {MAX_EVENTS_LIMIT}")
    ordered = sorted(events, key=lambda event: event.sequence)
    bounded = ordered[:max_events]
    return TraceExport(
        run_id=view.run_id,
        trace_id=view.trace_id,
        status=view.status,
        event_count=len(bounded),
        truncated=len(ordered) > len(bounded),
        events=[redact_event(event) for event in bounded],
    )
