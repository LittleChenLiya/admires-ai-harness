"""Bounded, redacted trace export over the persisted event stream."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field

from admires_harness.contracts import JsonValue, RunEvent, RunStatus, StrictModel, utc_now
from admires_harness.observability.projections import project_run

TRACE_EXPORT_SCHEMA_VERSION: Literal["1"] = "1"
REDACTED = "[REDACTED]"
DEFAULT_EXPORT_LIMIT = 1_000
MAX_EXPORT_LIMIT = 10_000

_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "proxy_authorization",
    "access_token",
    "refresh_token",
    "id_token",
    "client_secret",
    "private_key",
    "cookie",
    "set_cookie",
    "x_api_key",
}
_SENSITIVE_SEGMENTS = {
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "credentials",
    "cookie",
    "apikey",
}
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def is_sensitive_key(key: str) -> bool:
    """Match credential-like keys without flagging counters such as ``input_tokens``."""

    normalized = _NON_ALNUM.sub("_", _CAMEL_BOUNDARY.sub("_", key).lower()).strip("_")
    if normalized in _SENSITIVE_KEYS:
        return True
    return any(segment in _SENSITIVE_SEGMENTS for segment in normalized.split("_"))


def redact_value(value: JsonValue, *, key: str | None = None) -> JsonValue:
    """Recursively redact sensitive fields in arbitrarily nested payloads."""

    if key is not None and is_sensitive_key(key):
        return REDACTED
    if isinstance(value, dict):
        return {item_key: redact_value(item, key=item_key) for item_key, item in value.items()}
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    return value


class TraceExport(StrictModel):
    """Stable, versioned export envelope for one run's persisted events."""

    schema_version: Literal["1"] = TRACE_EXPORT_SCHEMA_VERSION
    run_id: UUID
    trace_id: UUID
    status: RunStatus
    exported_at: datetime = Field(default_factory=utc_now)
    total_events: int = Field(ge=0)
    truncated: bool
    events: list[RunEvent]


def build_trace_export(events: list[RunEvent], *, limit: int = DEFAULT_EXPORT_LIMIT) -> TraceExport:
    """Assemble a redacted, sequence-ordered export without touching runtime state."""

    if not events:
        raise ValueError("cannot export an empty event stream")
    if not 1 <= limit <= MAX_EXPORT_LIMIT:
        raise ValueError(f"export limit must be between 1 and {MAX_EXPORT_LIMIT}")
    ordered = sorted(events, key=lambda event: event.sequence)
    view = project_run(ordered)
    bounded = ordered[:limit]
    return TraceExport(
        run_id=view.run_id,
        trace_id=view.trace_id,
        status=view.status,
        total_events=len(ordered),
        truncated=len(bounded) < len(ordered),
        events=[
            event.model_copy(update={"payload": redact_value(event.payload)}) for event in bounded
        ],
    )
