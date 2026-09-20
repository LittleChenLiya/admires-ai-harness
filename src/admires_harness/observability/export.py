"""Bounded, redacted trace export for persisted run events.

The export document is a stable, versioned contract. It is read-only: building it
never invokes a model, a tool, or mutates run state. Sensitive fields are redacted
recursively so exported traces can be shared without leaking credentials.
"""

from __future__ import annotations

import re
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from admires_harness.contracts import JsonValue, RunEvent

TRACE_EXPORT_SCHEMA_VERSION: Literal["trace-export/1"] = "trace-export/1"

REDACTED = "[redacted]"

DEFAULT_EXPORT_LIMIT = 1_000
MAX_EXPORT_LIMIT = 5_000

_SENSITIVE_FRAGMENTS = (
    "apikey",
    "accesstoken",
    "authtoken",
    "token",
    "secret",
    "password",
    "passwd",
    "authorization",
    "credential",
    "cookie",
    "privatekey",
    "sessionid",
)

_KEY_NORMALIZER = re.compile(r"[^a-z0-9]")


def _is_sensitive_key(key: str) -> bool:
    normalized = _KEY_NORMALIZER.sub("", key.lower())
    return any(fragment in normalized for fragment in _SENSITIVE_FRAGMENTS)


def redact_value(value: JsonValue) -> JsonValue:
    """Recursively redact sensitive keys in nested mappings and sequences."""

    if isinstance(value, dict):
        return {
            key: REDACTED if _is_sensitive_key(key) else redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    return value


class TraceExportEvent(BaseModel):
    """One exported event: the persisted envelope with a redacted payload."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    event_id: str
    run_id: str
    sequence: int
    type: str
    timestamp: str
    parent_run_id: str | None
    node_id: str | None
    trace_id: str
    span_id: str
    payload: dict[str, JsonValue]


class TraceExport(BaseModel):
    """Versioned, bounded export document for one run's persisted events."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = TRACE_EXPORT_SCHEMA_VERSION
    run_id: str
    status: str
    event_count: int = Field(ge=0)
    total_events: int = Field(ge=0)
    truncated: bool
    events: list[TraceExportEvent]


def build_trace_export(
    run_id: UUID,
    status: str,
    events: list[RunEvent],
    *,
    limit: int = DEFAULT_EXPORT_LIMIT,
) -> TraceExport:
    """Build the export document, preserving sequence order and bounding output."""

    if not 1 <= limit <= MAX_EXPORT_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_EXPORT_LIMIT}")
    ordered = sorted(events, key=lambda event: event.sequence)
    selected = ordered[:limit]
    exported = []
    for event in selected:
        envelope = event.model_dump(mode="json")
        envelope["payload"] = redact_value(event.payload)
        exported.append(TraceExportEvent.model_validate(envelope))
    return TraceExport(
        run_id=str(run_id),
        status=status,
        event_count=len(exported),
        total_events=len(ordered),
        truncated=len(ordered) > len(selected),
        events=exported,
    )
