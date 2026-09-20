"""Unit tests for the bounded, redacted trace export builder."""

from uuid import uuid4

import pytest

from admires_harness.contracts import EventType, RunEvent
from admires_harness.observability.export import (
    REDACTED,
    TRACE_EXPORT_SCHEMA_VERSION,
    build_trace_export,
    redact_value,
)


def _event(sequence: int, run_id, trace_id, payload=None) -> RunEvent:
    return RunEvent(
        run_id=run_id,
        trace_id=trace_id,
        sequence=sequence,
        type=EventType.RUN_STATUS,
        payload=payload or {},
    )


def test_export_preserves_sequence_order_and_schema_version() -> None:
    run_id, trace_id = uuid4(), uuid4()
    events = [_event(sequence, run_id, trace_id) for sequence in range(3)]
    document = build_trace_export(run_id, "succeeded", list(reversed(events)))
    assert document.schema_version == TRACE_EXPORT_SCHEMA_VERSION
    assert document.run_id == str(run_id)
    assert document.status == "succeeded"
    assert [event.sequence for event in document.events] == [0, 1, 2]
    assert document.event_count == 3
    assert document.total_events == 3
    assert document.truncated is False


def test_export_is_bounded_and_marks_truncation() -> None:
    run_id, trace_id = uuid4(), uuid4()
    events = [_event(sequence, run_id, trace_id) for sequence in range(5)]
    document = build_trace_export(run_id, "running", events, limit=2)
    assert [event.sequence for event in document.events] == [0, 1]
    assert document.event_count == 2
    assert document.total_events == 5
    assert document.truncated is True
    with pytest.raises(ValueError, match="limit"):
        build_trace_export(run_id, "running", events, limit=0)


def test_redaction_replaces_sensitive_keys_at_any_depth() -> None:
    payload = {
        "api_key": "sk-live-secret",
        "nested": {
            "Authorization": "Bearer token",
            "items": [{"clientSecret": "abc"}, {"note": "ok"}],
        },
        "safe": "visible",
    }
    redacted = redact_value(payload)
    assert redacted == {
        "api_key": REDACTED,
        "nested": {
            "Authorization": REDACTED,
            "items": [{"clientSecret": REDACTED}, {"note": "ok"}],
        },
        "safe": "visible",
    }


def test_export_redacts_event_payloads_without_mutating_source() -> None:
    run_id, trace_id = uuid4(), uuid4()
    event = _event(
        0,
        run_id,
        trace_id,
        payload={"config": {"password": "hunter2"}, "detail": "kept"},
    )
    document = build_trace_export(run_id, "failed", [event])
    exported = document.events[0].payload
    assert exported["config"] == {"password": REDACTED}
    assert exported["detail"] == "kept"
    assert event.payload["config"] == {"password": "hunter2"}


def test_exported_events_carry_stable_envelope_fields() -> None:
    run_id, trace_id = uuid4(), uuid4()
    document = build_trace_export(run_id, "succeeded", [_event(0, run_id, trace_id)])
    exported = document.events[0].model_dump()
    assert exported["schema_version"] == "1"
    assert exported["run_id"] == str(run_id)
    assert exported["trace_id"] == str(trace_id)
    assert set(exported) == {
        "schema_version",
        "event_id",
        "run_id",
        "sequence",
        "type",
        "timestamp",
        "parent_run_id",
        "node_id",
        "trace_id",
        "span_id",
        "payload",
    }
