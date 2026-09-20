from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from admires_harness.contracts import EventType, JsonValue, RunEvent, RunStatus, utc_now
from admires_harness.observability import RunView
from admires_harness.services.export import (
    MAX_EXPORT_LIMIT,
    REDACTED,
    TRACE_EXPORT_KIND,
    TRACE_EXPORT_SCHEMA_VERSION,
    build_trace_export,
    redact_value,
)


def make_view(run_id: UUID) -> RunView:
    return RunView(
        run_id=run_id,
        trace_id=uuid4(),
        demo="research",
        input="Why use event sourcing?",
        graph_name="research-agent",
        graph_version="1",
        status=RunStatus.SUCCEEDED,
        started_at=utc_now(),
    )


def make_event(run_id: UUID, sequence: int, payload: dict[str, JsonValue]) -> RunEvent:
    return RunEvent(
        run_id=run_id,
        sequence=sequence,
        type=EventType.LOG,
        trace_id=uuid4(),
        payload=payload,
    )


def test_export_preserves_sequence_order_and_schema_version() -> None:
    run_id = uuid4()
    events = [make_event(run_id, sequence, {"n": sequence}) for sequence in (2, 0, 1)]
    document = build_trace_export(make_view(run_id), list(reversed(events)))
    assert document["schema_version"] == TRACE_EXPORT_SCHEMA_VERSION
    assert document["kind"] == TRACE_EXPORT_KIND
    exported = document["events"]
    assert isinstance(exported, list)
    assert [event["sequence"] for event in exported] == [0, 1, 2]  # type: ignore[index]
    assert document["event_count"] == 3
    assert document["total_events"] == 3
    assert document["truncated"] is False


def test_export_is_bounded_and_reports_truncation() -> None:
    run_id = uuid4()
    events = [make_event(run_id, sequence, {}) for sequence in range(5)]
    document = build_trace_export(make_view(run_id), events, limit=2)
    exported = document["events"]
    assert isinstance(exported, list)
    assert len(exported) == 2
    assert [event["sequence"] for event in exported] == [0, 1]  # type: ignore[index]
    assert document["event_count"] == 2
    assert document["total_events"] == 5
    assert document["truncated"] is True


def test_export_rejects_out_of_range_limits() -> None:
    run_id = uuid4()
    events = [make_event(run_id, 0, {})]
    with pytest.raises(ValueError, match="limit"):
        build_trace_export(make_view(run_id), events, limit=0)
    with pytest.raises(ValueError, match="limit"):
        build_trace_export(make_view(run_id), events, limit=MAX_EXPORT_LIMIT + 1)


def test_redaction_covers_nested_payloads_and_secret_patterns() -> None:
    payload = {
        "arguments": {
            "api_key": "super-secret-value",
            "headers": [{"Authorization": "Bearer abcdef1234567890"}],
            "note": "prefix sk-abcdef123456 suffix",
        },
        "input_tokens": 42,
        "text": "ordinary model chunk",
    }
    redacted = redact_value(payload)
    assert isinstance(redacted, dict)
    arguments = redacted["arguments"]
    assert isinstance(arguments, dict)
    assert arguments["api_key"] == REDACTED
    headers = arguments["headers"]
    assert isinstance(headers, list)
    assert headers[0]["Authorization"] == REDACTED  # type: ignore[index]
    note = arguments["note"]
    assert isinstance(note, str)
    assert "sk-abcdef123456" not in note
    assert REDACTED in note
    # Operational fields that merely contain "token" stay intact.
    assert redacted["input_tokens"] == 42
    assert redacted["text"] == "ordinary model chunk"


def test_export_redacts_event_payloads_but_not_envelope() -> None:
    run_id = uuid4()
    events = [make_event(run_id, 0, {"password": "hunter2", "safe": "ok"})]
    document = build_trace_export(make_view(run_id), events)
    exported = document["events"]
    assert isinstance(exported, list)
    first = exported[0]
    assert isinstance(first, dict)
    payload = first["payload"]
    assert isinstance(payload, dict)
    assert payload["password"] == REDACTED
    assert payload["safe"] == "ok"
    assert first["run_id"] == str(run_id)
