from uuid import uuid4

import pytest

from admires_harness.contracts import EventType, RunEvent, RunStatus
from admires_harness.observability import (
    TRACE_EXPORT_SCHEMA_VERSION,
    build_trace_export,
    is_sensitive_key,
    redact_value,
)


def make_events(*, terminal: EventType = EventType.RUN_COMPLETED) -> list[RunEvent]:
    run_id, trace_id = uuid4(), uuid4()
    return [
        RunEvent(
            run_id=run_id,
            trace_id=trace_id,
            sequence=0,
            type=EventType.RUN_CREATED,
            payload={
                "graph_name": "research-agent",
                "graph_version": "1",
                "initial_state": {"input": "question", "demo": "research"},
            },
        ),
        RunEvent(
            run_id=run_id,
            trace_id=trace_id,
            sequence=1,
            type=EventType.RUN_STARTED,
        ),
        RunEvent(
            run_id=run_id,
            trace_id=trace_id,
            sequence=2,
            type=terminal,
            payload={
                "status": "succeeded" if terminal is EventType.RUN_COMPLETED else "failed",
                "metrics": {
                    "steps": 1,
                    "model_calls": 1,
                    "tool_calls": 0,
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "cost_usd": 0,
                },
            },
        ),
    ]


def test_export_preserves_sequence_and_stamps_schema_version() -> None:
    events = make_events()
    export = build_trace_export(list(reversed(events)))
    assert export.schema_version == TRACE_EXPORT_SCHEMA_VERSION == "1"
    assert [event.sequence for event in export.events] == [0, 1, 2]
    assert export.total_events == 3
    assert export.truncated is False
    assert export.status is RunStatus.SUCCEEDED
    assert export.run_id == events[0].run_id
    assert export.trace_id == events[0].trace_id


def test_export_is_bounded_and_marks_truncation() -> None:
    export = build_trace_export(make_events(), limit=2)
    assert [event.sequence for event in export.events] == [0, 1]
    assert export.total_events == 3
    assert export.truncated is True
    with pytest.raises(ValueError, match="limit"):
        build_trace_export(make_events(), limit=0)


def test_export_redacts_sensitive_keys_recursively() -> None:
    payload = {
        "config": {
            "api_key": "sk-live-secret",
            "nested": [{"authorization": "Bearer token", "note": "ok"}],
            "clientSecret": "shh",
        },
        "headers": [{"name": "Cookie", "value": "session=abc"}],
        "input_tokens": 12,
        "outputTokens": 3,
        "safe": "visible",
    }
    redacted = redact_value(payload)
    assert isinstance(redacted, dict)
    config = redacted["config"]
    assert isinstance(config, dict)
    assert config["api_key"] == "[REDACTED]"
    assert config["clientSecret"] == "[REDACTED]"
    nested = config["nested"]
    assert isinstance(nested, list)
    first = nested[0]
    assert isinstance(first, dict)
    assert first["authorization"] == "[REDACTED]"
    assert first["note"] == "ok"
    headers = redacted["headers"]
    assert isinstance(headers, list)
    header = headers[0]
    assert isinstance(header, dict)
    assert header["value"] == "session=abc"
    assert redacted["input_tokens"] == 12
    assert redacted["outputTokens"] == 3
    assert redacted["safe"] == "visible"


def test_export_redacts_event_payloads_but_keeps_envelope() -> None:
    events = make_events()
    events[1] = events[1].model_copy(
        update={"payload": {"request": {"apiKey": "sk-test", "prompt": "hi"}}}
    )
    export = build_trace_export(events)
    request = export.events[1].payload["request"]
    assert isinstance(request, dict)
    assert request["apiKey"] == "[REDACTED]"
    assert request["prompt"] == "hi"
    assert export.events[1].event_id == events[1].event_id


def test_sensitive_key_matching_avoids_metric_counters() -> None:
    for key in ("api_key", "apiKey", "Authorization", "client_secret", "accessToken"):
        assert is_sensitive_key(key)
    for key in ("input_tokens", "outputTokens", "prompt", "monkey"):
        assert not is_sensitive_key(key)


def test_export_rejects_empty_event_stream() -> None:
    with pytest.raises(ValueError, match="empty"):
        build_trace_export([])


def test_export_supports_failed_runs() -> None:
    export = build_trace_export(make_events(terminal=EventType.RUN_FAILED))
    assert export.status is RunStatus.FAILED
    assert export.total_events == 3
    assert export.events[-1].type is EventType.RUN_FAILED
