from uuid import uuid4

import pytest

from admires_harness.contracts import EventType, RunEvent
from admires_harness.observability import (
    REDACTED,
    build_trace_export,
    project_run,
    redact_value,
)


def make_event(sequence: int, payload: dict | None = None) -> RunEvent:
    return RunEvent(
        run_id=RUN_ID,
        trace_id=TRACE_ID,
        sequence=sequence,
        type=EventType.LOG,
        payload=payload or {},
    )


RUN_ID = uuid4()
TRACE_ID = uuid4()


def test_redact_value_handles_nested_mappings_and_lists() -> None:
    value = {
        "password": "hunter2",
        "nested": {"X-Api-Key": "abc", "safe": 1},
        "items": [{"session_id": "s3"}, "plain"],
    }
    redacted = redact_value(value)
    assert redacted == {
        "password": REDACTED,
        "nested": {"X-Api-Key": REDACTED, "safe": 1},
        "items": [{"session_id": REDACTED}, "plain"],
    }


def test_build_trace_export_orders_bounds_and_redacts() -> None:
    events = [
        RunEvent(
            run_id=RUN_ID,
            trace_id=TRACE_ID,
            sequence=0,
            type=EventType.RUN_CREATED,
            payload={
                "graph_name": "research-agent",
                "graph_version": "1",
                "initial_state": {"input": "q", "demo": "research"},
            },
        ),
        make_event(2, {"token": "t"}),
        make_event(1),
    ]
    view = project_run([events[0], events[2], events[1]])
    export = build_trace_export(view, events, max_events=2)
    assert export.schema_version == "1"
    assert export.event_count == 2
    assert export.truncated is True
    assert [event.sequence for event in export.events] == [0, 1]
    full = build_trace_export(view, events, max_events=10)
    assert full.truncated is False
    assert full.events[-1].payload["token"] == REDACTED


def test_build_trace_export_rejects_invalid_bounds() -> None:
    events = [
        RunEvent(
            run_id=RUN_ID,
            trace_id=TRACE_ID,
            sequence=0,
            type=EventType.RUN_CREATED,
            payload={
                "graph_name": "research-agent",
                "graph_version": "1",
                "initial_state": {"input": "q", "demo": "research"},
            },
        )
    ]
    view = project_run(events)
    with pytest.raises(ValueError, match="max_events"):
        build_trace_export(view, events, max_events=0)
