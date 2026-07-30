from uuid import uuid4

import pytest

from admires_harness.contracts import EventType, RunEvent, RunStatus
from admires_harness.observability import project_run


def test_project_run_materializes_terminal_metrics():
    run_id, trace_id = uuid4(), uuid4()
    events = [
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
            type=EventType.RUN_COMPLETED,
            payload={
                "status": "succeeded",
                "output": "answer",
                "metrics": {
                    "steps": 2,
                    "model_calls": 1,
                    "tool_calls": 1,
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "cost_usd": 0,
                },
            },
        ),
    ]
    view = project_run(events)
    assert view.status is RunStatus.SUCCEEDED
    assert view.demo == "research"
    assert view.input == "question"
    assert view.output == "answer"
    assert view.metrics.input_tokens == 10
    assert view.ended_at == events[-1].timestamp


def test_project_run_rejects_empty_mixed_and_gapped_streams():
    with pytest.raises(ValueError, match="empty"):
        project_run([])
    trace_id = uuid4()
    first = RunEvent(run_id=uuid4(), sequence=0, type=EventType.RUN_CREATED, trace_id=trace_id)
    mixed = RunEvent(run_id=uuid4(), sequence=1, type=EventType.RUN_STARTED, trace_id=trace_id)
    with pytest.raises(ValueError, match="multiple"):
        project_run([first, mixed])
    gap = mixed.model_copy(update={"run_id": first.run_id, "sequence": 2})
    with pytest.raises(ValueError, match="gap"):
        project_run([first, gap])


def test_project_run_does_not_publish_terminal_checkpoint_before_completion():
    run_id, trace_id = uuid4(), uuid4()
    events = [
        RunEvent(
            run_id=run_id,
            trace_id=trace_id,
            sequence=0,
            type=EventType.RUN_CREATED,
            payload={"graph_name": "research-agent"},
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
            type=EventType.CHECKPOINT_SAVED,
            payload={"status": "succeeded", "revision": 0},
        ),
    ]

    assert project_run(events).status is RunStatus.RUNNING
