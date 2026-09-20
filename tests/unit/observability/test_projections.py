from uuid import uuid4

import pytest

from admires_harness.contracts import EventType, JsonValue, RunEvent, RunStatus
from admires_harness.observability import ProjectionIssueCode, project_run


def make_event(
    run_id,
    trace_id,
    sequence: int,
    event_type: EventType,
    payload: dict[str, JsonValue] | None = None,
    *,
    node_id: str | None = None,
    parent_run_id=None,
) -> RunEvent:
    return RunEvent(
        run_id=run_id,
        trace_id=trace_id,
        sequence=sequence,
        type=event_type,
        node_id=node_id,
        parent_run_id=parent_run_id,
        payload=payload or {},
    )


def renumber(events: list[RunEvent]) -> list[RunEvent]:
    return [event.model_copy(update={"sequence": index}) for index, event in enumerate(events)]


def success_events(run_id, trace_id) -> list[RunEvent]:
    return [
        make_event(
            run_id,
            trace_id,
            0,
            EventType.RUN_CREATED,
            {
                "graph_name": "research-agent",
                "graph_version": "1",
                "initial_state": {"input": "question", "demo": "research"},
            },
        ),
        make_event(run_id, trace_id, 1, EventType.RUN_STARTED),
        make_event(run_id, trace_id, 2, EventType.NODE_STARTED, {"step": 1}, node_id="plan"),
        make_event(
            run_id,
            trace_id,
            3,
            EventType.NODE_COMPLETED,
            {"next_node": "answer", "update": {"plan": ["search"]}},
            node_id="plan",
        ),
        make_event(
            run_id,
            trace_id,
            4,
            EventType.CHECKPOINT_SAVED,
            {"revision": 0, "status": "running", "next_node": "answer"},
        ),
        make_event(run_id, trace_id, 5, EventType.NODE_STARTED, {"step": 2}, node_id="answer"),
        make_event(
            run_id,
            trace_id,
            6,
            EventType.NODE_COMPLETED,
            {"next_node": None, "update": {"draft": "answer"}, "output": "answer"},
            node_id="answer",
        ),
        make_event(
            run_id,
            trace_id,
            7,
            EventType.CHECKPOINT_SAVED,
            {"revision": 1, "status": "succeeded", "next_node": None},
        ),
        make_event(
            run_id,
            trace_id,
            8,
            EventType.RUN_COMPLETED,
            {
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


def test_project_run_recovers_nodes_state_and_checkpoint():
    run_id, trace_id = uuid4(), uuid4()
    events = success_events(run_id, trace_id)

    view = project_run(events)

    assert view.status is RunStatus.SUCCEEDED
    assert view.initial_state == {"input": "question", "demo": "research"}
    assert view.state == {
        "input": "question",
        "demo": "research",
        "plan": ["search"],
        "draft": "answer",
    }
    assert view.next_node is None
    assert view.checkpoint_revision == 1
    assert [node.node_id for node in view.nodes] == ["plan", "answer"]
    assert all(node.status == "completed" for node in view.nodes)
    assert view.nodes[0].started_at == events[2].timestamp
    assert view.nodes[0].ended_at == events[3].timestamp
    assert view.issues == ()


def test_project_run_failed_run_recovers_error_and_metrics():
    run_id, trace_id = uuid4(), uuid4()
    events = [
        make_event(run_id, trace_id, 0, EventType.RUN_CREATED, {"graph_name": "coding-agent"}),
        make_event(run_id, trace_id, 1, EventType.RUN_STARTED),
        make_event(run_id, trace_id, 2, EventType.NODE_STARTED, {"step": 1}, node_id="implement"),
        make_event(
            run_id,
            trace_id,
            3,
            EventType.NODE_FAILED,
            {"error": "boom", "error_type": "RuntimeError"},
            node_id="implement",
        ),
        make_event(
            run_id,
            trace_id,
            4,
            EventType.RUN_FAILED,
            {
                "status": "failed",
                "error": "boom",
                "metrics": {"steps": 1, "model_calls": 0, "tool_calls": 0},
            },
        ),
    ]

    view = project_run(events)

    assert view.status is RunStatus.FAILED
    assert view.error == "boom"
    assert view.ended_at == events[-1].timestamp
    assert view.metrics.steps == 1
    assert view.nodes[0].status == "failed"
    assert view.nodes[0].error == "boom"


def test_project_run_cancelled_run_is_terminal():
    run_id, trace_id = uuid4(), uuid4()
    events = [
        make_event(run_id, trace_id, 0, EventType.RUN_CREATED, {"graph_name": "coding-agent"}),
        make_event(run_id, trace_id, 1, EventType.RUN_STARTED),
        make_event(run_id, trace_id, 2, EventType.RUN_STATUS, {"status": "cancelled"}),
    ]

    view = project_run(events)

    assert view.status is RunStatus.CANCELLED
    assert view.status.terminal
    assert view.ended_at == events[-1].timestamp


def test_project_run_duplicate_event_ids_are_idempotent():
    run_id, trace_id = uuid4(), uuid4()
    events = success_events(run_id, trace_id)

    assert project_run(events) == project_run([*events, *events])


def test_project_run_duplicate_sequence_same_content_is_deduped():
    run_id, trace_id = uuid4(), uuid4()
    events = success_events(run_id, trace_id)
    redelivered = events[3].model_copy(update={"event_id": uuid4()})

    assert project_run(events) == project_run([*events, redelivered])


def test_project_run_conflicting_sequence_raises():
    run_id, trace_id = uuid4(), uuid4()
    events = success_events(run_id, trace_id)
    conflict = events[3].model_copy(update={"event_id": uuid4(), "payload": {"update": {}}})

    with pytest.raises(ValueError, match="conflicting"):
        project_run([*events, conflict])


def test_project_run_out_of_order_input_matches_ordered_projection():
    run_id, trace_id = uuid4(), uuid4()
    events = success_events(run_id, trace_id)

    assert project_run(list(reversed(events))) == project_run(events)


def test_project_run_unknown_event_type_is_reported_and_skipped():
    run_id, trace_id = uuid4(), uuid4()
    events = success_events(run_id, trace_id)
    events[4] = events[4].model_copy(update={"type": "checkpoint.v2"})

    view = project_run(events)

    assert view.status is RunStatus.SUCCEEDED
    assert view.checkpoint_revision == 1
    unknown = [
        issue for issue in view.issues if issue.code is ProjectionIssueCode.UNKNOWN_EVENT_TYPE
    ]
    assert len(unknown) == 1
    assert unknown[0].sequence == 4


def test_project_run_missing_run_created_is_reported():
    run_id, trace_id = uuid4(), uuid4()
    events = renumber(success_events(run_id, trace_id)[1:])

    view = project_run(events)

    assert view.status is RunStatus.SUCCEEDED
    assert view.graph_name == "unknown"
    assert view.initial_state == {}
    assert view.input == ""
    missing = [
        issue for issue in view.issues if issue.code is ProjectionIssueCode.MISSING_RUN_CREATED
    ]
    assert len(missing) == 1
    assert missing[0].sequence == 0


def test_project_run_tool_failure_is_recovered():
    run_id, trace_id = uuid4(), uuid4()
    events = [
        make_event(run_id, trace_id, 0, EventType.RUN_CREATED, {"graph_name": "coding-agent"}),
        make_event(run_id, trace_id, 1, EventType.RUN_STARTED),
        make_event(run_id, trace_id, 2, EventType.NODE_STARTED, {"step": 1}, node_id="implement"),
        make_event(
            run_id,
            trace_id,
            3,
            EventType.TOOL_REQUESTED,
            {"tool": "workspace.write", "arguments": {"path": "a.py"}},
            node_id="implement",
        ),
        make_event(
            run_id,
            trace_id,
            4,
            EventType.TOOL_STARTED,
            {"tool": "workspace.write"},
            node_id="implement",
        ),
        make_event(
            run_id,
            trace_id,
            5,
            EventType.TOOL_FAILED,
            {"tool": "workspace.write", "error": "denied", "error_code": "policy_denied"},
            node_id="implement",
        ),
        make_event(
            run_id,
            trace_id,
            6,
            EventType.RUN_FAILED,
            {"status": "failed", "error": "denied", "metrics": {"tool_calls": 1}},
        ),
    ]

    view = project_run(events)

    assert view.status is RunStatus.FAILED
    assert len(view.tools) == 1
    assert view.tools[0].ok is False
    assert view.tools[0].tool == "workspace.write"
    assert view.tools[0].error_code == "policy_denied"
    assert view.error == "denied"


def test_project_run_approval_lifecycle_is_recovered():
    run_id, trace_id = uuid4(), uuid4()
    request_id = str(uuid4())
    request = {"request_id": request_id, "tool_name": "workspace.write", "checkpoint_revision": 0}
    events = [
        make_event(run_id, trace_id, 0, EventType.RUN_CREATED, {"graph_name": "coding-agent"}),
        make_event(run_id, trace_id, 1, EventType.RUN_STARTED),
        make_event(
            run_id,
            trace_id,
            2,
            EventType.APPROVAL_REQUESTED,
            {"request": request, "arguments": {"path": "a.py"}},
            node_id="implement",
        ),
        make_event(run_id, trace_id, 3, EventType.RUN_STATUS, {"status": "waiting_approval"}),
    ]

    waiting = project_run(events)
    assert waiting.status is RunStatus.WAITING_APPROVAL
    assert not waiting.status.terminal
    assert waiting.ended_at is None
    assert len(waiting.approvals) == 1
    assert waiting.approvals[0].status == "requested"
    assert waiting.approvals[0].tool_name == "workspace.write"

    resolved = [
        *events,
        make_event(
            run_id,
            trace_id,
            4,
            EventType.APPROVAL_RESOLVED,
            {"decision": {"request_id": request_id, "approved": True, "decided_by": "human"}},
        ),
        make_event(run_id, trace_id, 5, EventType.RUN_STATUS, {"status": "running"}),
        make_event(run_id, trace_id, 6, EventType.APPROVAL_CONSUMED, {"request": request}),
    ]
    view = project_run(resolved)
    assert view.status is RunStatus.RUNNING
    assert view.approvals[0].status == "consumed"
    assert view.approvals[0].decided_by == "human"


def test_project_run_sub_run_events_do_not_corrupt_parent_view():
    run_id, trace_id = uuid4(), uuid4()
    parent_id = uuid4()
    child_id = uuid4()
    events = [
        make_event(
            run_id,
            trace_id,
            0,
            EventType.RUN_CREATED,
            {"graph_name": "collaboration-agent", "initial_state": {"input": "task"}},
            parent_run_id=parent_id,
        ),
        make_event(run_id, trace_id, 1, EventType.RUN_STARTED),
        make_event(
            run_id,
            trace_id,
            2,
            EventType.AGENT_DELEGATED,
            {"child_run_id": str(child_id), "graph_name": "research-agent", "label": "analyst"},
            node_id="delegate",
        ),
        make_event(
            run_id,
            trace_id,
            3,
            EventType.AGENT_JOINED,
            {"child_run_id": str(child_id), "status": "succeeded", "label": "analyst"},
            node_id="delegate",
        ),
        make_event(
            run_id,
            trace_id,
            4,
            EventType.RUN_COMPLETED,
            {"status": "succeeded", "output": "done", "metrics": {"steps": 2}},
        ),
    ]

    view = project_run(events)

    assert view.status is RunStatus.SUCCEEDED
    assert view.parent_run_id == parent_id
    assert view.issues == ()


def test_project_run_malformed_payloads_are_reported_not_fatal():
    run_id, trace_id = uuid4(), uuid4()
    events = [
        make_event(run_id, trace_id, 0, EventType.RUN_CREATED, {"graph_name": "coding-agent"}),
        make_event(run_id, trace_id, 1, EventType.RUN_STARTED),
        make_event(run_id, trace_id, 2, EventType.RUN_STATUS, {"status": "not-a-status"}),
        make_event(run_id, trace_id, 3, EventType.APPROVAL_REQUESTED, {"request": "nope"}),
        make_event(
            run_id,
            trace_id,
            4,
            EventType.RUN_COMPLETED,
            {"status": "succeeded", "output": "ok", "metrics": "broken"},
        ),
    ]

    view = project_run(events)

    assert view.status is RunStatus.SUCCEEDED
    assert view.output == "ok"
    invalid = [issue for issue in view.issues if issue.code is ProjectionIssueCode.INVALID_PAYLOAD]
    assert [issue.sequence for issue in invalid] == [2, 3, 4]


def test_project_run_repeated_mixed_application_is_stable():
    run_id, trace_id = uuid4(), uuid4()
    events = success_events(run_id, trace_id)
    base = project_run(events)

    reapplied = project_run([*events, *reversed(events)])

    assert reapplied == base
    assert reapplied.issues == base.issues
