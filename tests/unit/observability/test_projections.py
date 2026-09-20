from datetime import UTC, datetime
from uuid import uuid4

import pytest

from admires_harness.contracts import (
    ApprovalDecision,
    ApprovalRequest,
    EventType,
    RiskLevel,
    RunEvent,
    RunStatus,
)
from admires_harness.observability import (
    ApprovalStatus,
    DiagnosticCode,
    NodeStatus,
    ToolStatus,
    project_run,
    project_run_checked,
)


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


def _event(run_id, trace_id, sequence, event_type, payload=None, **kwargs):
    return RunEvent(
        run_id=run_id,
        trace_id=trace_id,
        sequence=sequence,
        type=event_type,
        payload=payload or {},
        **kwargs,
    )


def _success_events(run_id, trace_id):
    return [
        _event(
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
        _event(run_id, trace_id, 1, EventType.RUN_STARTED),
        _event(run_id, trace_id, 2, EventType.NODE_STARTED, {"step": 1}, node_id="plan"),
        _event(
            run_id,
            trace_id,
            3,
            EventType.NODE_COMPLETED,
            {
                "next_node": None,
                "update": {"notes": "done"},
                "output": None,
                "action_key": None,
                "usage": {"input_tokens": 5, "output_tokens": 2, "cost_usd": 0.0},
                "model_calls": 1,
                "tool_calls": 0,
                "metadata": {},
            },
            node_id="plan",
        ),
        _event(
            run_id,
            trace_id,
            4,
            EventType.RUN_COMPLETED,
            {
                "status": "succeeded",
                "output": "answer",
                "error": None,
                "metrics": {
                    "steps": 1,
                    "model_calls": 1,
                    "tool_calls": 0,
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "cost_usd": 0.0,
                },
            },
        ),
    ]


def _codes(projection):
    return [diagnostic.code for diagnostic in projection.diagnostics]


def _codes(projection):
    return [diagnostic.code for diagnostic in projection.diagnostics]


def test_checked_projection_recovers_complete_successful_run():
    run_id, trace_id = uuid4(), uuid4()
    events = _success_events(run_id, trace_id)

    projection = project_run_checked(events)

    assert projection.diagnostics == ()
    assert projection.applied == len(events)
    assert projection.dropped == 0
    view = projection.view
    assert view.status is RunStatus.SUCCEEDED
    assert view.state == {"input": "question", "demo": "research", "notes": "done"}
    assert view.input == "question"
    assert view.demo == "research"
    assert view.output == "answer"
    assert view.ended_at == events[-1].timestamp
    assert view.metrics.steps == 1
    assert view.metrics.input_tokens == 5
    assert len(view.nodes) == 1
    node = view.nodes[0]
    assert node.node_id == "plan"
    assert node.status is NodeStatus.COMPLETED
    assert node.started_at == events[2].timestamp
    assert node.ended_at == events[3].timestamp


def test_checked_projection_recovers_failed_run():
    run_id, trace_id = uuid4(), uuid4()
    events = [
        _event(run_id, trace_id, 0, EventType.RUN_CREATED, {"graph_name": "coding-agent"}),
        _event(run_id, trace_id, 1, EventType.RUN_STARTED),
        _event(run_id, trace_id, 2, EventType.NODE_STARTED, {"step": 1}, node_id="act"),
        _event(
            run_id,
            trace_id,
            3,
            EventType.NODE_FAILED,
            {"error": "boom", "error_type": "RuntimeError"},
            node_id="act",
        ),
        _event(
            run_id,
            trace_id,
            4,
            EventType.RUN_FAILED,
            {"status": "failed", "error": "boom", "output": None, "metrics": {}},
        ),
    ]

    view = project_run_checked(events).view

    assert view.status is RunStatus.FAILED
    assert view.error == "boom"
    assert view.ended_at == events[-1].timestamp
    assert view.nodes[0].status is NodeStatus.FAILED
    assert view.nodes[0].error == "boom"


def test_checked_projection_recovers_cancelled_run():
    run_id, trace_id = uuid4(), uuid4()
    events = [
        _event(run_id, trace_id, 0, EventType.RUN_CREATED, {"graph_name": "coding-agent"}),
        _event(run_id, trace_id, 1, EventType.RUN_STARTED),
        _event(run_id, trace_id, 2, EventType.RUN_STATUS, {"status": "cancelled"}),
    ]

    view = project_run_checked(events).view

    assert view.status is RunStatus.CANCELLED
    assert view.status.terminal
    assert view.ended_at == events[-1].timestamp
    assert view.ended_at == events[-1].timestamp


def test_duplicate_events_are_dropped_idempotently():
    run_id, trace_id = uuid4(), uuid4()
    base = _success_events(run_id, trace_id)
    duplicated = [event for event in base for _ in range(2)]

    projection = project_run_checked(duplicated)

    assert projection.view == project_run_checked(base).view
    assert _codes(projection) == [DiagnosticCode.DUPLICATE_EVENT] * len(base)
    assert projection.applied == len(base)
    assert projection.dropped == len(base)
    assert project_run(duplicated) == project_run(base)


def test_reapplication_of_the_whole_stream_is_stable():
    run_id, trace_id = uuid4(), uuid4()
    base = _success_events(run_id, trace_id)
    first = project_run_checked(base)

    assert project_run_checked(base + base).view == first.view
    assert project_run_checked(list(reversed(base))).view == first.view
    assert project_run_checked(base).view == first.view
    assert project_run(base + base) == project_run(base)


def test_duplicate_sequence_conflict_is_deterministic():
    run_id, trace_id = uuid4(), uuid4()
    created = _event(run_id, trace_id, 0, EventType.RUN_CREATED, {"graph_name": "g"})
    pause = _event(run_id, trace_id, 1, EventType.RUN_STATUS, {"status": "paused"})
    waiting = _event(run_id, trace_id, 1, EventType.RUN_STATUS, {"status": "waiting_approval"})

    forward = project_run_checked([created, pause, waiting])
    backward = project_run_checked([created, waiting, pause])

    assert forward.view == backward.view
    assert _codes(forward) == [DiagnosticCode.SEQUENCE_CONFLICT]
    assert forward.dropped == 1
    winner = min([pause, waiting], key=lambda event: (event.timestamp, str(event.event_id)))
    assert forward.view.status is RunStatus(winner.payload["status"])


def test_sequence_gap_is_reported_not_silently_filled():
    run_id, trace_id = uuid4(), uuid4()
    events = [
        _event(run_id, trace_id, 0, EventType.RUN_CREATED, {"graph_name": "g"}),
        _event(run_id, trace_id, 1, EventType.RUN_STARTED),
        _event(run_id, trace_id, 3, EventType.RUN_COMPLETED, {"status": "succeeded"}),
    ]

    with pytest.raises(ValueError, match="gap"):
        project_run(events)
    projection = project_run_checked(events)
    assert _codes(projection) == [DiagnosticCode.SEQUENCE_GAP]
    assert "2" in projection.diagnostics[0].message
    assert projection.view.status is RunStatus.SUCCEEDED


def test_out_of_order_delivery_is_reported_and_reordered():
    run_id, trace_id = uuid4(), uuid4()
    events = _success_events(run_id, trace_id)
    shuffled = [events[2], events[0], events[4], events[1], events[3]]

    projection = project_run_checked(shuffled)

    assert _codes(projection) == [DiagnosticCode.OUT_OF_ORDER]
    assert projection.view == project_run(events)
    assert projection.view == project_run(events)


def test_unknown_event_type_is_skipped_and_reported():
    run_id, trace_id = uuid4(), uuid4()
    unknown = RunEvent.model_construct(
        run_id=run_id,
        trace_id=trace_id,
        sequence=2,
        type="run.future",
        payload={"output": "forged"},
    )
    events = [
        _event(run_id, trace_id, 0, EventType.RUN_CREATED, {"graph_name": "g"}),
        _event(run_id, trace_id, 1, EventType.RUN_STARTED),
        unknown,
        _event(run_id, trace_id, 3, EventType.RUN_COMPLETED, {"status": "succeeded"}),
    ]

    projection = project_run_checked(events)

    assert _codes(projection) == [DiagnosticCode.UNKNOWN_EVENT_TYPE]
    assert projection.dropped == 1
    assert projection.view.status is RunStatus.SUCCEEDED
    assert projection.view.output is None
    assert project_run(events).status is RunStatus.SUCCEEDED


def test_missing_run_created_is_reported():
    run_id, trace_id = uuid4(), uuid4()
    events = [
        _event(run_id, trace_id, 0, EventType.RUN_STARTED),
        _event(run_id, trace_id, 1, EventType.RUN_COMPLETED, {"status": "succeeded"}),
    ]

    with pytest.raises(ValueError, match=r"run\.created"):
        project_run(events)
    projection = project_run_checked(events)
    assert _codes(projection) == [DiagnosticCode.MISSING_RUN_CREATED]
    view = projection.view
    assert view.status is RunStatus.SUCCEEDED
    assert view.graph_name == "unknown"
    assert view.graph_version == "unknown"
    assert view.run_id == run_id
    assert view.started_at == events[0].timestamp


def test_tool_failure_is_recovered():
    run_id, trace_id = uuid4(), uuid4()
    events = [
        _event(run_id, trace_id, 0, EventType.RUN_CREATED, {"graph_name": "coding-agent"}),
        _event(run_id, trace_id, 1, EventType.RUN_STARTED),
        _event(run_id, trace_id, 2, EventType.NODE_STARTED, {"step": 1}, node_id="act"),
        _event(
            run_id,
            trace_id,
            3,
            EventType.TOOL_REQUESTED,
            {"tool": "workspace.write", "arguments": {"path": "a.py"}},
            node_id="act",
        ),
        _event(run_id, trace_id, 4, EventType.TOOL_STARTED, {"tool": "workspace.write"}),
        _event(
            run_id,
            trace_id,
            5,
            EventType.TOOL_FAILED,
            {"tool": "workspace.write", "error": "denied", "error_code": "approval_expired"},
        ),
        _event(
            run_id,
            trace_id,
            6,
            EventType.RUN_FAILED,
            {"status": "failed", "error": "denied", "metrics": {}},
        ),
    ]

    view = project_run_checked(events).view

    assert view.status is RunStatus.FAILED
    assert len(view.tools) == 1
    tool = view.tools[0]
    assert tool.tool == "workspace.write"
    assert tool.status is ToolStatus.FAILED
    assert tool.error == "denied"
    assert tool.error_code == "approval_expired"
    assert tool.requested_at == events[3].timestamp
    assert tool.ended_at == events[5].timestamp
    assert tool.ended_at == events[5].timestamp


def _approval_events(run_id, trace_id):
    request = ApprovalRequest(
        run_id=run_id,
        checkpoint_revision=0,
        tool_name="workspace.write",
        arguments_digest="digest",
        reason="write needs approval",
        risk=RiskLevel.HIGH,
        expires_at=datetime(2030, 1, 1, tzinfo=UTC),
    )
    events = [
        _event(run_id, trace_id, 0, EventType.RUN_CREATED, {"graph_name": "coding-agent"}),
        _event(run_id, trace_id, 1, EventType.RUN_STARTED),
        _event(run_id, trace_id, 2, EventType.NODE_STARTED, {"step": 1}, node_id="implement"),
        _event(
            run_id,
            trace_id,
            3,
            EventType.APPROVAL_REQUESTED,
            {"request": request.model_dump(mode="json"), "arguments": {"path": "a.py"}},
            node_id="implement",
        ),
        _event(
            run_id,
            trace_id,
            4,
            EventType.NODE_COMPLETED,
            {
                "next_node": "implement",
                "update": {"approval_request_id": str(request.request_id)},
                "output": None,
                "usage": {"input_tokens": 7, "output_tokens": 3, "cost_usd": 0.0},
                "model_calls": 1,
                "tool_calls": 0,
                "metadata": {},
            },
            node_id="implement",
        ),
        _event(run_id, trace_id, 5, EventType.RUN_STATUS, {"status": "waiting_approval"}),
    ]
    return request, events


def test_approval_wait_recovers_pending_request_and_accumulated_metrics():
    run_id, trace_id = uuid4(), uuid4()
    request, events = _approval_events(run_id, trace_id)

    projection = project_run_checked(events)

    assert projection.diagnostics == ()
    view = projection.view
    assert view.status is RunStatus.WAITING_APPROVAL
    assert view.ended_at is None
    assert len(view.approvals) == 1
    approval = view.approvals[0]
    assert approval.request_id == str(request.request_id)
    assert approval.tool_name == "workspace.write"
    assert approval.risk == "high"
    assert approval.status is ApprovalStatus.PENDING
    # No terminal metrics payload exists yet; consumption accumulates per node.
    assert view.metrics.steps == 1
    assert view.metrics.model_calls == 1
    assert view.metrics.input_tokens == 7


def test_approval_resolution_and_consumption_are_recovered():
    run_id, trace_id = uuid4(), uuid4()
    request, events = _approval_events(run_id, trace_id)
    decision = ApprovalDecision(request_id=request.request_id, approved=True, decided_by="operator")
    events += [
        _event(
            run_id,
            trace_id,
            6,
            EventType.APPROVAL_RESOLVED,
            {"decision": decision.model_dump(mode="json")},
        ),
        _event(
            run_id,
            trace_id,
            7,
            EventType.APPROVAL_CONSUMED,
            {"request": request.model_dump(mode="json")},
        ),
    ]

    view = project_run_checked(events).view

    assert view.approvals[0].status is ApprovalStatus.CONSUMED
    resolved_only = project_run_checked(events[:7]).view
    assert resolved_only.approvals[0].status is ApprovalStatus.APPROVED
    assert resolved_only.approvals[0].status is ApprovalStatus.APPROVED


def test_child_run_events_are_tracked_without_polluting_run_metrics():
    run_id, trace_id, child_id = uuid4(), uuid4(), uuid4()
    events = [
        _event(run_id, trace_id, 0, EventType.RUN_CREATED, {"graph_name": "orchestrator"}),
        _event(run_id, trace_id, 1, EventType.RUN_STARTED),
        _event(run_id, trace_id, 2, EventType.NODE_STARTED, {"step": 1}, node_id="delegate"),
        _event(
            run_id,
            trace_id,
            3,
            EventType.AGENT_DELEGATED,
            {
                "child_run_id": str(child_id),
                "graph_name": "child-agent",
                "label": "research",
                "capabilities": [],
            },
            node_id="delegate",
        ),
        _event(
            run_id,
            trace_id,
            4,
            EventType.AGENT_JOINED,
            {
                "child_run_id": str(child_id),
                "status": "succeeded",
                "label": "research",
                "metrics": {
                    "steps": 2,
                    "model_calls": 2,
                    "tool_calls": 1,
                    "input_tokens": 20,
                    "output_tokens": 8,
                    "cost_usd": 0.0,
                },
            },
            node_id="delegate",
        ),
        _event(
            run_id,
            trace_id,
            5,
            EventType.RUN_COMPLETED,
            {
                "status": "succeeded",
                "output": "done",
                "metrics": {
                    "steps": 3,
                    "model_calls": 2,
                    "tool_calls": 1,
                    "input_tokens": 20,
                    "output_tokens": 8,
                    "cost_usd": 0.0,
                },
            },
        ),
    ]

    view = project_run_checked(events).view

    assert len(view.child_runs) == 1
    child = view.child_runs[0]
    assert child.run_id == str(child_id)
    assert child.graph_name == "child-agent"
    assert child.label == "research"
    assert child.status is RunStatus.SUCCEEDED
    assert child.metrics is not None
    assert child.metrics.steps == 2
    # The terminal payload is authoritative; child metrics never overwrite it.
    assert view.metrics.steps == 3


def test_foreign_run_events_are_dropped_and_reported():
    run_id, trace_id = uuid4(), uuid4()
    base = _success_events(run_id, trace_id)
    foreign = _event(
        uuid4(),
        trace_id,
        5,
        EventType.RUN_STATUS,
        {"status": "failed"},
        parent_run_id=run_id,
    )

    projection = project_run_checked([*base, foreign])

    assert _codes(projection) == [DiagnosticCode.MIXED_RUN_IDS]
    assert projection.dropped == 1
    assert projection.view == project_run_checked(base).view
    with pytest.raises(ValueError, match="multiple"):
        project_run([*base, foreign])


def test_malformed_payloads_are_reported_not_fatal():
    run_id, trace_id = uuid4(), uuid4()
    events = [
        _event(
            run_id,
            trace_id,
            0,
            EventType.RUN_CREATED,
            {"graph_name": "g", "initial_state": "corrupt"},
        ),
        _event(run_id, trace_id, 1, EventType.RUN_STARTED),
        _event(
            run_id,
            trace_id,
            2,
            EventType.NODE_COMPLETED,
            {"update": "corrupt", "usage": {"input_tokens": 1}},
            node_id="n",
        ),
        _event(run_id, trace_id, 3, EventType.APPROVAL_REQUESTED, {"request": "corrupt"}),
        _event(run_id, trace_id, 4, EventType.RUN_STATUS, {"status": "not-a-status"}),
        _event(
            run_id,
            trace_id,
            5,
            EventType.RUN_COMPLETED,
            {"status": "succeeded", "metrics": "corrupt", "output": "ok"},
        ),
    ]

    projection = project_run_checked(events)

    assert _codes(projection) == [DiagnosticCode.MALFORMED_PAYLOAD] * 5
    view = projection.view
    assert view.status is RunStatus.SUCCEEDED
    assert view.state == {}
    assert view.approvals == ()
    assert view.metrics.steps == 1
    assert view.metrics.input_tokens == 1
    assert view.output == "ok"
    assert project_run(events).status is RunStatus.SUCCEEDED
