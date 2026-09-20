"""Pure projections over the append-only event stream.

The projection is a deterministic fold: it depends only on the persisted
RunEvent history, never on mutable in-process state. Delivery defects are
handled explicitly:

* redelivered events (same event_id, or same sequence with identical
  content) are deduplicated, so re-applying a stream is idempotent;
* out-of-order input is sorted by sequence before folding;
* sequence gaps, conflicting duplicates, and mixed run IDs raise ValueError
  instead of silently producing a plausible-looking but fabricated snapshot;
* recoverable anomalies (missing run.created, unknown event types,
  malformed payloads) are recorded as ProjectionIssue diagnostics on the
  resulting view while the remaining events still fold deterministically.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import Field

from admires_harness.contracts import EventType, JsonValue, RunEvent, RunStatus, StrictModel


class ProjectionIssueCode(StrEnum):
    """Stable machine-readable codes for recoverable projection anomalies."""

    MISSING_RUN_CREATED = "missing_run_created"
    UNKNOWN_EVENT_TYPE = "unknown_event_type"
    INVALID_PAYLOAD = "invalid_payload"


class ProjectionIssue(StrictModel):
    """One recoverable anomaly observed while folding the event stream."""

    code: ProjectionIssueCode
    message: str
    sequence: int | None = None


class NodeView(StrictModel):
    """Projected lifecycle of one graph node."""

    node_id: str
    status: Literal["running", "completed", "failed"]
    started_at: datetime
    ended_at: datetime | None = None
    error: str | None = None


class ApprovalView(StrictModel):
    """Projected state of one approval request."""

    request_id: str
    tool_name: str | None = None
    status: Literal["requested", "approved", "rejected", "consumed"]
    decided_by: str | None = None


class ToolView(StrictModel):
    """Projected outcome of one finished tool call."""

    tool: str
    ok: bool
    error_code: str | None = None


class RunMetrics(StrictModel):
    steps: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)


class RunView(StrictModel):
    run_id: UUID
    trace_id: UUID
    parent_run_id: UUID | None = None
    demo: str
    input: str
    graph_name: str
    graph_version: str
    status: RunStatus
    started_at: datetime
    ended_at: datetime | None = None
    output: str | None = None
    error: str | None = None
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    initial_state: dict[str, JsonValue] = Field(default_factory=dict)
    state: dict[str, JsonValue] = Field(default_factory=dict)
    next_node: str | None = None
    checkpoint_revision: int | None = None
    nodes: tuple[NodeView, ...] = ()
    approvals: tuple[ApprovalView, ...] = ()
    tools: tuple[ToolView, ...] = ()
    issues: tuple[ProjectionIssue, ...] = ()


_STATUS_EVENTS = {EventType.RUN_STATUS, EventType.RUN_COMPLETED, EventType.RUN_FAILED}


def _integer(value: JsonValue | None, default: int = 0) -> int:
    return (
        value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else default
    )


def _number(value: JsonValue | None, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return default


def _same_delivery(left: RunEvent, right: RunEvent) -> bool:
    """Two sequence duplicates are one logical delivery only if content matches."""

    return (
        left.type == right.type and left.node_id == right.node_id and left.payload == right.payload
    )


def _normalize(events: list[RunEvent]) -> list[RunEvent]:
    """Deduplicate, order, and integrity-check the raw stream."""

    if not events:
        raise ValueError("cannot project an empty event stream")
    if len({event.run_id for event in events}) > 1:
        raise ValueError("event stream contains multiple run IDs")
    by_id: dict[UUID, RunEvent] = {}
    for event in events:
        existing = by_id.get(event.event_id)
        if existing is None:
            by_id[event.event_id] = event
        elif existing != event:
            raise ValueError(f"event stream has conflicting deliveries of event {event.event_id}")
    by_sequence: dict[int, RunEvent] = {}
    for event in by_id.values():
        existing = by_sequence.get(event.sequence)
        if existing is None:
            by_sequence[event.sequence] = event
        elif not _same_delivery(existing, event):
            raise ValueError(f"event stream has conflicting events at sequence {event.sequence}")
    ordered = [by_sequence[sequence] for sequence in sorted(by_sequence)]
    for expected, event in enumerate(ordered):
        if event.sequence != expected:
            raise ValueError(
                f"event stream has a sequence gap: expected {expected}, found {event.sequence}"
            )
    return ordered


def _tool_name(payload: dict[str, JsonValue]) -> str:
    raw = payload.get("tool")
    return raw if isinstance(raw, str) else "unknown"


def project_run(events: Iterable[RunEvent]) -> RunView:
    """Materialize a run without consulting mutable application state."""

    ordered = _normalize(list(events))
    issues: list[ProjectionIssue] = []

    def report(code: ProjectionIssueCode, message: str, sequence: int | None = None) -> None:
        issues.append(ProjectionIssue(code=code, message=message, sequence=sequence))

    seed = ordered[0] if ordered[0].type is EventType.RUN_CREATED else None
    if seed is None:
        report(
            ProjectionIssueCode.MISSING_RUN_CREATED,
            "stream does not start with run.created; graph metadata and initial state "
            "are projected from fallback defaults",
            sequence=ordered[0].sequence,
        )
        seed = next((event for event in ordered if event.type is EventType.RUN_CREATED), None)

    seed_payload = seed.payload if seed is not None else {}
    graph_name = str(seed_payload.get("graph_name", "unknown"))
    graph_version = str(seed_payload.get("graph_version", "unknown"))
    raw_initial = seed_payload.get("initial_state")
    initial_state = dict(raw_initial) if isinstance(raw_initial, dict) else {}

    status = RunStatus.CREATED
    output: str | None = None
    error: str | None = None
    metrics = RunMetrics()
    ended_at: datetime | None = None
    demo = graph_name.removesuffix("-agent")
    task_input = ""
    state = dict(initial_state)
    next_node: str | None = None
    checkpoint_revision: int | None = None
    nodes: dict[str, NodeView] = {}
    approvals: dict[str, ApprovalView] = {}
    tools: list[ToolView] = []

    for event in ordered:
        event_type = event.type
        if not isinstance(event_type, EventType):
            # Forward compatibility: a newer writer may append event types this
            # reader does not know. Skip them loudly instead of guessing.
            report(
                ProjectionIssueCode.UNKNOWN_EVENT_TYPE,
                f"unknown event type {event_type!r}; event ignored",
                sequence=event.sequence,
            )
            continue
        payload = event.payload

        # A node-boundary checkpoint may already contain a terminal status while
        # the runtime is still appending its authoritative run.completed event.
        # Only lifecycle events are allowed to advance the public run status; this
        # keeps API polling and replay snapshots linearizable at terminal states.
        raw_status = payload.get("status")
        if event_type in _STATUS_EVENTS and isinstance(raw_status, str):
            try:
                status = RunStatus(raw_status)
            except ValueError:
                report(
                    ProjectionIssueCode.INVALID_PAYLOAD,
                    f"unrecognized run status {raw_status!r}; status left unchanged",
                    sequence=event.sequence,
                )
        elif event_type is EventType.RUN_STARTED:
            status = RunStatus.RUNNING
        elif event_type is EventType.RUN_COMPLETED:
            status = RunStatus.SUCCEEDED
        elif event_type is EventType.RUN_FAILED:
            status = RunStatus.FAILED

        if event_type is EventType.NODE_STARTED:
            node_id = event.node_id or "unknown"
            nodes[node_id] = NodeView(node_id=node_id, status="running", started_at=event.timestamp)
        elif event_type in {EventType.NODE_COMPLETED, EventType.NODE_FAILED}:
            node_id = event.node_id or "unknown"
            previous = nodes.get(node_id)
            raw_node_error = payload.get("error")
            nodes[node_id] = NodeView(
                node_id=node_id,
                status="completed" if event_type is EventType.NODE_COMPLETED else "failed",
                started_at=previous.started_at if previous is not None else event.timestamp,
                ended_at=event.timestamp,
                error=raw_node_error if isinstance(raw_node_error, str) else None,
            )

        if event_type is EventType.NODE_COMPLETED:
            update = payload.get("update")
            if isinstance(update, dict):
                state.update(update)
                raw_input = update.get("input")
                raw_demo = update.get("demo")
                if isinstance(raw_input, str):
                    task_input = raw_input
                if isinstance(raw_demo, str):
                    demo = raw_demo
            raw_next = payload.get("next_node")
            next_node = raw_next if isinstance(raw_next, str) else None

        if event_type is EventType.CHECKPOINT_SAVED:
            raw_revision = payload.get("revision")
            if isinstance(raw_revision, int) and not isinstance(raw_revision, bool):
                checkpoint_revision = raw_revision

        if event_type is EventType.TOOL_COMPLETED:
            tools.append(ToolView(tool=_tool_name(payload), ok=True))
        elif event_type is EventType.TOOL_FAILED:
            raw_code = payload.get("error_code")
            tools.append(
                ToolView(
                    tool=_tool_name(payload),
                    ok=False,
                    error_code=raw_code if isinstance(raw_code, str) else None,
                )
            )

        if event_type is EventType.APPROVAL_REQUESTED:
            request = payload.get("request")
            if isinstance(request, dict) and isinstance(request.get("request_id"), str):
                request_id = str(request["request_id"])
                raw_tool = request.get("tool_name")
                approvals[request_id] = ApprovalView(
                    request_id=request_id,
                    tool_name=raw_tool if isinstance(raw_tool, str) else None,
                    status="requested",
                )
            else:
                report(
                    ProjectionIssueCode.INVALID_PAYLOAD,
                    "approval.requested payload is missing a request object",
                    sequence=event.sequence,
                )
        elif event_type is EventType.APPROVAL_RESOLVED:
            decision = payload.get("decision")
            if isinstance(decision, dict) and isinstance(decision.get("request_id"), str):
                request_id = str(decision["request_id"])
                existing = approvals.get(request_id)
                raw_decider = decision.get("decided_by")
                approvals[request_id] = ApprovalView(
                    request_id=request_id,
                    tool_name=existing.tool_name if existing is not None else None,
                    status="approved" if decision.get("approved") is True else "rejected",
                    decided_by=raw_decider if isinstance(raw_decider, str) else None,
                )
            else:
                report(
                    ProjectionIssueCode.INVALID_PAYLOAD,
                    "approval.resolved payload is missing a decision object",
                    sequence=event.sequence,
                )
        elif event_type is EventType.APPROVAL_CONSUMED:
            request = payload.get("request")
            if isinstance(request, dict) and isinstance(request.get("request_id"), str):
                request_id = str(request["request_id"])
                existing = approvals.get(request_id)
                approvals[request_id] = ApprovalView(
                    request_id=request_id,
                    tool_name=existing.tool_name if existing is not None else None,
                    status="consumed",
                    decided_by=existing.decided_by if existing is not None else None,
                )
            else:
                report(
                    ProjectionIssueCode.INVALID_PAYLOAD,
                    "approval.consumed payload is missing a request object",
                    sequence=event.sequence,
                )

        raw_output = payload.get("output")
        if isinstance(raw_output, str):
            output = raw_output
        raw_error = payload.get("error")
        if isinstance(raw_error, str):
            error = raw_error
        if "metrics" in payload:
            raw_metrics = payload.get("metrics")
            if isinstance(raw_metrics, dict):
                metrics = RunMetrics(
                    steps=_integer(raw_metrics.get("steps")),
                    model_calls=_integer(raw_metrics.get("model_calls")),
                    tool_calls=_integer(raw_metrics.get("tool_calls")),
                    input_tokens=_integer(raw_metrics.get("input_tokens")),
                    output_tokens=_integer(raw_metrics.get("output_tokens")),
                    cost_usd=_number(raw_metrics.get("cost_usd")),
                )
            else:
                report(
                    ProjectionIssueCode.INVALID_PAYLOAD,
                    "metrics payload is not an object; metrics left unchanged",
                    sequence=event.sequence,
                )
        if status.terminal:
            ended_at = event.timestamp

    # Initial state is intentionally included in run.created for durable API projection.
    initial_input = initial_state.get("input")
    initial_demo = initial_state.get("demo")
    if isinstance(initial_input, str):
        task_input = initial_input
    if isinstance(initial_demo, str):
        demo = initial_demo

    anchor = seed if seed is not None else ordered[0]
    return RunView(
        run_id=ordered[0].run_id,
        trace_id=anchor.trace_id,
        parent_run_id=anchor.parent_run_id,
        demo=demo,
        input=task_input,
        graph_name=graph_name,
        graph_version=graph_version,
        status=status,
        started_at=anchor.timestamp,
        ended_at=ended_at,
        output=output,
        error=error,
        metrics=metrics,
        initial_state=initial_state,
        state=state,
        next_node=next_node,
        checkpoint_revision=checkpoint_revision,
        nodes=tuple(nodes.values()),
        approvals=tuple(approvals.values()),
        tools=tuple(tools),
        issues=tuple(issues),
    )
