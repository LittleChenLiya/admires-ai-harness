"""Pure projections over the append-only event stream.

The projection layer is the read side of the event-sourced runtime. It must be
safe against the realities of durable delivery: at-least-once appends produce
duplicate events, reconnects surface out-of-order batches, and a crashed writer
can leave sequence gaps. Projections therefore

* deduplicate by event_id and resolve sequence conflicts deterministically, so
  re-applying the same stream is a no-op (idempotent replay);
* reconstruct initial state, node states, approvals, tool results, metrics, and
  the terminal lifecycle from the same envelope every client consumes;
* report every anomaly as an explicit ProjectionDiagnostic instead of silently
  fabricating a plausible-looking snapshot.

project_run is the strict compatibility surface: it raises ValueError for
integrity failures (empty stream, mixed run IDs, sequence gaps, a missing
run.created) exactly as earlier versions did. project_run_checked never raises
for stream content and instead returns a RunProjection pairing the best-effort
view with its diagnostics.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from itertools import pairwise
from uuid import UUID

from pydantic import Field, ValidationError

from admires_harness.contracts import (
    ApprovalDecision,
    ApprovalRequest,
    EventType,
    JsonValue,
    RunEvent,
    RunStatus,
    StrictModel,
)


class DiagnosticCode(StrEnum):
    """Machine-readable reasons a projection had to repair or skip stream data."""

    OUT_OF_ORDER = "out_of_order"
    DUPLICATE_EVENT = "duplicate_event"
    SEQUENCE_CONFLICT = "sequence_conflict"
    SEQUENCE_GAP = "sequence_gap"
    MISSING_RUN_CREATED = "missing_run_created"
    MIXED_RUN_IDS = "mixed_run_ids"
    UNKNOWN_EVENT_TYPE = "unknown_event_type"
    MALFORMED_PAYLOAD = "malformed_payload"


class ProjectionDiagnostic(StrictModel):
    """One explicit, non-fatal anomaly found while projecting an event stream."""

    code: DiagnosticCode
    message: str
    sequence: int | None = None
    event_id: UUID | None = None


class NodeStatus(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CONSUMED = "consumed"


class ToolStatus(StrEnum):
    REQUESTED = "requested"
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class RunMetrics(StrictModel):
    steps: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)


class NodeView(StrictModel):
    """Reconstructed state of one graph node within the run."""

    node_id: str
    status: NodeStatus
    started_at: datetime | None = None
    ended_at: datetime | None = None
    next_node: str | None = None
    error: str | None = None


class ApprovalView(StrictModel):
    """Reconstructed lifecycle of one approval request."""

    request_id: str
    tool_name: str | None = None
    risk: str | None = None
    status: ApprovalStatus = ApprovalStatus.PENDING


class ToolView(StrictModel):
    """Reconstructed outcome of one tool invocation."""

    tool: str
    status: ToolStatus = ToolStatus.REQUESTED
    requested_at: datetime | None = None
    ended_at: datetime | None = None
    error: str | None = None
    error_code: str | None = None


class ChildRunView(StrictModel):
    """Reconstructed record of a delegated child run."""

    run_id: str
    label: str | None = None
    graph_name: str | None = None
    status: RunStatus | None = None
    metrics: RunMetrics | None = None


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
    state: dict[str, JsonValue] = Field(default_factory=dict)
    nodes: tuple[NodeView, ...] = ()
    approvals: tuple[ApprovalView, ...] = ()
    tools: tuple[ToolView, ...] = ()
    child_runs: tuple[ChildRunView, ...] = ()


class RunProjection(StrictModel):
    """A best-effort run view plus every anomaly repaired to produce it."""

    view: RunView
    diagnostics: tuple[ProjectionDiagnostic, ...] = ()
    applied: int = Field(default=0, ge=0)
    dropped: int = Field(default=0, ge=0)


def _integer(value: JsonValue, default: int = 0) -> int:
    return (
        value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else default
    )


def _number(value: JsonValue, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return default


def _metrics_from_mapping(raw: dict[str, JsonValue]) -> RunMetrics:
    return RunMetrics(
        steps=_integer(raw.get("steps")),
        model_calls=_integer(raw.get("model_calls")),
        tool_calls=_integer(raw.get("tool_calls")),
        input_tokens=_integer(raw.get("input_tokens")),
        output_tokens=_integer(raw.get("output_tokens")),
        cost_usd=_number(raw.get("cost_usd")),
    )


def _event_type(event: RunEvent) -> EventType | None:
    # Tolerate loosely typed historical envelopes whose type is a plain string,
    # and envelopes from a newer schema version with types we do not know yet.
    try:
        return EventType(event.type)
    except ValueError:
        return None


@dataclass(slots=True)
class _NodeAcc:
    node_id: str
    status: NodeStatus = NodeStatus.STARTED
    started_at: datetime | None = None
    ended_at: datetime | None = None
    next_node: str | None = None
    error: str | None = None


@dataclass(slots=True)
class _ApprovalAcc:
    request_id: str
    tool_name: str | None = None
    risk: str | None = None
    status: ApprovalStatus = ApprovalStatus.PENDING


@dataclass(slots=True)
class _ToolAcc:
    tool: str
    status: ToolStatus = ToolStatus.REQUESTED
    requested_at: datetime | None = None
    ended_at: datetime | None = None
    error: str | None = None
    error_code: str | None = None


@dataclass(slots=True)
class _ChildAcc:
    run_id: str
    label: str | None = None
    graph_name: str | None = None
    status: RunStatus | None = None
    metrics: RunMetrics | None = None


@dataclass(slots=True)
class _Accumulator:
    status: RunStatus = RunStatus.CREATED
    demo: str = ""
    task_input: str = ""
    output: str | None = None
    error: str | None = None
    ended_at: datetime | None = None
    state: dict[str, JsonValue] = field(default_factory=dict)
    steps: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    metrics_override: RunMetrics | None = None
    nodes: dict[str, _NodeAcc] = field(default_factory=dict)
    approvals: dict[str, _ApprovalAcc] = field(default_factory=dict)
    tools: list[_ToolAcc] = field(default_factory=list)
    children: dict[str, _ChildAcc] = field(default_factory=dict)

    def metrics(self) -> RunMetrics:
        if self.metrics_override is not None:
            return self.metrics_override
        return RunMetrics(
            steps=self.steps,
            model_calls=self.model_calls,
            tool_calls=self.tool_calls,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost_usd=self.cost_usd,
        )


_LIFECYCLE_TYPES = {EventType.RUN_STATUS, EventType.RUN_COMPLETED, EventType.RUN_FAILED}


def _malformed(
    diagnostics: list[ProjectionDiagnostic], event: RunEvent, field_name: str, detail: str
) -> None:
    diagnostics.append(
        ProjectionDiagnostic(
            code=DiagnosticCode.MALFORMED_PAYLOAD,
            message=(
                f"event {event.event_id} ({event.type}) has a malformed "
                f"{field_name}: {detail}; the field was ignored"
            ),
            sequence=event.sequence,
            event_id=event.event_id,
        )
    )


def _apply_lifecycle(
    acc: _Accumulator,
    event: RunEvent,
    etype: EventType,
    diagnostics: list[ProjectionDiagnostic],
) -> None:
    payload_status = event.payload.get("status")
    # A node-boundary checkpoint may already contain "succeeded" while the
    # runtime is still appending its authoritative run.completed event. Only
    # lifecycle events are allowed to advance the public run status; this keeps
    # API polling and replay snapshots linearizable at terminal states.
    if etype in _LIFECYCLE_TYPES and payload_status is not None:
        if not isinstance(payload_status, str):
            _malformed(diagnostics, event, "status", "expected a status string")
        else:
            try:
                acc.status = RunStatus(payload_status)
            except ValueError:
                _malformed(diagnostics, event, "status", f"unknown status {payload_status!r}")
    if etype is EventType.RUN_STARTED:
        acc.status = RunStatus.RUNNING
    elif etype is EventType.RUN_COMPLETED:
        acc.status = RunStatus.SUCCEEDED
    elif etype is EventType.RUN_FAILED:
        acc.status = RunStatus.FAILED


def _apply_node_event(acc: _Accumulator, event: RunEvent, etype: EventType) -> None:
    if event.node_id is None:
        return
    node = acc.nodes.setdefault(event.node_id, _NodeAcc(node_id=event.node_id))
    if etype is EventType.NODE_STARTED:
        node.status = NodeStatus.STARTED
        node.started_at = event.timestamp
    elif etype is EventType.NODE_COMPLETED:
        node.status = NodeStatus.COMPLETED
        node.ended_at = event.timestamp
        next_node = event.payload.get("next_node")
        if isinstance(next_node, str):
            node.next_node = next_node
    elif etype is EventType.NODE_FAILED:
        node.status = NodeStatus.FAILED
        node.ended_at = event.timestamp
        error = event.payload.get("error")
        if isinstance(error, str):
            node.error = error


def _apply_node_completed(
    acc: _Accumulator, event: RunEvent, diagnostics: list[ProjectionDiagnostic]
) -> None:
    update = event.payload.get("update")
    if update is not None:
        if isinstance(update, dict):
            acc.state.update(update)
            raw_input = update.get("input")
            raw_demo = update.get("demo")
            if isinstance(raw_input, str):
                acc.task_input = raw_input
            if isinstance(raw_demo, str):
                acc.demo = raw_demo
        else:
            _malformed(diagnostics, event, "update", "expected an object")
    # Metrics accumulate per completed node so incomplete runs (paused, waiting
    # for approval, still running) still project their consumption so far. A
    # terminal metrics payload is cumulative and authoritative; it overrides.
    acc.steps += 1
    usage = event.payload.get("usage")
    if isinstance(usage, dict):
        acc.input_tokens += _integer(usage.get("input_tokens"))
        acc.output_tokens += _integer(usage.get("output_tokens"))
        acc.cost_usd += _number(usage.get("cost_usd"))
    acc.model_calls += _integer(event.payload.get("model_calls"))
    acc.tool_calls += _integer(event.payload.get("tool_calls"))


def _apply_approval(
    acc: _Accumulator,
    event: RunEvent,
    etype: EventType,
    diagnostics: list[ProjectionDiagnostic],
) -> None:
    if etype is EventType.APPROVAL_REQUESTED:
        raw_request = event.payload.get("request")
        try:
            request = ApprovalRequest.model_validate(raw_request)
        except ValidationError:
            _malformed(diagnostics, event, "request", "not a valid approval request")
            return
        acc.approvals[str(request.request_id)] = _ApprovalAcc(
            request_id=str(request.request_id),
            tool_name=request.tool_name,
            risk=request.risk.value,
        )
        return
    if etype is EventType.APPROVAL_RESOLVED:
        raw_decision = event.payload.get("decision")
        try:
            decision = ApprovalDecision.model_validate(raw_decision)
        except ValidationError:
            _malformed(diagnostics, event, "decision", "not a valid approval decision")
            return
        key = str(decision.request_id)
        approval = acc.approvals.setdefault(key, _ApprovalAcc(request_id=key))
        approval.status = ApprovalStatus.APPROVED if decision.approved else ApprovalStatus.REJECTED
        return
    raw_request = event.payload.get("request")
    request_id = raw_request.get("request_id") if isinstance(raw_request, dict) else None
    if not isinstance(request_id, str):
        _malformed(diagnostics, event, "request", "missing request_id for consumed approval")
        return
    approval = acc.approvals.setdefault(request_id, _ApprovalAcc(request_id=request_id))
    approval.status = ApprovalStatus.CONSUMED


def _apply_tool(
    acc: _Accumulator,
    event: RunEvent,
    etype: EventType,
    diagnostics: list[ProjectionDiagnostic],
) -> None:
    tool = event.payload.get("tool")
    if not isinstance(tool, str):
        _malformed(diagnostics, event, "tool", "expected a tool name")
        return
    if etype is EventType.TOOL_REQUESTED:
        acc.tools.append(_ToolAcc(tool=tool, requested_at=event.timestamp))
        return
    target = next(
        (
            entry
            for entry in reversed(acc.tools)
            if entry.tool == tool and entry.status in {ToolStatus.REQUESTED, ToolStatus.STARTED}
        ),
        None,
    )
    if target is None:
        # Older streams may lack tool.requested; reconstruct the invocation.
        target = _ToolAcc(tool=tool)
        acc.tools.append(target)
    if etype is EventType.TOOL_STARTED:
        target.status = ToolStatus.STARTED
    elif etype is EventType.TOOL_COMPLETED:
        target.status = ToolStatus.COMPLETED
        target.ended_at = event.timestamp
    else:
        target.status = ToolStatus.FAILED
        target.ended_at = event.timestamp
        error = event.payload.get("error")
        error_code = event.payload.get("error_code")
        if isinstance(error, str):
            target.error = error
        if isinstance(error_code, str):
            target.error_code = error_code


def _apply_child(
    acc: _Accumulator,
    event: RunEvent,
    etype: EventType,
    diagnostics: list[ProjectionDiagnostic],
) -> None:
    child_id = event.payload.get("child_run_id")
    if not isinstance(child_id, str):
        _malformed(diagnostics, event, "child_run_id", "expected a child run ID")
        return
    child = acc.children.setdefault(child_id, _ChildAcc(run_id=child_id))
    label = event.payload.get("label")
    if isinstance(label, str):
        child.label = label
    if etype is EventType.AGENT_DELEGATED:
        graph_name = event.payload.get("graph_name")
        if isinstance(graph_name, str):
            child.graph_name = graph_name
        return
    status = event.payload.get("status")
    if status is not None:
        if isinstance(status, str):
            try:
                child.status = RunStatus(status)
            except ValueError:
                _malformed(diagnostics, event, "status", f"unknown child status {status!r}")
        else:
            _malformed(diagnostics, event, "status", "expected a status string")
    metrics = event.payload.get("metrics")
    if metrics is not None:
        if isinstance(metrics, dict):
            child.metrics = _metrics_from_mapping(metrics)
        else:
            _malformed(diagnostics, event, "metrics", "expected an object")


def _apply(
    acc: _Accumulator,
    event: RunEvent,
    diagnostics: list[ProjectionDiagnostic],
) -> bool:
    """Fold one event into the accumulator. Returns False if the event was skipped."""

    etype = _event_type(event)
    if etype is None:
        diagnostics.append(
            ProjectionDiagnostic(
                code=DiagnosticCode.UNKNOWN_EVENT_TYPE,
                message=f"unknown event type {event.type!r}; the event was skipped",
                sequence=event.sequence,
                event_id=event.event_id,
            )
        )
        return False
    _apply_lifecycle(acc, event, etype, diagnostics)
    if etype in {EventType.NODE_STARTED, EventType.NODE_COMPLETED, EventType.NODE_FAILED}:
        _apply_node_event(acc, event, etype)
    if etype is EventType.NODE_COMPLETED:
        _apply_node_completed(acc, event, diagnostics)
    if etype in {
        EventType.APPROVAL_REQUESTED,
        EventType.APPROVAL_RESOLVED,
        EventType.APPROVAL_CONSUMED,
    }:
        _apply_approval(acc, event, etype, diagnostics)
    if etype in {
        EventType.TOOL_REQUESTED,
        EventType.TOOL_STARTED,
        EventType.TOOL_COMPLETED,
        EventType.TOOL_FAILED,
    }:
        _apply_tool(acc, event, etype, diagnostics)
    if etype in {EventType.AGENT_DELEGATED, EventType.AGENT_JOINED}:
        _apply_child(acc, event, etype, diagnostics)
    if etype in _LIFECYCLE_TYPES:
        raw_metrics = event.payload.get("metrics")
        if raw_metrics is not None:
            if isinstance(raw_metrics, dict):
                acc.metrics_override = _metrics_from_mapping(raw_metrics)
            else:
                _malformed(diagnostics, event, "metrics", "expected an object")
    raw_output = event.payload.get("output")
    if isinstance(raw_output, str):
        acc.output = raw_output
    raw_error = event.payload.get("error")
    if isinstance(raw_error, str):
        acc.error = raw_error
    if acc.status.terminal:
        acc.ended_at = event.timestamp
    return True


def _prepare(
    events: list[RunEvent], diagnostics: list[ProjectionDiagnostic]
) -> tuple[list[RunEvent], int]:
    """Return the deterministic, deduplicated event order and the drop count."""

    dropped = 0
    sequences = [event.sequence for event in events]
    if any(earlier > later for earlier, later in pairwise(sequences)):
        diagnostics.append(
            ProjectionDiagnostic(
                code=DiagnosticCode.OUT_OF_ORDER,
                message="event stream arrived out of order; events were reordered by sequence",
            )
        )
    # A total deterministic order makes the projection independent of delivery
    # order, which is what keeps repeated application of the same stream stable.
    ordered = sorted(
        events, key=lambda event: (event.sequence, event.timestamp, str(event.event_id))
    )

    primary_run_id = ordered[0].run_id
    same_run: list[RunEvent] = []
    for event in ordered:
        if event.run_id != primary_run_id:
            diagnostics.append(
                ProjectionDiagnostic(
                    code=DiagnosticCode.MIXED_RUN_IDS,
                    message=(
                        f"event {event.event_id} belongs to run {event.run_id}, not "
                        f"{primary_run_id}; the event was dropped"
                    ),
                    sequence=event.sequence,
                    event_id=event.event_id,
                )
            )
            dropped += 1
        else:
            same_run.append(event)

    seen: set[UUID] = set()
    unique: list[RunEvent] = []
    for event in same_run:
        if event.event_id in seen:
            diagnostics.append(
                ProjectionDiagnostic(
                    code=DiagnosticCode.DUPLICATE_EVENT,
                    message=f"duplicate delivery of event {event.event_id}; dropped",
                    sequence=event.sequence,
                    event_id=event.event_id,
                )
            )
            dropped += 1
            continue
        seen.add(event.event_id)
        unique.append(event)

    by_sequence: dict[int, list[RunEvent]] = {}
    for event in unique:
        by_sequence.setdefault(event.sequence, []).append(event)
    final: list[RunEvent] = []
    for sequence in sorted(by_sequence):
        group = by_sequence[sequence]
        final.append(group[0])
        for conflict in group[1:]:
            diagnostics.append(
                ProjectionDiagnostic(
                    code=DiagnosticCode.SEQUENCE_CONFLICT,
                    message=(
                        f"multiple distinct events claim sequence {sequence}; kept "
                        f"{group[0].event_id} and dropped {conflict.event_id}"
                    ),
                    sequence=sequence,
                    event_id=conflict.event_id,
                )
            )
            dropped += 1

    present = {event.sequence for event in final}
    missing = [number for number in range(max(present) + 1) if number not in present]
    if missing:
        shown = ", ".join(str(number) for number in missing[:5])
        suffix = ", ..." if len(missing) > 5 else ""
        diagnostics.append(
            ProjectionDiagnostic(
                code=DiagnosticCode.SEQUENCE_GAP,
                message=f"event stream has a sequence gap (missing sequences: {shown}{suffix})",
            )
        )
    return final, dropped


def project_run_checked(events: Iterable[RunEvent]) -> RunProjection:
    """Project an event stream, reporting anomalies instead of raising them.

    Only an empty stream raises, because no run identity exists to anchor a view.
    Everything else produces a best-effort view plus explicit diagnostics.
    """

    materialized = list(events)
    if not materialized:
        raise ValueError("cannot project an empty event stream")
    diagnostics: list[ProjectionDiagnostic] = []
    final, dropped = _prepare(materialized, diagnostics)

    created = next((event for event in final if _event_type(event) is EventType.RUN_CREATED), None)
    if created is None:
        diagnostics.append(
            ProjectionDiagnostic(
                code=DiagnosticCode.MISSING_RUN_CREATED,
                message="event stream is missing run.created; identity fields are inferred",
            )
        )
    anchor = created if created is not None else final[0]

    graph_name = "unknown"
    graph_version = "unknown"
    initial_state: dict[str, JsonValue] = {}
    if created is not None:
        raw_name = created.payload.get("graph_name")
        if isinstance(raw_name, str):
            graph_name = raw_name
        elif raw_name is not None:
            _malformed(diagnostics, created, "graph_name", "expected a string")
        raw_version = created.payload.get("graph_version")
        if isinstance(raw_version, str):
            graph_version = raw_version
        elif raw_version is not None:
            _malformed(diagnostics, created, "graph_version", "expected a string")
        raw_initial = created.payload.get("initial_state")
        if isinstance(raw_initial, dict):
            initial_state = raw_initial
        elif raw_initial is not None:
            _malformed(diagnostics, created, "initial_state", "expected an object")

    acc = _Accumulator()
    acc.state = dict(initial_state)
    acc.demo = graph_name.removesuffix("-agent")
    applied = 0
    for event in final:
        if _apply(acc, event, diagnostics):
            applied += 1
        else:
            dropped += 1

    # Initial state is intentionally included in run.created for durable API
    # projection, and it wins over later node updates for identity fields.
    initial_input = initial_state.get("input")
    initial_demo = initial_state.get("demo")
    if isinstance(initial_input, str):
        acc.task_input = initial_input
    if isinstance(initial_demo, str):
        acc.demo = initial_demo

    view = RunView(
        run_id=anchor.run_id,
        trace_id=anchor.trace_id,
        parent_run_id=anchor.parent_run_id,
        demo=acc.demo,
        input=acc.task_input,
        graph_name=graph_name,
        graph_version=graph_version,
        status=acc.status,
        started_at=anchor.timestamp,
        ended_at=acc.ended_at,
        output=acc.output,
        error=acc.error,
        metrics=acc.metrics(),
        state=acc.state,
        nodes=tuple(
            NodeView(
                node_id=node.node_id,
                status=node.status,
                started_at=node.started_at,
                ended_at=node.ended_at,
                next_node=node.next_node,
                error=node.error,
            )
            for node in acc.nodes.values()
        ),
        approvals=tuple(
            ApprovalView(
                request_id=approval.request_id,
                tool_name=approval.tool_name,
                risk=approval.risk,
                status=approval.status,
            )
            for approval in acc.approvals.values()
        ),
        tools=tuple(
            ToolView(
                tool=tool.tool,
                status=tool.status,
                requested_at=tool.requested_at,
                ended_at=tool.ended_at,
                error=tool.error,
                error_code=tool.error_code,
            )
            for tool in acc.tools
        ),
        child_runs=tuple(
            ChildRunView(
                run_id=child.run_id,
                label=child.label,
                graph_name=child.graph_name,
                status=child.status,
                metrics=child.metrics,
            )
            for child in acc.children.values()
        ),
    )
    return RunProjection(
        view=view, diagnostics=tuple(diagnostics), applied=applied, dropped=dropped
    )


def project_run(events: list[RunEvent]) -> RunView:
    """Materialize a run without consulting mutable application state.

    Strict compatibility surface: integrity failures (mixed run IDs, sequence
    gaps, a missing run.created) raise ValueError. Duplicates, out-of-order
    delivery, unknown event types, and malformed payloads are repaired
    deterministically exactly as project_run_checked does.
    """

    projection = project_run_checked(events)
    for diagnostic in projection.diagnostics:
        if diagnostic.code is DiagnosticCode.MIXED_RUN_IDS:
            raise ValueError("event stream contains multiple run IDs")
        if diagnostic.code is DiagnosticCode.SEQUENCE_GAP:
            raise ValueError("event stream has a sequence gap")
        if diagnostic.code is DiagnosticCode.MISSING_RUN_CREATED:
            raise ValueError("event stream is missing run.created")
    return projection.view
