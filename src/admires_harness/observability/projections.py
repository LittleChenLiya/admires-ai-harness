"""Pure projections over the append-only event stream."""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime
from uuid import UUID

from pydantic import Field

from admires_harness.contracts import EventType, JsonValue, RunEvent, RunStatus, StrictModel


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


def _integer(value: JsonValue, default: int = 0) -> int:
    return (
        value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else default
    )


def _number(value: JsonValue, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return default


def project_run(events: list[RunEvent]) -> RunView:
    """Materialize a run without consulting mutable application state."""

    if not events:
        raise ValueError("cannot project an empty event stream")
    ordered = sorted(events, key=lambda event: event.sequence)
    for expected, event in enumerate(ordered):
        if event.run_id != ordered[0].run_id:
            raise ValueError("event stream contains multiple run IDs")
        if event.sequence != expected:
            raise ValueError("event stream has a sequence gap")
    created = ordered[0]
    graph_name = str(created.payload.get("graph_name", "unknown"))
    graph_version = str(created.payload.get("graph_version", "unknown"))
    status = RunStatus.CREATED
    output: str | None = None
    error: str | None = None
    metrics = RunMetrics()
    ended_at: datetime | None = None
    demo = graph_name.removesuffix("-agent")
    task_input = ""

    for event in ordered:
        payload_status = event.payload.get("status")
        # A node-boundary checkpoint may already contain ``succeeded`` while the
        # runtime is still appending its authoritative ``run.completed`` event.
        # Only lifecycle events are allowed to advance the public run status; this
        # keeps API polling and replay snapshots linearizable at terminal states.
        if event.type in {
            EventType.RUN_STATUS,
            EventType.RUN_COMPLETED,
            EventType.RUN_FAILED,
        } and isinstance(payload_status, str):
            with suppress(ValueError):
                status = RunStatus(payload_status)
        elif event.type is EventType.RUN_STARTED:
            status = RunStatus.RUNNING
        elif event.type is EventType.RUN_COMPLETED:
            status = RunStatus.SUCCEEDED
        elif event.type is EventType.RUN_FAILED:
            status = RunStatus.FAILED

        if event.type is EventType.NODE_COMPLETED:
            update = event.payload.get("update")
            if isinstance(update, dict):
                raw_input = update.get("input")
                raw_demo = update.get("demo")
                if isinstance(raw_input, str):
                    task_input = raw_input
                if isinstance(raw_demo, str):
                    demo = raw_demo

        raw_output = event.payload.get("output")
        if isinstance(raw_output, str):
            output = raw_output
        raw_error = event.payload.get("error")
        if isinstance(raw_error, str):
            error = raw_error
        raw_metrics = event.payload.get("metrics")
        if isinstance(raw_metrics, dict):
            metrics = RunMetrics(
                steps=_integer(raw_metrics.get("steps")),
                model_calls=_integer(raw_metrics.get("model_calls")),
                tool_calls=_integer(raw_metrics.get("tool_calls")),
                input_tokens=_integer(raw_metrics.get("input_tokens")),
                output_tokens=_integer(raw_metrics.get("output_tokens")),
                cost_usd=_number(raw_metrics.get("cost_usd")),
            )
        if status.terminal:
            ended_at = event.timestamp

    # Initial state is intentionally included in run.created for durable API projection.
    initial = created.payload.get("initial_state")
    if isinstance(initial, dict):
        initial_input = initial.get("input")
        initial_demo = initial.get("demo")
        if isinstance(initial_input, str):
            task_input = initial_input
        if isinstance(initial_demo, str):
            demo = initial_demo

    return RunView(
        run_id=created.run_id,
        trace_id=created.trace_id,
        parent_run_id=created.parent_run_id,
        demo=demo,
        input=task_input,
        graph_name=graph_name,
        graph_version=graph_version,
        status=status,
        started_at=created.timestamp,
        ended_at=ended_at,
        output=output,
        error=error,
        metrics=metrics,
    )
