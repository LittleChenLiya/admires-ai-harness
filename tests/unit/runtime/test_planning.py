from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from uuid import uuid4

import pytest
from pydantic import ValidationError

from admires_harness.contracts import (
    EventType,
    JsonValue,
    RunBudget,
    RunEvent,
    RunStatus,
    utc_now,
)
from admires_harness.runtime import (
    END,
    GraphRuntime,
    InMemoryCheckpointStore,
    InMemoryEventStore,
    NodeResult,
    Plan,
    PlanStep,
    StateGraph,
    execute_plan,
)
from admires_harness.runtime.errors import BudgetExceededError, DelegationError
from admires_harness.runtime.graph import CompiledGraph
from admires_harness.runtime.types import NodeContext, RunResult


def make_run_result(
    *,
    status: RunStatus = RunStatus.SUCCEEDED,
    output: JsonValue = None,
    error: str | None = None,
) -> RunResult:
    return RunResult(
        run_id=uuid4(),
        trace_id=uuid4(),
        graph_name="worker",
        graph_version="1",
        status=status,
        state={},
        output=output,
        started_at=utc_now(),
        ended_at=utc_now(),
        error=error,
    )


class Harness:
    """Deterministic delegate/emit harness driving execute_plan without a runtime."""

    def __init__(self, budget: RunBudget | None = None, *, emit_broken: bool = False) -> None:
        self.run_id = uuid4()
        self.trace_id = uuid4()
        self.budget = budget or RunBudget()
        self.emit_broken = emit_broken
        self.events: list[tuple[str, dict[str, JsonValue]]] = []
        self.starts: list[str] = []
        self.capabilities: dict[str, tuple[str, ...]] = {}
        self.states: dict[str, dict[str, JsonValue]] = {}
        self.handlers: dict[str, object] = {}

    async def emit(self, event_type: EventType, payload: Mapping[str, JsonValue]) -> RunEvent:
        if self.emit_broken:
            raise RuntimeError("event sink unavailable")
        self.events.append((event_type.value, dict(payload)))
        return RunEvent(
            run_id=self.run_id, sequence=len(self.events), type=event_type, trace_id=self.trace_id
        )

    async def delegate(
        self,
        graph: CompiledGraph,
        state: Mapping[str, JsonValue],
        label: str | None,
        budget: RunBudget | None,
        capabilities: Sequence[str],
    ) -> RunResult:
        assert label is not None
        self.starts.append(label)
        self.capabilities[label] = tuple(capabilities)
        self.states[label] = dict(state)
        handler = self.handlers.get(label)
        if handler is None:
            return make_run_result(output={"step": label})
        result = handler(state)  # type: ignore[operator]
        if asyncio.iscoroutine(result):
            return await result  # type: ignore[no-any-return]
        return result  # type: ignore[no-any-return]

    def context(self) -> NodeContext:
        return NodeContext.create(
            run_id=self.run_id,
            trace_id=self.trace_id,
            parent_run_id=None,
            node_id="plan",
            checkpoint_revision=None,
            state={},
            budget=self.budget,
            dependencies={},
            emit=self.emit,
            delegate=self.delegate,
        )

    def plan_events(self, name: str) -> list[dict[str, JsonValue]]:
        return [
            payload
            for event_type, payload in self.events
            if event_type == EventType.LOG.value and payload.get("plan_event") == name
        ]


def assert_no_pending_tasks() -> None:
    leftovers = [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and not task.done()
    ]
    assert leftovers == []


def test_plan_rejects_cycles_and_unknown_dependencies() -> None:
    with pytest.raises(ValidationError, match="unknown dependencies"):
        Plan(
            objective="x",
            steps=(PlanStep(id="a", objective="a", agent="worker", depends_on=("b",)),),
        )
    with pytest.raises(ValidationError, match="cycle"):
        Plan(
            objective="x",
            steps=(
                PlanStep(id="a", objective="a", agent="worker", depends_on=("b",)),
                PlanStep(id="b", objective="b", agent="worker", depends_on=("a",)),
            ),
        )


@pytest.mark.asyncio
async def test_execute_plan_delegates_child_graphs_and_joins_outputs() -> None:
    async def worker(context):  # type: ignore[no-untyped-def]
        dependencies = context.state["dependency_outputs"]
        return NodeResult(output={"objective": context.state["objective"], "deps": dependencies})

    worker_graph = (
        StateGraph("worker")
        .add_node("work", worker)
        .set_entry_point("work")
        .add_edge("work", END)
        .compile()
    )
    plan = Plan(
        objective="research and write",
        steps=(
            PlanStep(id="research", objective="research", agent="worker"),
            PlanStep(
                id="write",
                objective="write",
                agent="worker",
                depends_on=("research",),
            ),
        ),
    )

    async def supervisor(context):  # type: ignore[no-untyped-def]
        return await execute_plan(context, plan, {"worker": worker_graph})

    supervisor_graph = (
        StateGraph("supervisor")
        .add_node("delegate", supervisor)
        .set_entry_point("delegate")
        .add_edge("delegate", END)
        .compile()
    )
    events = InMemoryEventStore()
    runtime = GraphRuntime(event_store=events, checkpoint_store=InMemoryCheckpointStore())

    result = await runtime.run(supervisor_graph)

    assert result.status.value == "succeeded"
    assert result.steps == 3  # supervisor plus both child runs
    outputs = result.state["plan_results"]
    assert outputs["write"]["deps"] == {"research": outputs["research"]}
    parent_events = await events.list(result.run_id)
    assert [event.type.value for event in parent_events].count("agent.delegated") == 2
    assert [event.type.value for event in parent_events].count("agent.joined") == 2


@pytest.mark.asyncio
async def test_execute_plan_rejects_unknown_agent() -> None:
    plan = Plan(objective="x", steps=(PlanStep(id="a", objective="x", agent="missing"),))

    async def supervisor(context):  # type: ignore[no-untyped-def]
        return await execute_plan(context, plan, {})

    graph = StateGraph("supervisor").add_node("a", supervisor).set_entry_point("a").compile()
    runtime = GraphRuntime(
        event_store=InMemoryEventStore(), checkpoint_store=InMemoryCheckpointStore()
    )
    result = await runtime.run(graph)
    assert result.status.value == "failed"
    assert "unknown plan agents" in (result.error or "")

    async def impossible_delegate(context):  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="positive"):
            await execute_plan(context, plan, {}, max_concurrency=0)
        return NodeResult(finish=True)

    impossible = (
        StateGraph("invalid-concurrency")
        .add_node("a", impossible_delegate)
        .set_entry_point("a")
        .compile()
    )
    concurrency_result = await runtime.run(impossible)
    assert concurrency_result.status.value == "succeeded"


def test_plan_rejects_empty_steps() -> None:
    with pytest.raises(ValidationError):
        Plan(objective="x", steps=())


def test_topological_order_preserves_author_order_for_ready_steps() -> None:
    plan = Plan(
        objective="x",
        steps=(
            PlanStep(id="b", objective="b", agent="worker"),
            PlanStep(id="a", objective="a", agent="worker"),
            PlanStep(id="c", objective="c", agent="worker", depends_on=("a",)),
        ),
    )
    assert plan.topological_order() == ("b", "a", "c")


@pytest.mark.asyncio
async def test_execute_plan_multi_layer_dag_stable_order_and_dependency_outputs() -> None:
    harness = Harness()
    plan = Plan(
        objective="layered",
        steps=(
            PlanStep(id="a", objective="a", agent="worker", capabilities=("db",)),
            PlanStep(id="b", objective="b", agent="worker", depends_on=("a",)),
            PlanStep(id="c", objective="c", agent="worker", depends_on=("a",)),
            PlanStep(id="d", objective="d", agent="worker", depends_on=("b", "c")),
        ),
    )

    result = await execute_plan(harness.context(), plan, {"worker": None}, max_concurrency=2)

    assert harness.starts == ["a", "b", "c", "d"]
    assert harness.capabilities["a"] == ("db",)
    assert harness.states["d"]["dependency_outputs"] == {
        "b": {"step": "b"},
        "c": {"step": "c"},
    }
    assert result.output == {step: {"step": step} for step in ("a", "b", "c", "d")}
    assert result.update["plan_failures"] == {}
    assert result.update["plan_skipped"] == {}
    assert result.metadata == {
        "planned_steps": 4,
        "completed_steps": 4,
        "failed_steps": 0,
        "skipped_steps": 0,
        "event_callback_errors": 0,
    }
    assert [event["step_id"] for event in harness.plan_events("plan_step_started")] == [
        "a",
        "b",
        "c",
        "d",
    ]
    assert len(harness.plan_events("plan_started")) == 1
    assert len(harness.plan_events("plan_completed")) == 1
    assert_no_pending_tasks()


@pytest.mark.asyncio
async def test_execute_plan_enforces_concurrency_limit_in_stable_order() -> None:
    harness = Harness()
    step_ids = ("a", "b", "c")
    gates = {step_id: asyncio.Event() for step_id in step_ids}
    entered = {step_id: asyncio.Event() for step_id in step_ids}
    active = 0
    peak = 0

    def make_handler(step_id: str):  # type: ignore[no-untyped-def]
        async def run(state):  # type: ignore[no-untyped-def]
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            entered[step_id].set()
            await gates[step_id].wait()
            active -= 1
            return make_run_result(output={"step": step_id})

        return run

    for step_id in step_ids:
        harness.handlers[step_id] = make_handler(step_id)
    plan = Plan(
        objective="x",
        steps=tuple(
            PlanStep(id=step_id, objective=step_id, agent="worker") for step_id in step_ids
        ),
    )

    task = asyncio.ensure_future(
        execute_plan(harness.context(), plan, {"worker": None}, max_concurrency=2)
    )
    await asyncio.wait_for(entered["a"].wait(), 1)
    await asyncio.wait_for(entered["b"].wait(), 1)
    await asyncio.sleep(0)
    assert not entered["c"].is_set()

    gates["a"].set()
    await asyncio.wait_for(entered["c"].wait(), 1)
    gates["b"].set()
    gates["c"].set()
    result = await asyncio.wait_for(task, 1)

    assert peak == 2
    assert harness.starts == ["a", "b", "c"]
    assert result.metadata["completed_steps"] == 3
    assert_no_pending_tasks()


@pytest.mark.asyncio
async def test_execute_plan_failure_propagates_to_dependents_without_fail_fast() -> None:
    harness = Harness()
    harness.handlers["a"] = lambda state: make_run_result(status=RunStatus.FAILED, error="boom")
    plan = Plan(
        objective="x",
        steps=(
            PlanStep(id="a", objective="a", agent="worker"),
            PlanStep(id="b", objective="b", agent="worker", depends_on=("a",)),
            PlanStep(id="c", objective="c", agent="worker"),
            PlanStep(id="d", objective="d", agent="worker", depends_on=("b",)),
        ),
    )

    result = await execute_plan(harness.context(), plan, {"worker": None}, fail_fast=False)

    assert result.update["plan_failures"] == {"a": "boom"}
    assert result.update["plan_skipped"] == {
        "b": "dependency 'a' failed",
        "d": "dependency 'b' skipped",
    }
    assert result.update["plan_results"] == {"c": {"step": "c"}}
    assert harness.starts == ["a", "c"]
    assert result.metadata["failed_steps"] == 1
    assert result.metadata["skipped_steps"] == 2
    skipped = {
        event["step_id"]: event["reason"] for event in harness.plan_events("plan_step_skipped")
    }
    assert skipped == {"b": "dependency 'a' failed", "d": "dependency 'b' skipped"}
    assert_no_pending_tasks()


@pytest.mark.asyncio
async def test_execute_plan_fail_fast_cancels_started_steps() -> None:
    harness = Harness()
    entered_b = asyncio.Event()
    cancelled_b = asyncio.Event()

    async def run_b(state):  # type: ignore[no-untyped-def]
        entered_b.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled_b.set()
        return make_run_result()

    harness.handlers["a"] = lambda state: make_run_result(status=RunStatus.FAILED, error="boom")
    harness.handlers["b"] = run_b
    plan = Plan(
        objective="x",
        steps=(
            PlanStep(id="a", objective="a", agent="worker"),
            PlanStep(id="b", objective="b", agent="worker"),
            PlanStep(id="c", objective="c", agent="worker", depends_on=("b",)),
        ),
    )

    with pytest.raises(DelegationError, match="delegated step failed"):
        await execute_plan(harness.context(), plan, {"worker": None}, max_concurrency=2)

    assert entered_b.is_set()
    assert cancelled_b.is_set()
    skipped = {
        event["step_id"]: event["reason"] for event in harness.plan_events("plan_step_skipped")
    }
    assert skipped["b"] == "fail-fast: cancelled after step 'a' failed"
    assert skipped["c"] == "fail-fast: cancelled after step 'a' failed"
    assert_no_pending_tasks()


@pytest.mark.asyncio
async def test_execute_plan_external_cancellation_converges_started_steps() -> None:
    harness = Harness()
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def run_a(state):  # type: ignore[no-untyped-def]
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return make_run_result()

    harness.handlers["a"] = run_a
    plan = Plan(objective="x", steps=(PlanStep(id="a", objective="a", agent="worker"),))

    task = asyncio.ensure_future(execute_plan(harness.context(), plan, {"worker": None}))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled.is_set()
    skipped = harness.plan_events("plan_step_skipped")
    assert skipped[0]["reason"] == "cancelled: external cancellation"
    assert_no_pending_tasks()


@pytest.mark.asyncio
async def test_execute_plan_step_timeout_marks_failure_and_skips_dependents() -> None:
    harness = Harness()

    async def run_a(state):  # type: ignore[no-untyped-def]
        await asyncio.Event().wait()
        return make_run_result()

    harness.handlers["a"] = run_a
    plan = Plan(
        objective="x",
        steps=(
            PlanStep(id="a", objective="a", agent="worker"),
            PlanStep(id="b", objective="b", agent="worker", depends_on=("a",)),
        ),
    )

    result = await execute_plan(
        harness.context(),
        plan,
        {"worker": None},
        fail_fast=False,
        step_timeout_seconds=0.05,
    )

    failures = result.update["plan_failures"]
    assert failures["a"] == "timed out after 0.05 seconds"
    assert result.update["plan_skipped"] == {"b": "dependency 'a' failed"}
    assert harness.starts == ["a"]
    assert_no_pending_tasks()


@pytest.mark.asyncio
async def test_execute_plan_rejects_invalid_step_timeout() -> None:
    harness = Harness()
    plan = Plan(objective="x", steps=(PlanStep(id="a", objective="a", agent="worker"),))
    with pytest.raises(ValueError, match="step_timeout_seconds"):
        await execute_plan(harness.context(), plan, {"worker": None}, step_timeout_seconds=0)


@pytest.mark.asyncio
async def test_execute_plan_rejects_insufficient_parent_budget() -> None:
    harness = Harness(budget=RunBudget(max_steps=1))
    plan = Plan(objective="x", steps=(PlanStep(id="a", objective="a", agent="worker"),))

    with pytest.raises(DelegationError, match="parent budget is too small"):
        await execute_plan(harness.context(), plan, {"worker": None})
    assert harness.starts == []


@pytest.mark.asyncio
async def test_execute_plan_records_mid_plan_budget_exhaustion() -> None:
    harness = Harness()

    def raise_budget(state):  # type: ignore[no-untyped-def]
        raise BudgetExceededError("no parent budget remains for a child run")

    harness.handlers["a"] = raise_budget
    plan = Plan(
        objective="x",
        steps=(
            PlanStep(id="a", objective="a", agent="worker"),
            PlanStep(id="b", objective="b", agent="worker", depends_on=("a",)),
        ),
    )

    result = await execute_plan(harness.context(), plan, {"worker": None}, fail_fast=False)

    assert result.update["plan_failures"] == {
        "a": "budget exceeded: no parent budget remains for a child run"
    }
    assert result.update["plan_skipped"] == {"b": "dependency 'a' failed"}
    assert_no_pending_tasks()


@pytest.mark.asyncio
async def test_execute_plan_tolerates_event_callback_exceptions() -> None:
    harness = Harness(emit_broken=True)
    plan = Plan(
        objective="x",
        steps=(
            PlanStep(id="a", objective="a", agent="worker"),
            PlanStep(id="b", objective="b", agent="worker", depends_on=("a",)),
        ),
    )

    result = await execute_plan(harness.context(), plan, {"worker": None})

    assert harness.starts == ["a", "b"]
    assert result.metadata["completed_steps"] == 2
    assert result.metadata["event_callback_errors"] > 0
    assert_no_pending_tasks()
