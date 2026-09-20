"""Deterministic async tests for the dependency-aware plan scheduler."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from admires_harness.contracts import JsonValue, RunBudget, RunEvent, RunStatus
from admires_harness.runtime import (
    DelegationError,
    NodeContext,
    Plan,
    PlanExecutor,
    PlanStep,
    PlanStepEvent,
    StepStatus,
    execute_plan,
)
from admires_harness.runtime.types import RunResult


def _budget(**overrides: object) -> RunBudget:
    base: dict[str, object] = {
        "max_steps": 100,
        "max_tokens": 10_000,
        "max_cost_usd": 10.0,
        "timeout_seconds": 300.0,
    }
    base.update(overrides)
    return RunBudget.model_validate(base)


def _child_result(
    status: RunStatus = RunStatus.SUCCEEDED,
    output: JsonValue = None,
    error: str | None = None,
) -> RunResult:
    now = datetime.now(UTC)
    return RunResult(
        run_id=uuid4(),
        trace_id=uuid4(),
        graph_name="fake-worker",
        graph_version="1",
        status=status,
        state={},
        output=output,
        started_at=now,
        ended_at=now,
        error=error,
    )


async def _emit(event_type: object, payload: Mapping[str, JsonValue]) -> RunEvent:
    return RunEvent(
        run_id=uuid4(),
        sequence=0,
        type=event_type,  # type: ignore[arg-type]
        trace_id=uuid4(),
        payload=dict(payload),
    )


def _context(delegate: object, budget: RunBudget | None = None) -> NodeContext:
    return NodeContext.create(
        run_id=uuid4(),
        trace_id=uuid4(),
        parent_run_id=None,
        node_id="supervisor",
        checkpoint_revision=None,
        state={},
        budget=budget or _budget(),
        dependencies={},
        emit=_emit,  # type: ignore[arg-type]
        delegate=delegate,  # type: ignore[arg-type]
    )


def _plan(*steps: PlanStep) -> Plan:
    return Plan(objective="test plan", steps=steps)


def _step(step_id: str, *depends_on: str) -> PlanStep:
    return PlanStep(id=step_id, objective=f"do {step_id}", agent="worker", depends_on=depends_on)


def _agents() -> dict[str, object]:
    return {"worker": object()}


class FakeDelegate:
    """Gate-controlled delegate: every step starts only when released by the test."""

    def __init__(self, plan: Plan) -> None:
        self.started = {step.id: asyncio.Event() for step in plan.steps}
        self.release = {step.id: asyncio.Event() for step in plan.steps}
        self.calls: list[str] = []
        self.cancelled: list[str] = []
        self.budgets: dict[str, RunBudget] = {}
        self.deps: dict[str, JsonValue] = {}
        self.behaviors: dict[str, str] = {}

    async def __call__(
        self,
        graph: object,
        state: Mapping[str, JsonValue],
        label: str | None,
        budget: RunBudget | None,
        capabilities: Sequence[str],
    ) -> RunResult:
        step_id = str(state["step_id"])
        self.calls.append(step_id)
        self.budgets[step_id] = budget  # type: ignore[assignment]
        self.deps[step_id] = state["dependency_outputs"]
        behavior = self.behaviors.get(step_id, "ok")
        if behavior == "raise":
            raise RuntimeError("no parent budget remains for a child run")
        if behavior == "fail":
            return _child_result(RunStatus.FAILED, error=f"{step_id} exploded")
        self.started[step_id].set()
        try:
            if behavior == "hang":
                await asyncio.Future()
            else:
                await self.release[step_id].wait()
        except asyncio.CancelledError:
            self.cancelled.append(step_id)
            raise
        return _child_result(output={"step": step_id})

    async def wait_started(self, step_id: str) -> None:
        await asyncio.wait_for(self.started[step_id].wait(), timeout=5)


async def test_empty_plan_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Plan(objective="empty", steps=())


async def test_multi_layer_dag_runs_in_stable_order() -> None:
    plan = _plan(
        _step("a"),
        _step("b", "a"),
        _step("c", "a"),
        _step("d", "b", "c"),
    )
    delegate = FakeDelegate(plan)
    executor = PlanExecutor(_context(delegate), plan, _agents())  # type: ignore[arg-type]

    run_task = asyncio.create_task(executor.run())
    await delegate.wait_started("a")
    assert delegate.calls == ["a"]
    delegate.release["a"].set()
    await delegate.wait_started("b")
    await delegate.wait_started("c")
    assert delegate.calls == ["a", "b", "c"]
    assert not delegate.started["d"].is_set()
    delegate.release["b"].set()
    delegate.release["c"].set()
    await delegate.wait_started("d")
    assert delegate.deps["b"] == {"a": {"step": "a"}}
    assert delegate.deps["d"] == {"b": {"step": "b"}, "c": {"step": "c"}}
    delegate.release["d"].set()
    result = await run_task

    assert executor.in_flight == 0
    assert list(result.update["plan_results"]) == ["a", "b", "c", "d"]  # type: ignore[index]
    assert result.update["plan_failures"] == {}
    statuses = result.update["plan_step_statuses"]
    assert all(entry["status"] == "succeeded" for entry in statuses.values())  # type: ignore[index,union-attr]
    assert result.metadata == {
        "planned_steps": 4,
        "succeeded_steps": 4,
        "failed_steps": 0,
        "skipped_steps": 0,
        "cancelled_steps": 0,
        "timed_out_steps": 0,
        "callback_errors": 0,
    }


async def test_same_layer_concurrency_limit_is_never_exceeded() -> None:
    plan = _plan(*(_step(f"s{index}") for index in range(5)))
    started = {step.id: asyncio.Event() for step in plan.steps}
    release = {step.id: asyncio.Event() for step in plan.steps}
    active = 0
    max_active = 0

    async def delegate(
        graph: object,
        state: Mapping[str, JsonValue],
        label: str | None,
        budget: RunBudget | None,
        capabilities: Sequence[str],
    ) -> RunResult:
        nonlocal active, max_active
        step_id = str(state["step_id"])
        active += 1
        max_active = max(max_active, active)
        started[step_id].set()
        await release[step_id].wait()
        active -= 1
        return _child_result(output=step_id)

    executor = PlanExecutor(_context(delegate), plan, _agents(), max_concurrency=2)  # type: ignore[arg-type]
    run_task = asyncio.create_task(executor.run())

    await asyncio.wait_for(started["s0"].wait(), timeout=5)
    await asyncio.wait_for(started["s1"].wait(), timeout=5)
    await asyncio.sleep(0)
    assert not started["s2"].is_set()
    release["s0"].set()
    await asyncio.wait_for(started["s2"].wait(), timeout=5)
    await asyncio.sleep(0)
    assert not started["s3"].is_set()
    for event in release.values():
        event.set()
    result = await run_task

    assert max_active == 2
    assert executor.in_flight == 0
    assert result.metadata["succeeded_steps"] == 5  # type: ignore[typeddict-item]


async def test_failed_dependency_skips_descendants_without_fail_fast() -> None:
    plan = _plan(
        _step("a"),
        _step("b", "a"),
        _step("c", "b"),
        _step("d"),
    )
    delegate = FakeDelegate(plan)
    delegate.behaviors["a"] = "fail"
    delegate.release["d"].set()
    executor = PlanExecutor(_context(delegate), plan, _agents(), fail_fast=False)  # type: ignore[arg-type]

    result = await executor.run()

    assert delegate.calls == ["a", "d"]  # b and c never ran
    assert executor.statuses == {
        "a": StepStatus.FAILED,
        "b": StepStatus.SKIPPED,
        "c": StepStatus.SKIPPED,
        "d": StepStatus.SUCCEEDED,
    }
    assert executor.reasons["a"] == "a exploded"
    assert executor.reasons["b"] == "dependency a failed"
    assert executor.reasons["c"] == "dependency b skipped"
    failures = result.update["plan_failures"]
    assert set(failures) == {"a", "b", "c"}  # type: ignore[arg-type]
    assert failures["b"] == "dependency a failed"  # type: ignore[index]
    assert list(result.update["plan_results"]) == ["d"]  # type: ignore[index]
    assert result.metadata["failed_steps"] == 1  # type: ignore[typeddict-item]
    assert result.metadata["skipped_steps"] == 2  # type: ignore[typeddict-item]
    assert executor.in_flight == 0


async def test_fail_fast_cancels_in_flight_steps_and_skips_rest() -> None:
    plan = _plan(_step("x"), _step("y"), _step("z"))
    delegate = FakeDelegate(plan)
    delegate.behaviors["x"] = "fail"
    delegate.behaviors["y"] = "hang"
    executor = PlanExecutor(
        _context(delegate),
        plan,
        _agents(),
        max_concurrency=2,
        fail_fast=True,  # type: ignore[arg-type]
    )

    with pytest.raises(DelegationError, match="delegated step failed: x: x exploded"):
        await executor.run()

    assert delegate.cancelled == ["y"]
    assert executor.statuses == {
        "x": StepStatus.FAILED,
        "y": StepStatus.CANCELLED,
        "z": StepStatus.SKIPPED,
    }
    assert executor.reasons["y"] == "fail-fast abort"
    assert executor.reasons["z"] == "fail-fast abort"
    assert executor.in_flight == 0
    leftover = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    assert leftover == []


async def test_external_cancellation_converges_running_and_pending_steps() -> None:
    plan = _plan(_step("a"), _step("b", "a"))
    delegate = FakeDelegate(plan)
    delegate.behaviors["a"] = "hang"
    executor = PlanExecutor(_context(delegate), plan, _agents())  # type: ignore[arg-type]

    run_task = asyncio.create_task(executor.run())
    await delegate.wait_started("a")
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    assert delegate.cancelled == ["a"]
    assert executor.statuses == {"a": StepStatus.CANCELLED, "b": StepStatus.CANCELLED}
    assert executor.reasons == {"a": "executor cancelled", "b": "executor cancelled"}
    assert executor.in_flight == 0
    leftover = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    assert leftover == []


async def test_step_timeout_marks_step_timed_out() -> None:
    plan = _plan(_step("slow"), _step("fast"))
    delegate = FakeDelegate(plan)
    delegate.behaviors["slow"] = "hang"
    delegate.release["fast"].set()
    executor = PlanExecutor(
        _context(delegate),
        plan,
        _agents(),  # type: ignore[arg-type]
        fail_fast=False,
        step_timeout=0.05,
    )

    result = await executor.run()

    assert delegate.cancelled == ["slow"]
    assert executor.statuses["slow"] is StepStatus.TIMED_OUT
    assert executor.statuses["fast"] is StepStatus.SUCCEEDED
    assert "timed out" in executor.reasons["slow"]
    assert result.update["plan_failures"]["slow"] == executor.reasons["slow"]  # type: ignore[index]
    assert result.metadata["timed_out_steps"] == 1  # type: ignore[typeddict-item]
    assert executor.in_flight == 0


async def test_parent_budget_too_small_rejects_before_any_delegation() -> None:
    plan = _plan(_step("a"), _step("b"), _step("c"))
    delegate = FakeDelegate(plan)
    with pytest.raises(DelegationError, match="parent budget is too small"):
        PlanExecutor(_context(delegate, _budget(max_steps=3)), plan, _agents())  # type: ignore[arg-type]
    with pytest.raises(DelegationError, match="parent budget is too small"):
        PlanExecutor(_context(delegate, _budget(max_tokens=2)), plan, _agents())  # type: ignore[arg-type]
    assert delegate.calls == []


async def test_budget_reservation_failure_is_a_traceable_step_failure() -> None:
    plan = _plan(_step("a"), _step("b"))
    delegate = FakeDelegate(plan)
    delegate.behaviors["a"] = "raise"
    delegate.release["b"].set()
    executor = PlanExecutor(_context(delegate), plan, _agents(), fail_fast=False)  # type: ignore[arg-type]

    result = await executor.run()

    assert executor.statuses["a"] is StepStatus.FAILED
    assert executor.reasons["a"] == "no parent budget remains for a child run"
    assert executor.statuses["b"] is StepStatus.SUCCEEDED
    assert result.metadata["failed_steps"] == 1  # type: ignore[typeddict-item]
    assert executor.in_flight == 0


async def test_per_step_budget_split_is_deterministic_and_within_parent() -> None:
    plan = _plan(_step("a"), _step("b"), _step("c"))
    delegate = FakeDelegate(plan)
    for event in delegate.release.values():
        event.set()
    budget = _budget(max_steps=31, max_tokens=900, max_cost_usd=9.0, timeout_seconds=90.0)
    executor = PlanExecutor(_context(delegate, budget), plan, _agents())  # type: ignore[arg-type]

    await executor.run()

    per_step = executor.per_step_budget
    assert per_step.max_steps == 10
    assert per_step.max_tokens == 300
    assert per_step.max_cost_usd == pytest.approx(3.0)
    assert per_step.timeout_seconds == pytest.approx(30.0)
    assert list(delegate.budgets.values()) == [per_step] * 3
    assert 3 * per_step.max_steps <= budget.max_steps - 1
    assert 3 * per_step.max_tokens <= budget.max_tokens
    assert 3 * per_step.max_cost_usd <= budget.max_cost_usd


async def test_callback_exceptions_do_not_corrupt_the_schedule() -> None:
    plan = _plan(_step("a"), _step("b", "a"))
    delegate = FakeDelegate(plan)
    for event in delegate.release.values():
        event.set()
    events: list[tuple[str, str]] = []

    async def on_step_event(event: PlanStepEvent) -> None:
        events.append((event.step_id, event.status.value))
        raise RuntimeError("observer broke")

    executor = PlanExecutor(
        _context(delegate),
        plan,
        _agents(),
        on_step_event=on_step_event,  # type: ignore[arg-type]
    )
    result = await executor.run()

    assert events == [
        ("a", "running"),
        ("a", "succeeded"),
        ("b", "running"),
        ("b", "succeeded"),
    ]
    assert result.metadata["callback_errors"] == 4  # type: ignore[typeddict-item]
    assert result.metadata["succeeded_steps"] == 2  # type: ignore[typeddict-item]
    assert executor.in_flight == 0


async def test_execute_plan_wrapper_delegates_to_executor() -> None:
    plan = _plan(_step("only"))
    delegate = FakeDelegate(plan)
    delegate.release["only"].set()

    result = await execute_plan(_context(delegate), plan, _agents())  # type: ignore[arg-type]

    assert result.update["plan_results"] == {"only": {"step": "only"}}
    assert result.update["plan_step_statuses"]["only"]["status"] == "succeeded"  # type: ignore[index]
    with pytest.raises(ValueError, match="positive"):
        await execute_plan(_context(delegate), plan, _agents(), max_concurrency=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        await execute_plan(_context(delegate), plan, _agents(), step_timeout=0)  # type: ignore[arg-type]
