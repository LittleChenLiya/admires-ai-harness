"""Plan validation and dependency-aware multi-agent delegation."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import Mapping

from pydantic import Field, model_validator

from admires_harness.contracts import EventType, JsonValue, RunBudget, RunStatus, StrictModel
from admires_harness.runtime.errors import BudgetExceededError, DelegationError
from admires_harness.runtime.graph import CompiledGraph
from admires_harness.runtime.types import NodeContext, NodeResult


class PlanStep(StrictModel):
    """One independently executable unit in a planner-produced DAG."""

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    objective: str = Field(min_length=1)
    agent: str = Field(min_length=1)
    depends_on: tuple[str, ...] = ()
    inputs: dict[str, JsonValue] = Field(default_factory=dict)
    capabilities: tuple[str, ...] = ()


class Plan(StrictModel):
    """A validated acyclic plan safe for dependency-aware scheduling."""

    objective: str = Field(min_length=1)
    steps: tuple[PlanStep, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dag(self) -> Plan:
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("plan step ids must be unique")
        known = set(ids)
        for step in self.steps:
            unknown = set(step.depends_on) - known
            if unknown:
                raise ValueError(f"step {step.id!r} has unknown dependencies: {sorted(unknown)}")
            if step.id in step.depends_on:
                raise ValueError(f"step {step.id!r} cannot depend on itself")
        if len(self.topological_order()) != len(self.steps):
            raise ValueError("plan dependencies contain a cycle")
        return self

    def topological_order(self) -> tuple[str, ...]:
        """Return a stable order, preserving author order among ready steps."""

        positions = {step.id: index for index, step in enumerate(self.steps)}
        indegree = {step.id: len(step.depends_on) for step in self.steps}
        downstream: dict[str, list[str]] = defaultdict(list)
        for step in self.steps:
            for dependency in step.depends_on:
                downstream[dependency].append(step.id)
        ready = deque(step.id for step in self.steps if indegree[step.id] == 0)
        ordered: list[str] = []
        while ready:
            current = ready.popleft()
            ordered.append(current)
            newly_ready: list[str] = []
            for child in downstream[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    newly_ready.append(child)
            ready.extend(sorted(newly_ready, key=positions.__getitem__))
        return tuple(ordered)


async def execute_plan(
    context: NodeContext,
    plan: Plan,
    agents: Mapping[str, CompiledGraph],
    *,
    max_concurrency: int = 4,
    fail_fast: bool = True,
    step_timeout_seconds: float | None = None,
) -> NodeResult:
    """Execute ready plan steps concurrently and join their child-run results.

    Each child receives its explicit inputs plus the outputs of declared
    dependencies, and the scheduler enforces the concurrency cap, per-step
    timeout, and parent budget without races. Ready steps start in stable
    topological order. A step whose dependency failed or was skipped never runs
    and is recorded in `plan_skipped` with a traceable reason. On fail-fast
    or external cancellation every started step is cancelled and awaited before
    the error propagates, so no asyncio task is left pending. The returned
    update is JSON-compatible and can be routed by ordinary graph edges.
    """

    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    if step_timeout_seconds is not None and step_timeout_seconds <= 0:
        raise ValueError("step_timeout_seconds must be positive")
    missing_agents = sorted({step.agent for step in plan.steps} - set(agents))
    if missing_agents:
        raise DelegationError(f"unknown plan agents: {', '.join(missing_agents)}")

    steps = {step.id: step for step in plan.steps}
    order = plan.topological_order()
    position = {step_id: index for index, step_id in enumerate(order)}
    step_count = len(plan.steps)
    child_steps = context.budget.max_steps - 1
    if child_steps < step_count or context.budget.max_tokens < step_count:
        raise DelegationError("parent budget is too small for every planned child run")
    per_child_budget = RunBudget(
        max_steps=max(1, child_steps // step_count),
        max_tokens=max(1, context.budget.max_tokens // step_count),
        max_cost_usd=context.budget.max_cost_usd / step_count,
        timeout_seconds=context.budget.timeout_seconds / step_count,
        max_repeated_actions=context.budget.max_repeated_actions,
    )
    step_timeout = per_child_budget.timeout_seconds
    if step_timeout_seconds is not None:
        step_timeout = min(step_timeout, step_timeout_seconds)

    event_callback_errors = 0

    async def emit_plan_event(event: str, **payload: JsonValue) -> None:
        nonlocal event_callback_errors
        try:
            await context.emit(EventType.LOG, {"plan_event": event, **payload})
        except Exception:
            event_callback_errors += 1

    async def run_step(
        step: PlanStep,
    ) -> tuple[str, JsonValue, str | None]:
        dependency_outputs = {key: completed[key] for key in step.depends_on}
        child_state: dict[str, JsonValue] = {
            "objective": step.objective,
            "step_id": step.id,
            "inputs": step.inputs,
            "dependency_outputs": dependency_outputs,
        }
        try:
            async with asyncio.timeout(step_timeout):
                result = await context.delegate(
                    agents[step.agent],
                    child_state,
                    label=step.id,
                    budget=per_child_budget,
                    capabilities=step.capabilities,
                )
        except TimeoutError:
            return step.id, None, f"timed out after {step_timeout:g} seconds"
        except BudgetExceededError as exc:
            return step.id, None, f"budget exceeded: {exc}"
        except Exception as exc:
            return step.id, None, f"{type(exc).__name__}: {exc}"
        if result.status != RunStatus.SUCCEEDED:
            return step.id, None, result.error or result.status.value
        output = result.output if result.output is not None else result.state
        return step.id, output, None

    pending: set[str] = set(order)
    running: dict[asyncio.Task[tuple[str, JsonValue, str | None]], str] = {}
    completed: dict[str, JsonValue] = {}
    failures: dict[str, str] = {}
    skipped: dict[str, str] = {}
    first_failure: str | None = None

    async def cancel_running(reason: str) -> None:
        for task in running:
            task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)
        for step_id in running.values():
            skipped[step_id] = reason
            await emit_plan_event("plan_step_skipped", step_id=step_id, reason=reason)
        running.clear()

    await emit_plan_event(
        "plan_started",
        objective=plan.objective,
        steps=step_count,
        max_concurrency=max_concurrency,
    )
    try:
        while pending or running:
            for step_id in order:
                if step_id not in pending:
                    continue
                blocking = next(
                    (
                        dependency
                        for dependency in steps[step_id].depends_on
                        if dependency in failures or dependency in skipped
                    ),
                    None,
                )
                if blocking is not None:
                    pending.discard(step_id)
                    cause = "failed" if blocking in failures else "skipped"
                    reason = f"dependency '{blocking}' {cause}"
                    skipped[step_id] = reason
                    await emit_plan_event("plan_step_skipped", step_id=step_id, reason=reason)
            capacity = max_concurrency - len(running)
            if capacity > 0 and first_failure is None:
                for step_id in order:
                    if capacity == 0:
                        break
                    if step_id in pending and all(
                        dependency in completed for dependency in steps[step_id].depends_on
                    ):
                        pending.discard(step_id)
                        task = asyncio.ensure_future(run_step(steps[step_id]))
                        running[task] = step_id
                        capacity -= 1
                        await emit_plan_event(
                            "plan_step_started", step_id=step_id, agent=steps[step_id].agent
                        )
            if not running:
                if pending:
                    blocked = ", ".join(sorted(pending))
                    raise DelegationError(f"plan cannot make progress; blocked steps: {blocked}")
                break
            done, _ = await asyncio.wait(set(running), return_when=asyncio.FIRST_COMPLETED)
            for task in sorted(done, key=lambda item: position[running[item]]):
                step_id = running.pop(task)
                _, output, error = task.result()
                if error is None:
                    completed[step_id] = output
                    await emit_plan_event("plan_step_completed", step_id=step_id)
                else:
                    failures[step_id] = error
                    await emit_plan_event("plan_step_failed", step_id=step_id, error=error)
                    if fail_fast and first_failure is None:
                        first_failure = step_id
            if first_failure is not None:
                reason = f"fail-fast: cancelled after step '{first_failure}' failed"
                await cancel_running(reason)
                for step_id in order:
                    if step_id in pending:
                        pending.discard(step_id)
                        skipped[step_id] = reason
                        await emit_plan_event("plan_step_skipped", step_id=step_id, reason=reason)
                details = ", ".join(f"{key}: {value}" for key, value in failures.items())
                raise DelegationError(f"delegated step failed: {details}")
    except asyncio.CancelledError:
        await cancel_running("cancelled: external cancellation")
        raise

    await emit_plan_event(
        "plan_completed",
        completed=len(completed),
        failed=len(failures),
        skipped=len(skipped),
    )

    failure_payload: dict[str, JsonValue] = dict(failures)
    skipped_payload: dict[str, JsonValue] = dict(skipped)
    update: dict[str, JsonValue] = {
        "plan_objective": plan.objective,
        "plan_results": completed,
        "plan_failures": failure_payload,
        "plan_skipped": skipped_payload,
    }
    return NodeResult(
        update=update,
        output=completed,
        metadata={
            "planned_steps": step_count,
            "completed_steps": len(completed),
            "failed_steps": len(failures),
            "skipped_steps": len(skipped),
            "event_callback_errors": event_callback_errors,
        },
    )
