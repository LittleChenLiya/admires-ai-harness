"""Plan validation and dependency-aware multi-agent delegation."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import Mapping

from pydantic import Field, model_validator

from admires_harness.contracts import JsonValue, RunBudget, RunStatus, StrictModel
from admires_harness.runtime.errors import DelegationError
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
) -> NodeResult:
    """Execute ready plan steps concurrently and join their child-run results.

    Each child receives its explicit inputs plus the outputs of declared dependencies.
    The returned update is JSON-compatible and can be routed by ordinary graph edges.
    """

    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    missing_agents = sorted({step.agent for step in plan.steps} - set(agents))
    if missing_agents:
        raise DelegationError(f"unknown plan agents: {', '.join(missing_agents)}")

    steps = {step.id: step for step in plan.steps}
    pending = set(steps)
    completed: dict[str, JsonValue] = {}
    failures: dict[str, str] = {}
    semaphore = asyncio.Semaphore(max_concurrency)
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
        async with semaphore:
            result = await context.delegate(
                agents[step.agent],
                child_state,
                label=step.id,
                budget=per_child_budget,
                capabilities=step.capabilities,
            )
        if result.status != RunStatus.SUCCEEDED:
            return step.id, None, result.error or result.status.value
        output = result.output if result.output is not None else result.state
        return step.id, output, None

    while pending:
        ready = [
            steps[step_id]
            for step_id in plan.topological_order()
            if step_id in pending
            and all(dependency in completed for dependency in steps[step_id].depends_on)
        ]
        if not ready:
            skipped = [
                step_id
                for step_id in pending
                if any(dependency in failures for dependency in steps[step_id].depends_on)
            ]
            if skipped:
                for step_id in skipped:
                    pending.remove(step_id)
                    failures[step_id] = "dependency failed"
                continue
            blocked = ", ".join(sorted(pending))
            raise DelegationError(f"plan cannot make progress; blocked steps: {blocked}")
        batch = await asyncio.gather(*(run_step(step) for step in ready))
        for step_id, output, error in batch:
            pending.remove(step_id)
            if error is None:
                completed[step_id] = output
            else:
                failures[step_id] = error
        if failures and fail_fast:
            details = ", ".join(f"{key}: {value}" for key, value in failures.items())
            raise DelegationError(f"delegated step failed: {details}")
        if failures:
            skipped = [
                step_id
                for step_id in pending
                if any(dependency in failures for dependency in steps[step_id].depends_on)
            ]
            for step_id in skipped:
                pending.remove(step_id)
                failures[step_id] = "dependency failed"

    failure_payload: dict[str, JsonValue] = dict(failures)
    update: dict[str, JsonValue] = {
        "plan_objective": plan.objective,
        "plan_results": completed,
        "plan_failures": failure_payload,
    }
    return NodeResult(
        update=update,
        output=completed,
        metadata={"planned_steps": len(plan.steps), "failed_steps": len(failures)},
    )
