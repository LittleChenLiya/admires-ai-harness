"""Plan validation and dependency-aware multi-agent delegation."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Mapping
from enum import StrEnum

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


class StepStatus(StrEnum):
    """Lifecycle states reported for every planned step."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


_FAILED_DEPENDENCY_STATUSES = frozenset(
    {StepStatus.FAILED, StepStatus.SKIPPED, StepStatus.CANCELLED, StepStatus.TIMED_OUT}
)
_CHILD_STATUS_MAP: dict[RunStatus, StepStatus] = {
    RunStatus.CANCELLED: StepStatus.CANCELLED,
    RunStatus.TIMED_OUT: StepStatus.TIMED_OUT,
}


class PlanStepEvent(StrictModel):
    """One lifecycle transition delivered to step-event callbacks."""

    step_id: str
    status: StepStatus
    reason: str | None = None


type StepEventCallback = Callable[[PlanStepEvent], Awaitable[None] | None]
"""Sync or async hook invoked for every step transition; exceptions are isolated."""

type _StepOutcome = tuple[str, JsonValue, StepStatus | None, str | None]


class PlanExecutor:
    """Dependency-aware scheduler that executes a validated :class:`Plan`.

    Ready steps always launch in stable topological order and at most
    'max_concurrency' steps run at once; slot accounting happens without an
    intervening 'await', so the limit cannot be raced. Every step receives a
    fixed pre-computed slice of the parent budget, which keeps the aggregate of
    all child budgets within the parent budget regardless of completion order.
    Steps whose dependencies did not succeed are skipped with a traceable
    reason, and started steps are always cancelled and awaited before the
    executor returns or re-raises on fail-fast or external cancellation.
    """

    def __init__(
        self,
        context: NodeContext,
        plan: Plan,
        agents: Mapping[str, CompiledGraph],
        *,
        max_concurrency: int = 4,
        fail_fast: bool = True,
        step_timeout: float | None = None,
        on_step_event: StepEventCallback | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if step_timeout is not None and step_timeout <= 0:
            raise ValueError("step_timeout must be positive")
        missing_agents = sorted({step.agent for step in plan.steps} - set(agents))
        if missing_agents:
            raise DelegationError(f"unknown plan agents: {', '.join(missing_agents)}")
        step_count = len(plan.steps)
        child_steps = context.budget.max_steps - 1
        if child_steps < step_count or context.budget.max_tokens < step_count:
            raise DelegationError("parent budget is too small for every planned child run")
        self._context = context
        self._plan = plan
        self._agents = agents
        self._max_concurrency = max_concurrency
        self._fail_fast = fail_fast
        self._on_step_event = on_step_event
        self._per_step_budget = RunBudget(
            max_steps=max(1, child_steps // step_count),
            max_tokens=max(1, context.budget.max_tokens // step_count),
            max_cost_usd=context.budget.max_cost_usd / step_count,
            timeout_seconds=context.budget.timeout_seconds / step_count,
            max_repeated_actions=context.budget.max_repeated_actions,
        )
        self._step_timeout = (
            step_timeout if step_timeout is not None else self._per_step_budget.timeout_seconds
        )
        self._steps = {step.id: step for step in plan.steps}
        self._order = plan.topological_order()
        self._statuses = {step_id: StepStatus.PENDING for step_id in self._order}
        self._reasons: dict[str, str] = {}
        self._outputs: dict[str, JsonValue] = {}
        self._callback_errors = 0
        self._tasks: dict[asyncio.Task[_StepOutcome], str] = {}

    @property
    def per_step_budget(self) -> RunBudget:
        """The identical budget slice issued to every child run."""

        return self._per_step_budget

    @property
    def in_flight(self) -> int:
        """Number of step tasks currently running; always zero after 'run()' exits."""

        return len(self._tasks)

    @property
    def statuses(self) -> dict[str, StepStatus]:
        return dict(self._statuses)

    @property
    def reasons(self) -> dict[str, str]:
        return dict(self._reasons)

    async def run(self) -> NodeResult:
        """Execute the plan and join child outputs into a single node result."""

        try:
            while True:
                await self._skip_blocked()
                await self._launch_ready()
                if not self._tasks:
                    break
                done, _ = await asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    step_id = self._tasks.pop(task)
                    await self._record(step_id, task.result())
                if self._fail_fast and self._has_failures():
                    await self._cancel_running("fail-fast abort")
                    for step_id in self._order:
                        if self._statuses[step_id] is StepStatus.PENDING:
                            await self._mark(step_id, StepStatus.SKIPPED, "fail-fast abort")
                    break
        except asyncio.CancelledError:
            await self._cancel_running("executor cancelled")
            for step_id in self._order:
                if self._statuses[step_id] is StepStatus.PENDING:
                    await self._mark(step_id, StepStatus.CANCELLED, "executor cancelled")
            raise
        if self._fail_fast and self._has_failures():
            details = ", ".join(
                f"{step_id}: {self._reasons[step_id]}"
                for step_id in self._order
                if step_id in self._reasons
            )
            raise DelegationError(f"delegated step failed: {details}")
        return self._build_result()

    async def _run_step(self, step: PlanStep) -> _StepOutcome:
        dependency_outputs = {key: self._outputs[key] for key in step.depends_on}
        child_state: dict[str, JsonValue] = {
            "objective": step.objective,
            "step_id": step.id,
            "inputs": step.inputs,
            "dependency_outputs": dependency_outputs,
        }
        try:
            async with asyncio.timeout(self._step_timeout):
                result = await self._context.delegate(
                    self._agents[step.agent],
                    child_state,
                    label=step.id,
                    budget=self._per_step_budget,
                    capabilities=step.capabilities,
                )
        except TimeoutError:
            reason = f"step timed out after {self._step_timeout:g} seconds"
            return step.id, None, StepStatus.TIMED_OUT, reason
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # budget reservation and delegate validation failures
            return step.id, None, StepStatus.FAILED, str(exc) or type(exc).__name__
        if result.status is not RunStatus.SUCCEEDED:
            status = _CHILD_STATUS_MAP.get(result.status, StepStatus.FAILED)
            return step.id, None, status, result.error or result.status.value
        output = result.output if result.output is not None else result.state
        return step.id, output, None, None

    async def _launch_ready(self) -> None:
        for step_id in self._order:
            if len(self._tasks) >= self._max_concurrency:
                break
            if self._statuses[step_id] is not StepStatus.PENDING:
                continue
            step = self._steps[step_id]
            if any(
                self._statuses[dependency] is not StepStatus.SUCCEEDED
                for dependency in step.depends_on
            ):
                continue
            await self._mark(step_id, StepStatus.RUNNING, None)
            task = asyncio.create_task(self._run_step(step), name=f"plan-step-{step_id}")
            self._tasks[task] = step_id

    async def _skip_blocked(self) -> None:
        moved = True
        while moved:
            moved = False
            for step_id in self._order:
                if self._statuses[step_id] is not StepStatus.PENDING:
                    continue
                failed_dependency = next(
                    (
                        dependency
                        for dependency in self._steps[step_id].depends_on
                        if self._statuses[dependency] in _FAILED_DEPENDENCY_STATUSES
                    ),
                    None,
                )
                if failed_dependency is not None:
                    dependency_status = self._statuses[failed_dependency]
                    await self._mark(
                        step_id,
                        StepStatus.SKIPPED,
                        f"dependency {failed_dependency} {dependency_status.value}",
                    )
                    moved = True

    async def _record(self, step_id: str, outcome: _StepOutcome) -> None:
        _, output, failure_status, reason = outcome
        if failure_status is None:
            self._outputs[step_id] = output
            await self._mark(step_id, StepStatus.SUCCEEDED, None)
        else:
            await self._mark(step_id, failure_status, reason)

    async def _cancel_running(self, reason: str) -> None:
        for task in self._tasks:
            task.cancel()
        pending = [task for task in self._tasks if not task.done()]
        while pending:
            # Re-cancellation during cleanup must not leak half-joined tasks.
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*pending, return_exceptions=True)
            pending = [task for task in self._tasks if not task.done()]
        for task, step_id in list(self._tasks.items()):
            del self._tasks[task]
            if task.cancelled():
                await self._mark(step_id, StepStatus.CANCELLED, reason)
            else:
                await self._record(step_id, task.result())

    async def _mark(self, step_id: str, status: StepStatus, reason: str | None) -> None:
        self._statuses[step_id] = status
        if reason is not None:
            self._reasons[step_id] = reason
        if self._on_step_event is None:
            return
        event = PlanStepEvent(step_id=step_id, status=status, reason=reason)
        try:
            callback_result = self._on_step_event(event)
            if inspect.isawaitable(callback_result):
                await callback_result
        except Exception:  # a broken observer must not corrupt the schedule
            self._callback_errors += 1

    def _has_failures(self) -> bool:
        return any(
            self._statuses[step_id] in {StepStatus.FAILED, StepStatus.TIMED_OUT}
            for step_id in self._order
        )

    def _build_result(self) -> NodeResult:
        completed = {
            step_id: self._outputs[step_id] for step_id in self._order if step_id in self._outputs
        }
        failures: dict[str, JsonValue] = {
            step_id: self._reasons.get(step_id, self._statuses[step_id].value)
            for step_id in self._order
            if self._statuses[step_id] in _FAILED_DEPENDENCY_STATUSES
        }
        step_statuses: dict[str, JsonValue] = {
            step_id: {
                "status": self._statuses[step_id].value,
                "reason": self._reasons.get(step_id),
            }
            for step_id in self._order
        }
        update: dict[str, JsonValue] = {
            "plan_objective": self._plan.objective,
            "plan_results": completed,
            "plan_failures": failures,
            "plan_step_statuses": step_statuses,
        }
        counts = {
            status: sum(1 for step_id in self._order if self._statuses[step_id] is status)
            for status in StepStatus
        }
        return NodeResult(
            update=update,
            output=completed,
            metadata={
                "planned_steps": len(self._plan.steps),
                "succeeded_steps": counts[StepStatus.SUCCEEDED],
                "failed_steps": counts[StepStatus.FAILED],
                "skipped_steps": counts[StepStatus.SKIPPED],
                "cancelled_steps": counts[StepStatus.CANCELLED],
                "timed_out_steps": counts[StepStatus.TIMED_OUT],
                "callback_errors": self._callback_errors,
            },
        )


async def execute_plan(
    context: NodeContext,
    plan: Plan,
    agents: Mapping[str, CompiledGraph],
    *,
    max_concurrency: int = 4,
    fail_fast: bool = True,
    step_timeout: float | None = None,
    on_step_event: StepEventCallback | None = None,
) -> NodeResult:
    """Execute ready plan steps concurrently and join their child-run results.

    Each child receives its explicit inputs plus the outputs of declared
    dependencies. Steps whose dependencies failed, timed out, or were cancelled
    never run and are reported in 'plan_failures'/'plan_step_statuses' with
    the reason. With 'fail_fast' the first failure cancels in-flight steps and
    raises :class:`DelegationError` once every started task has converged.
    The returned update is JSON-compatible and can be routed by ordinary edges.
    """

    executor = PlanExecutor(
        context,
        plan,
        agents,
        max_concurrency=max_concurrency,
        fail_fast=fail_fast,
        step_timeout=step_timeout,
        on_step_event=on_step_event,
    )
    return await executor.run()
