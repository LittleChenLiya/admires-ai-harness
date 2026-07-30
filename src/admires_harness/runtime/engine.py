"""Event-sourced execution engine for compiled agent graphs."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from time import monotonic
from typing import Any
from uuid import UUID, uuid4

from admires_harness.contracts import (
    Checkpoint,
    EventType,
    JsonValue,
    ModelUsage,
    RunBudget,
    RunEvent,
    RunStatus,
    utc_now,
)
from admires_harness.ports import CheckpointStore, EventStore
from admires_harness.runtime.errors import (
    CheckpointConflictError,
    IncompatibleCheckpointError,
    RunAlreadyExistsError,
    RunLeaseActiveError,
    RunNotFoundError,
    TerminalRunError,
)
from admires_harness.runtime.graph import CompiledGraph
from admires_harness.runtime.stream import EventBroker
from admires_harness.runtime.types import NodeContext, NodeResult, ReplaySnapshot, RunResult


class _BudgetExceeded(Exception):
    pass


@dataclass(slots=True)
class _Metrics:
    steps: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, result: NodeResult) -> None:
        self.steps += 1
        self.model_calls += result.model_calls
        self.tool_calls += result.tool_calls
        self.input_tokens += result.usage.input_tokens
        self.output_tokens += result.usage.output_tokens
        self.cost_usd += result.usage.cost_usd


@dataclass(slots=True)
class _Session:
    runtime: GraphRuntime
    graph: CompiledGraph
    run_id: UUID
    trace_id: UUID
    parent_run_id: UUID | None
    state: dict[str, JsonValue]
    budget: RunBudget
    next_node: str | None
    started_at: datetime
    sequence: int = 0
    checkpoint_revision: int | None = None
    metrics: _Metrics = field(default_factory=_Metrics)
    last_action: str | None = None
    repeated_actions: int = 0
    current_node: str | None = None
    output: JsonValue = None
    elapsed_active_seconds: float = 0.0
    reserved_steps: int = 0
    reserved_tokens: int = 0
    reserved_cost_usd: float = 0.0
    _active_started_monotonic: float | None = None
    _event_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _budget_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def emit(
        self,
        event_type: EventType,
        payload: Mapping[str, JsonValue] | None = None,
        *,
        node_id: str | None = None,
    ) -> RunEvent:
        async with self._event_lock:
            event = RunEvent(
                run_id=self.run_id,
                sequence=self.sequence,
                type=event_type,
                parent_run_id=self.parent_run_id,
                node_id=node_id if node_id is not None else self.current_node,
                trace_id=self.trace_id,
                payload=dict(payload or {}),
            )
            await self.runtime.event_store.append(event)
            self.sequence += 1
            await self.runtime.broker.publish(event)
            return event


class GraphRuntime:
    """Executes graphs with durable events, checkpoints, budgets, and child runs.

    The runtime contains no model, tool, memory, or sandbox implementation. Those
    capabilities are injected through ``dependencies`` and consumed by nodes. This
    keeps the orchestration kernel testable and adapters replaceable.
    """

    def __init__(
        self,
        *,
        event_store: EventStore,
        checkpoint_store: CheckpointStore,
        dependencies: Mapping[str, Any] | None = None,
        broker: EventBroker | None = None,
    ) -> None:
        self.event_store = event_store
        self.checkpoint_store = checkpoint_store
        self.dependencies = dict(dependencies or {})
        self.broker = broker or EventBroker()

    async def run(
        self,
        graph: CompiledGraph,
        initial_state: Mapping[str, JsonValue] | None = None,
        *,
        budget: RunBudget | None = None,
        run_id: UUID | None = None,
        trace_id: UUID | None = None,
        parent_run_id: UUID | None = None,
    ) -> RunResult:
        """Start a fresh graph execution."""

        actual_run_id = run_id or uuid4()
        if await self.event_store.list(actual_run_id) or await self.checkpoint_store.latest(
            actual_run_id
        ):
            raise RunAlreadyExistsError(f"run already exists: {actual_run_id}")
        session = _Session(
            runtime=self,
            graph=graph,
            run_id=actual_run_id,
            trace_id=trace_id or uuid4(),
            parent_run_id=parent_run_id,
            state=dict(initial_state or {}),
            budget=budget or RunBudget(),
            next_node=graph.entry_point,
            started_at=utc_now(),
        )
        await session.emit(
            EventType.RUN_CREATED,
            {
                "graph_name": graph.name,
                "graph_version": graph.version,
                # Persisting the seed makes projections and replay self-contained;
                # mutable in-process request state is never required as a second truth.
                "initial_state": session.state,
                "budget": session.budget.model_dump(mode="json"),
            },
            node_id=None,
        )
        await session.emit(EventType.RUN_STARTED, node_id=None)
        return await self._drive(session)

    async def resume(
        self,
        graph: CompiledGraph,
        run_id: UUID,
        *,
        state_update: Mapping[str, JsonValue] | None = None,
        budget: RunBudget | None = None,
    ) -> RunResult:
        """Resume a paused run from its latest optimistic checkpoint."""

        checkpoint = await self.checkpoint_store.latest(run_id)
        if checkpoint is None:
            raise RunNotFoundError(f"checkpoint not found for run: {run_id}")
        if checkpoint.status.terminal:
            raise TerminalRunError(f"run {run_id} is already {checkpoint.status}")
        if (checkpoint.graph_name, checkpoint.graph_version) != (graph.name, graph.version):
            raise IncompatibleCheckpointError(
                f"checkpoint uses {checkpoint.graph_name}@{checkpoint.graph_version}, "
                f"but runtime received {graph.name}@{graph.version}"
            )
        events = await self.event_store.list(run_id)
        if not events:
            raise RunNotFoundError(f"event history not found for run: {run_id}")
        actual_budget = budget or self._budget_from_events(events)
        lease_ttl = max(0.0, actual_budget.timeout_seconds - checkpoint.elapsed_active_seconds)
        if (
            checkpoint.status == RunStatus.RUNNING
            and (utc_now() - checkpoint.created_at).total_seconds() < lease_ttl
        ):
            raise RunLeaseActiveError(f"run {run_id} already has an active execution lease")
        state = dict(checkpoint.state)
        state.update(state_update or {})
        session = _Session(
            runtime=self,
            graph=graph,
            run_id=run_id,
            trace_id=events[0].trace_id,
            parent_run_id=events[0].parent_run_id,
            state=state,
            budget=actual_budget,
            next_node=checkpoint.next_node,
            started_at=events[0].timestamp,
            sequence=events[-1].sequence + 1,
            checkpoint_revision=checkpoint.revision,
            metrics=self._metrics_from_events(events),
            elapsed_active_seconds=checkpoint.elapsed_active_seconds,
        )
        self._restore_action_guard(session, events)
        # Acquire a durable compare-and-swap lease before invoking any node. A
        # concurrent resume that read the same revision loses here, before side effects.
        try:
            await self._save_checkpoint(session, RunStatus.RUNNING)
        except CheckpointConflictError as exc:
            # Both contenders can observe the same resumable revision before
            # either compare-and-swap completes. Expose that race as the same
            # stable lease error as an already-visible RUNNING checkpoint; no
            # node has executed at this point.
            if session.checkpoint_revision == checkpoint.revision:
                raise RunLeaseActiveError(
                    f"run {run_id} already has an active execution lease"
                ) from exc
            # The lease checkpoint was committed, so a later event-store
            # conflict is an integrity error rather than a lost lease race.
            raise
        await session.emit(
            EventType.RUN_STATUS,
            {"status": RunStatus.RUNNING.value, "resumed_revision": checkpoint.revision},
            node_id=None,
        )
        return await self._drive(session)

    async def replay(self, run_id: UUID) -> ReplaySnapshot:
        """Reconstruct externally visible state without executing any node."""

        events = await self.event_store.list(run_id)
        checkpoint = await self.checkpoint_store.latest(run_id)
        if not events and checkpoint is None:
            raise RunNotFoundError(f"run not found: {run_id}")
        # Terminal visibility belongs to the lifecycle event stream. A terminal
        # checkpoint can become durable just before ``run.completed``/``run.status``
        # is appended; replay must not publish that terminal state prematurely.
        status = (
            checkpoint.status
            if checkpoint is not None and not checkpoint.status.terminal
            else RunStatus.CREATED
        )
        for event in events:
            if event.type == EventType.RUN_STARTED:
                status = RunStatus.RUNNING
            elif event.type == EventType.RUN_COMPLETED:
                status = RunStatus.SUCCEEDED
            elif event.type == EventType.RUN_FAILED:
                status = RunStatus.FAILED
            raw_status = event.payload.get("status")
            if event.type in {
                EventType.RUN_STATUS,
                EventType.RUN_COMPLETED,
                EventType.RUN_FAILED,
            } and isinstance(raw_status, str):
                with suppress(ValueError):
                    status = RunStatus(raw_status)
        return ReplaySnapshot(
            run_id=run_id,
            status=status,
            state=dict(checkpoint.state) if checkpoint is not None else {},
            next_node=checkpoint.next_node if checkpoint is not None else None,
            checkpoint_revision=checkpoint.revision if checkpoint is not None else None,
            events=tuple(events),
        )

    async def events(
        self, run_id: UUID, *, after: int = -1, follow: bool = False
    ) -> AsyncIterator[RunEvent]:
        """Yield persisted events and optionally follow live events until terminal."""

        if not follow:
            for event in await self.event_store.list(run_id, after=after):
                yield event
            return

        queue = await self.broker.attach(run_id)
        try:
            history = await self.event_store.list(run_id, after=after)
            for event in history:
                yield event
            if history and self._is_terminal_event(history[-1]):
                return
            cursor = history[-1].sequence if history else after
            while True:
                item = await queue.get()
                if not isinstance(item, RunEvent):
                    await self.broker.detach(run_id, queue)
                    queue = await self.broker.attach(run_id)
                    missed = await self.event_store.list(run_id, after=cursor)
                    for missed_event in missed:
                        yield missed_event
                        cursor = missed_event.sequence
                    if missed and self._is_terminal_event(missed[-1]):
                        return
                    if not missed:
                        # The publisher may have disconnected us just before its
                        # durable append became visible. The new attachment closes
                        # that race and the next event will advance the cursor.
                        await asyncio.sleep(0)
                    continue
                if item.sequence <= cursor:
                    continue
                yield item
                cursor = item.sequence
                if self._is_terminal_event(item):
                    return
        finally:
            await self.broker.detach(run_id, queue)

    async def _drive(self, session: _Session) -> RunResult:
        remaining_timeout = session.budget.timeout_seconds - session.elapsed_active_seconds
        if remaining_timeout <= 0:
            return await self._finish(session, RunStatus.TIMED_OUT, error="run timed out")
        active_started = monotonic()
        session._active_started_monotonic = active_started
        try:
            async with asyncio.timeout(remaining_timeout):
                while session.next_node is not None:
                    self._check_budget(session)
                    controlled = await self._execute_node(session)
                    if controlled is not None:
                        return controlled
                return await self._finish(session, RunStatus.SUCCEEDED)
        except TimeoutError:
            return await self._finish(session, RunStatus.TIMED_OUT, error="run timed out")
        except _BudgetExceeded as exc:
            return await self._finish(session, RunStatus.BUDGET_EXCEEDED, error=str(exc))
        except asyncio.CancelledError:
            await self._finish(session, RunStatus.CANCELLED, error="run cancelled")
            raise
        except Exception as exc:
            if session.current_node is not None:
                await session.emit(
                    EventType.NODE_FAILED,
                    {"error": str(exc), "error_type": type(exc).__name__},
                )
            return await self._finish(session, RunStatus.FAILED, error=str(exc))
        finally:
            session.elapsed_active_seconds += monotonic() - active_started
            session._active_started_monotonic = None
            session.current_node = None

    async def _execute_node(self, session: _Session) -> RunResult | None:
        node_id = session.next_node
        assert node_id is not None
        session.current_node = node_id
        await session.emit(EventType.NODE_STARTED, {"step": session.metrics.steps + 1})

        async def emit_node_event(
            event_type: EventType, payload: Mapping[str, JsonValue]
        ) -> RunEvent:
            return await session.emit(event_type, payload)

        async def delegate(
            child_graph: CompiledGraph,
            state: Mapping[str, JsonValue],
            label: str | None,
            budget: RunBudget | None,
            capabilities: Sequence[str],
        ) -> RunResult:
            if isinstance(capabilities, str) or not all(
                isinstance(name, str) for name in capabilities
            ):
                raise TypeError("child capabilities must be a sequence of dependency names")
            requested_capabilities = tuple(capabilities)
            unknown = sorted(set(requested_capabilities) - set(self.dependencies))
            if unknown:
                raise ValueError(f"unknown child capabilities: {', '.join(unknown)}")
            child_budget, reservation = await self._reserve_child_budget(session, budget)
            child_id = uuid4()
            await session.emit(
                EventType.AGENT_DELEGATED,
                {
                    "child_run_id": str(child_id),
                    "graph_name": child_graph.name,
                    "label": label,
                    "capabilities": list(requested_capabilities),
                },
            )
            try:
                child_runtime = GraphRuntime(
                    event_store=self.event_store,
                    checkpoint_store=self.checkpoint_store,
                    dependencies={name: self.dependencies[name] for name in requested_capabilities},
                    broker=self.broker,
                )
                result = await child_runtime.run(
                    child_graph,
                    state,
                    budget=child_budget,
                    run_id=child_id,
                    trace_id=session.trace_id,
                    parent_run_id=session.run_id,
                )
            except BaseException:
                await self._release_child_budget(session, reservation, None)
                raise
            await self._release_child_budget(session, reservation, result)
            await session.emit(
                EventType.AGENT_JOINED,
                {
                    "child_run_id": str(child_id),
                    "status": result.status.value,
                    "label": label,
                    "metrics": {
                        "steps": result.steps,
                        "model_calls": result.model_calls,
                        "tool_calls": result.tool_calls,
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                        "cost_usd": result.cost_usd,
                    },
                },
            )
            return result

        context = NodeContext.create(
            run_id=session.run_id,
            trace_id=session.trace_id,
            parent_run_id=session.parent_run_id,
            node_id=node_id,
            checkpoint_revision=session.checkpoint_revision,
            state=session.state,
            budget=session.budget,
            dependencies=self.dependencies,
            emit=emit_node_event,
            delegate=delegate,
        )
        raw_result = session.graph.node(node_id)(context)
        if inspect.isawaitable(raw_result):
            raw_result = await raw_result
        result = self._normalize_node_result(raw_result)
        updated_state = {**session.state, **result.update}
        next_node: str | None
        if result.pause is not None and result.goto is None:
            next_node = node_id
        else:
            next_node = await session.graph.next_node(node_id, updated_state, result)
        session.state = updated_state
        session.next_node = next_node
        session.output = result.output
        session.metrics.add(result)
        self._record_action(session, result.action_key)
        await session.emit(
            EventType.NODE_COMPLETED,
            {
                "next_node": next_node,
                "update": result.update,
                "output": result.output,
                "action_key": result.action_key,
                "usage": result.usage.model_dump(mode="json"),
                "model_calls": result.model_calls,
                "tool_calls": result.tool_calls,
                "metadata": result.metadata,
            },
        )
        status = result.pause or (
            RunStatus.SUCCEEDED if session.next_node is None else RunStatus.RUNNING
        )
        await self._save_checkpoint(session, status)
        self._check_consumed_budget(session)
        self._check_repeated_action(session)
        session.current_node = None
        if result.pause is not None:
            await session.emit(EventType.RUN_STATUS, {"status": result.pause.value}, node_id=None)
            return self._result(session, result.pause)
        return None

    async def _save_checkpoint(self, session: _Session, status: RunStatus) -> None:
        revision = 0 if session.checkpoint_revision is None else session.checkpoint_revision + 1
        checkpoint = Checkpoint(
            run_id=session.run_id,
            revision=revision,
            graph_name=session.graph.name,
            graph_version=session.graph.version,
            next_node=session.next_node,
            status=status,
            state=session.state,
            elapsed_active_seconds=self._elapsed_active_seconds(session),
        )
        await self.checkpoint_store.save(checkpoint, expected_revision=session.checkpoint_revision)
        session.checkpoint_revision = revision
        await session.emit(
            EventType.CHECKPOINT_SAVED,
            {"revision": revision, "status": status.value, "next_node": session.next_node},
        )

    async def _finish(
        self, session: _Session, status: RunStatus, *, error: str | None = None
    ) -> RunResult:
        if status == RunStatus.SUCCEEDED:
            event_type = EventType.RUN_COMPLETED
        elif status == RunStatus.FAILED:
            event_type = EventType.RUN_FAILED
        else:
            event_type = EventType.RUN_STATUS
        if status != RunStatus.SUCCEEDED:
            await self._save_checkpoint(session, status)
        await session.emit(
            event_type,
            {
                "status": status.value,
                "output": session.output,
                "error": error,
                "metrics": self._metrics_payload(session.metrics),
            },
            node_id=None,
        )
        await self.broker.close(session.run_id)
        return self._result(session, status, error=error)

    async def _reserve_child_budget(
        self, session: _Session, requested: RunBudget | None
    ) -> tuple[RunBudget, tuple[int, int, float]]:
        """Atomically reserve a strict subset of the parent's remaining budget."""

        async with session._budget_lock:
            remaining_steps = (
                session.budget.max_steps - session.metrics.steps - session.reserved_steps
            )
            remaining_tokens = (
                session.budget.max_tokens
                - session.metrics.input_tokens
                - session.metrics.output_tokens
                - session.reserved_tokens
            )
            remaining_cost = (
                session.budget.max_cost_usd - session.metrics.cost_usd - session.reserved_cost_usd
            )
            remaining_timeout = session.budget.timeout_seconds - self._elapsed_active_seconds(
                session
            )
            if (
                remaining_steps < 1
                or remaining_tokens < 1
                or remaining_cost < 0
                or remaining_timeout <= 0
            ):
                raise _BudgetExceeded("no parent budget remains for a child run")
            if requested is None:
                child = RunBudget(
                    max_steps=remaining_steps,
                    max_tokens=remaining_tokens,
                    max_cost_usd=remaining_cost,
                    timeout_seconds=remaining_timeout,
                    max_repeated_actions=session.budget.max_repeated_actions,
                )
            else:
                if (
                    requested.max_steps > remaining_steps
                    or requested.max_tokens > remaining_tokens
                    or requested.max_cost_usd > remaining_cost
                    or requested.timeout_seconds > remaining_timeout
                ):
                    raise _BudgetExceeded("child budget exceeds the parent's remaining budget")
                child = requested
            reservation = (child.max_steps, child.max_tokens, child.max_cost_usd)
            session.reserved_steps += reservation[0]
            session.reserved_tokens += reservation[1]
            session.reserved_cost_usd += reservation[2]
            return child, reservation

    async def _release_child_budget(
        self,
        session: _Session,
        reservation: tuple[int, int, float],
        result: RunResult | None,
    ) -> None:
        async with session._budget_lock:
            session.reserved_steps -= reservation[0]
            session.reserved_tokens -= reservation[1]
            session.reserved_cost_usd -= reservation[2]
            if result is not None:
                session.metrics.steps += result.steps
                session.metrics.model_calls += result.model_calls
                session.metrics.tool_calls += result.tool_calls
                session.metrics.input_tokens += result.input_tokens
                session.metrics.output_tokens += result.output_tokens
                session.metrics.cost_usd += result.cost_usd

    def _result(
        self, session: _Session, status: RunStatus, *, error: str | None = None
    ) -> RunResult:
        return RunResult(
            run_id=session.run_id,
            trace_id=session.trace_id,
            parent_run_id=session.parent_run_id,
            graph_name=session.graph.name,
            graph_version=session.graph.version,
            status=status,
            state=session.state,
            output=session.output,
            next_node=session.next_node,
            checkpoint_revision=session.checkpoint_revision,
            steps=session.metrics.steps,
            model_calls=session.metrics.model_calls,
            tool_calls=session.metrics.tool_calls,
            input_tokens=session.metrics.input_tokens,
            output_tokens=session.metrics.output_tokens,
            cost_usd=session.metrics.cost_usd,
            started_at=session.started_at,
            ended_at=utc_now(),
            error=error,
        )

    @staticmethod
    def _normalize_node_result(raw: object) -> NodeResult:
        if raw is None:
            return NodeResult()
        if isinstance(raw, NodeResult):
            return raw
        if isinstance(raw, Mapping):
            return NodeResult(update=dict(raw))
        raise TypeError(
            "nodes must return NodeResult, a JSON-compatible mapping, or None; "
            f"got {type(raw).__name__}"
        )

    @staticmethod
    def _check_budget(session: _Session) -> None:
        metrics = session.metrics
        budget = session.budget
        if metrics.steps >= budget.max_steps:
            raise _BudgetExceeded(f"step budget exceeded ({budget.max_steps})")
        GraphRuntime._check_consumed_budget(session)

    @staticmethod
    def _check_consumed_budget(session: _Session) -> None:
        metrics = session.metrics
        budget = session.budget
        if metrics.input_tokens + metrics.output_tokens > budget.max_tokens:
            raise _BudgetExceeded(f"token budget exceeded ({budget.max_tokens})")
        if metrics.cost_usd > budget.max_cost_usd:
            raise _BudgetExceeded(f"cost budget exceeded (${budget.max_cost_usd:.4f})")

    @staticmethod
    def _record_action(session: _Session, action_key: str | None) -> None:
        if action_key is None:
            session.last_action = None
            session.repeated_actions = 0
            return
        if action_key == session.last_action:
            session.repeated_actions += 1
        else:
            session.last_action = action_key
            session.repeated_actions = 1

    @staticmethod
    def _check_repeated_action(session: _Session) -> None:
        if session.repeated_actions > session.budget.max_repeated_actions:
            raise _BudgetExceeded(
                f"repeated action guard exceeded for {session.last_action!r} "
                f"({session.budget.max_repeated_actions})"
            )

    @staticmethod
    def _metrics_payload(metrics: _Metrics) -> dict[str, JsonValue]:
        return {
            "steps": metrics.steps,
            "model_calls": metrics.model_calls,
            "tool_calls": metrics.tool_calls,
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "cost_usd": metrics.cost_usd,
        }

    @staticmethod
    def _elapsed_active_seconds(session: _Session) -> float:
        elapsed = session.elapsed_active_seconds
        if session._active_started_monotonic is not None:
            elapsed += monotonic() - session._active_started_monotonic
        return elapsed

    @staticmethod
    def _metrics_from_events(events: list[RunEvent]) -> _Metrics:
        metrics = _Metrics()
        for event in events:
            if event.type != EventType.NODE_COMPLETED:
                continue
            usage = event.payload.get("usage")
            if isinstance(usage, dict):
                parsed = ModelUsage.model_validate(usage)
                metrics.input_tokens += parsed.input_tokens
                metrics.output_tokens += parsed.output_tokens
                metrics.cost_usd += parsed.cost_usd
            model_calls = event.payload.get("model_calls", 0)
            tool_calls = event.payload.get("tool_calls", 0)
            metrics.model_calls += model_calls if isinstance(model_calls, int) else 0
            metrics.tool_calls += tool_calls if isinstance(tool_calls, int) else 0
            metrics.steps += 1
        return metrics

    @staticmethod
    def _restore_action_guard(session: _Session, events: list[RunEvent]) -> None:
        for event in reversed(events):
            if event.type != EventType.NODE_COMPLETED:
                continue
            action = event.payload.get("action_key")
            if not isinstance(action, str):
                break
            if session.last_action is None:
                session.last_action = action
            if action != session.last_action:
                break
            session.repeated_actions += 1

    @staticmethod
    def _budget_from_events(events: list[RunEvent]) -> RunBudget:
        for event in events:
            if event.type == EventType.RUN_CREATED:
                raw = event.payload.get("budget")
                if isinstance(raw, dict):
                    return RunBudget.model_validate(raw)
        return RunBudget()

    @staticmethod
    def _is_terminal_event(event: RunEvent) -> bool:
        if event.type in {EventType.RUN_COMPLETED, EventType.RUN_FAILED}:
            return True
        if event.type != EventType.RUN_STATUS:
            return False
        status = event.payload.get("status")
        return isinstance(status, str) and status in {
            member.value for member in RunStatus if member.terminal
        }
