"""Lifecycle service shared by the CLI and HTTP API."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from admires_harness.adapters.models import MockModelProvider, OpenAICompatibleProvider
from admires_harness.agents.reference import build_demo, list_demos, offline_responder
from admires_harness.config import Settings
from admires_harness.contracts import (
    ApprovalDecision,
    ApprovalRequest,
    EventType,
    JsonValue,
    RunBudget,
    RunEvent,
    RunStatus,
    utc_now,
)
from admires_harness.memory import RAGRetriever, SQLiteMemoryStore
from admires_harness.observability import RunView, project_run
from admires_harness.runtime import CheckpointConflictError, GraphRuntime
from admires_harness.safety import ApprovalGate, SafetyPolicy, SecuredToolExecutor, WorkspaceSandbox
from admires_harness.services.export import DEFAULT_EXPORT_LIMIT, build_trace_export
from admires_harness.storage import SQLiteCheckpointStore, SQLiteEventStore
from admires_harness.tools import workspace_tools

_REFERENCE_KNOWLEDGE = (
    (
        "architecture.md",
        "An append-only event log records every agent transition with a stable sequence. "
        "CLI, API, live streams, replay, tracing, and evaluation consume this same envelope.",
    ),
    (
        "durable-execution.md",
        "A checkpoint stores graph version, state, and next node at a node boundary. "
        "Compare-and-swap revisions prevent two workers from advancing one run.",
    ),
    (
        "replay.md",
        "Replay reads persisted events and checkpoints without invoking a model or tool. "
        "Side effects use stable idempotency keys because universal exactly-once is impossible.",
    ),
)


class RunManager:
    """Own durable adapters, run tasks, and per-run workspaces."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.events = SQLiteEventStore(settings.database_path)
        self.checkpoints = SQLiteCheckpointStore(settings.database_path)
        self.memory_store = SQLiteMemoryStore(settings.database_path.with_name("memory.db"))
        self.rag = RAGRetriever(self.memory_store)
        self._tasks: dict[UUID, asyncio.Task[Any]] = {}
        self._workspaces: dict[UUID, WorkspaceSandbox] = {}
        self._approvals: dict[UUID, ApprovalGate] = {}
        self._initialized = False
        self._initialize_lock = asyncio.Lock()
        self._decision_locks: dict[UUID, asyncio.Lock] = {}

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._initialize_lock:
            if self._initialized:
                return
            self.settings.ensure_directories()
            await self.events.initialize()
            await self.checkpoints.initialize()
            await self.memory_store.initialize()
            existing = await self.memory_store.search(
                tuple([0.0] * 256), namespace="reference-research", limit=1
            )
            if not existing:
                for source, content in _REFERENCE_KNOWLEDGE:
                    await self.rag.ingest(content, namespace="reference-research", source=source)
            self._initialized = True

    def _model(self) -> Any:
        provider = self.settings.model_provider.strip().casefold()
        if provider == "mock":
            return MockModelProvider(responder=offline_responder)
        if provider in {"openai", "openai-compatible"}:
            if not self.settings.model_api_key:
                raise ValueError(
                    "ADMIRE_MODEL_API_KEY is required when ADMIRE_MODEL_PROVIDER is not mock"
                )
            return OpenAICompatibleProvider(
                model=self.settings.model_name,
                api_key=self.settings.model_api_key,
                base_url=self.settings.model_base_url,
            )
        raise ValueError(f"unknown model provider: {self.settings.model_provider}")

    def _workspace(self, run_id: UUID) -> WorkspaceSandbox:
        workspace = self._workspaces.get(run_id)
        if workspace is None:
            workspace = WorkspaceSandbox(self.settings.workspace_root / str(run_id))
            self._workspaces[run_id] = workspace
        return workspace

    async def _approval_gate(self, run_id: UUID) -> ApprovalGate:
        existing = self._approvals.get(run_id)
        if existing is not None:
            return existing
        gate = ApprovalGate()
        requests: dict[UUID, ApprovalRequest] = {}
        decisions: dict[UUID, ApprovalDecision] = {}
        for event in await self.events.list(run_id):
            raw_request = event.payload.get("request")
            if event.type in {EventType.APPROVAL_REQUESTED, EventType.APPROVAL_CONSUMED}:
                if isinstance(raw_request, dict):
                    request = ApprovalRequest.model_validate(raw_request)
                    requests[request.request_id] = request
            elif event.type is EventType.APPROVAL_RESOLVED:
                raw_decision = event.payload.get("decision")
                if isinstance(raw_decision, dict):
                    decision = ApprovalDecision.model_validate(raw_decision)
                    decisions[decision.request_id] = decision
        for request in requests.values():
            await gate.restore(request, decisions.get(request.request_id))
        self._approvals[run_id] = gate
        return gate

    async def _runtime(self, run_id: UUID) -> GraphRuntime:
        workspace = self._workspace(run_id)
        registry = workspace_tools(workspace)
        policy = SafetyPolicy(registry.safety_specs, allow_network=self.settings.allow_network)
        approvals = await self._approval_gate(run_id)
        return GraphRuntime(
            event_store=self.events,
            checkpoint_store=self.checkpoints,
            dependencies={
                "model": self._model(),
                "rag": self.rag,
                "workspace": workspace,
                "tools": SecuredToolExecutor(registry, policy=policy, approvals=approvals),
                "approvals": approvals,
                "tool_policy": policy,
            },
        )

    async def run_sync(
        self,
        demo_id: str,
        task_input: str,
        *,
        run_id: UUID | None = None,
        budget: RunBudget | None = None,
        auto_approve: bool = False,
    ) -> RunView:
        await self.initialize()
        if not task_input.strip():
            raise ValueError("task input cannot be empty")
        actual_id = run_id or uuid4()
        runtime = await self._runtime(actual_id)
        await runtime.run(
            build_demo(demo_id),
            {"input": task_input, "demo": demo_id},
            run_id=actual_id,
            budget=budget,
        )
        view = await self.get(actual_id)
        if auto_approve and view.status is RunStatus.WAITING_APPROVAL:
            pending = await self.list_approvals(actual_id)
            for item in pending:
                raw_id = item.get("requestId")
                if isinstance(raw_id, str):
                    view = await self.decide_approval(
                        actual_id,
                        UUID(raw_id),
                        approved=True,
                        decided_by="offline-cli",
                        comment="Explicit auto-approval for the deterministic local demo.",
                    )
        return view

    async def start(self, demo_id: str, task_input: str) -> RunView:
        await self.initialize()
        valid = {demo.id for demo in list_demos()}
        if demo_id not in valid:
            raise ValueError(f"unknown demo: {demo_id}")
        if not task_input.strip():
            raise ValueError("task input cannot be empty")
        run_id = uuid4()
        runtime = await self._runtime(run_id)

        async def execute() -> None:
            await runtime.run(
                build_demo(demo_id), {"input": task_input, "demo": demo_id}, run_id=run_id
            )

        task = asyncio.create_task(execute(), name=f"run-{run_id}")
        self._tasks[run_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(run_id, None))
        for _ in range(100):
            if await self.events.list(run_id):
                break
            await asyncio.sleep(0)
        return await self.get(run_id)

    async def get(self, run_id: UUID) -> RunView:
        await self.initialize()
        events = await self.events.list(run_id)
        if not events:
            raise KeyError(str(run_id))
        return project_run(events)

    async def list_runs(self, *, limit: int = 50) -> list[RunView]:
        await self.initialize()
        run_ids = await self.events.list_run_ids(limit=limit)
        views: list[RunView] = []
        for run_id in run_ids:
            events = await self.events.list(run_id)
            if events and events[0].parent_run_id is None:
                views.append(project_run(events))
        return views

    async def list_events(self, run_id: UUID, *, after: int = -1) -> list[RunEvent]:
        await self.initialize()
        return await self.events.list(run_id, after=after)

    async def export_trace(
        self, run_id: UUID, *, limit: int = DEFAULT_EXPORT_LIMIT
    ) -> dict[str, JsonValue]:
        """Build a bounded, redacted export of a run's persisted events.

        Read-only: never invokes a model and never mutates run state. A run
        with no persisted events is indistinguishable from a missing run and
        raises KeyError.
        """

        await self.initialize()
        events = await self.events.list(run_id)
        if not events:
            raise KeyError(str(run_id))
        return build_trace_export(project_run(events), events, limit=limit)

    async def wait(self, run_id: UUID) -> RunView:
        """Wait until this manager has finished its local work for a run.

        A projected ``waiting_approval`` status can become visible just before the
        background task has flushed its final checkpoint/event.  Waiting at an
        ownership boundary (for example, graceful process shutdown) prevents a
        replacement manager from racing that final append.
        """

        await self._await_local_task(run_id)
        return await self.get(run_id)

    async def close(self) -> None:
        """Gracefully drain background runs owned by this manager."""

        while tasks := tuple(self._tasks.values()):
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _await_local_task(self, run_id: UUID) -> None:
        task = self._tasks.get(run_id)
        if task is not None and not task.done():
            with suppress(asyncio.CancelledError):
                await task

    async def stream(self, run_id: UUID, *, after: int = -1) -> AsyncIterator[RunEvent]:
        await self.initialize()
        cursor = after
        while True:
            events = await self.events.list(run_id, after=cursor)
            for event in events:
                yield event
                cursor = event.sequence
            if events and _terminal(events[-1]):
                return
            if not events:
                all_events = await self.events.list(run_id)
                if all_events and _terminal(all_events[-1]):
                    return
            await asyncio.sleep(0.05)

    async def cancel(self, run_id: UUID) -> RunView:
        task = self._tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            return await self.get(run_id)
        current = await self.get(run_id)
        if current.status.terminal:
            return current
        checkpoint = await self.checkpoints.latest(run_id)
        if checkpoint is None:
            raise ValueError("run has no durable checkpoint to cancel")
        cancelled = checkpoint.model_copy(
            update={
                "revision": checkpoint.revision + 1,
                "status": RunStatus.CANCELLED,
                "created_at": utc_now(),
            }
        )
        await self.checkpoints.save(cancelled, expected_revision=checkpoint.revision)
        history = await self.events.list(run_id)
        await self.events.append(
            RunEvent(
                run_id=run_id,
                sequence=history[-1].sequence + 1,
                type=EventType.RUN_STATUS,
                trace_id=history[0].trace_id,
                payload={
                    "status": RunStatus.CANCELLED.value,
                    "error": "run cancelled by user",
                    "metrics": current.metrics.model_dump(mode="json"),
                },
            )
        )
        return await self.get(run_id)

    async def resume(self, run_id: UUID) -> RunView:
        current = await self.get(run_id)
        if current.status is RunStatus.WAITING_APPROVAL:
            raise ValueError("resolve the pending approval before resuming this run")
        if current.status is not RunStatus.PAUSED:
            raise ValueError(f"run is not resumable: {current.status.value}")
        runtime = await self._runtime(run_id)
        await runtime.resume(build_demo(current.demo), run_id)
        return await self.get(run_id)

    async def replay(self, run_id: UUID) -> dict[str, JsonValue]:
        await self.initialize()
        runtime = await self._runtime(run_id)
        snapshot = await runtime.replay(run_id)
        event_payloads: list[JsonValue] = [
            event.model_dump(mode="json") for event in snapshot.events
        ]
        return {
            "runId": str(snapshot.run_id),
            "status": snapshot.status.value,
            "state": snapshot.state,
            "nextNode": snapshot.next_node,
            "checkpointRevision": snapshot.checkpoint_revision,
            "events": event_payloads,
        }

    async def list_approvals(self, run_id: UUID) -> list[dict[str, JsonValue]]:
        await self.initialize()
        events = await self.events.list(run_id)
        if not events:
            raise KeyError(str(run_id))
        if project_run(events).status is not RunStatus.WAITING_APPROVAL:
            return []
        decisions: dict[str, ApprovalDecision] = {}
        consumed: set[str] = set()
        requests: dict[str, tuple[ApprovalRequest, dict[str, JsonValue]]] = {}
        for event in events:
            if event.type is EventType.APPROVAL_REQUESTED:
                raw_request = event.payload.get("request")
                raw_arguments = event.payload.get("arguments")
                if isinstance(raw_request, dict) and isinstance(raw_arguments, dict):
                    request = ApprovalRequest.model_validate(raw_request)
                    requests[str(request.request_id)] = (request, raw_arguments)
            elif event.type is EventType.APPROVAL_RESOLVED:
                raw_decision = event.payload.get("decision")
                if isinstance(raw_decision, dict):
                    request_id = raw_decision.get("request_id")
                    if isinstance(request_id, str):
                        decisions[request_id] = ApprovalDecision.model_validate(raw_decision)
            elif event.type is EventType.APPROVAL_CONSUMED:
                raw_request = event.payload.get("request")
                if isinstance(raw_request, dict):
                    request_id = raw_request.get("request_id")
                    if isinstance(request_id, str):
                        consumed.add(request_id)
        return [
            {
                "requestId": request_id,
                "toolName": request.tool_name,
                "reason": request.reason,
                "risk": request.risk.value,
                "expiresAt": request.expires_at.isoformat(),
                "arguments": arguments,
                "checkpointRevision": request.checkpoint_revision,
                "argumentsDigest": request.arguments_digest,
                "decided": request_id in decisions,
                "approved": (decisions[request_id].approved if request_id in decisions else None),
            }
            for request_id, (request, arguments) in requests.items()
            # A resolved-but-unconsumed request remains visible. Retrying the
            # same decision is the recovery path if the process exited after
            # persisting APPROVAL_RESOLVED but before starting resume().
            if request_id not in consumed
        ]

    async def decide_approval(
        self,
        run_id: UUID,
        request_id: UUID,
        *,
        approved: bool,
        decided_by: str,
        comment: str | None = None,
    ) -> RunView:
        lock = self._decision_locks.setdefault(run_id, asyncio.Lock())
        async with lock:
            await self._await_local_task(run_id)
            current = await self.get(run_id)
            if current.status is not RunStatus.WAITING_APPROVAL:
                raise ValueError(f"run is not waiting for approval: {current.status.value}")
            gate = await self._approval_gate(run_id)
            state = await gate.get(request_id)
            if state is None or state[0].run_id != run_id:
                raise KeyError(str(request_id))
            _, existing_decision = state
            if existing_decision is not None:
                if existing_decision.approved != approved:
                    raise ValueError("approval was already resolved with a different decision")
                decision = existing_decision
            else:
                decision = ApprovalDecision(
                    request_id=request_id,
                    approved=approved,
                    decided_by=decided_by,
                    comment=comment,
                )
                await gate.resolve(decision)
                history = await self.events.list(run_id)
                try:
                    await self.events.append(
                        RunEvent(
                            run_id=run_id,
                            sequence=history[-1].sequence + 1,
                            type=EventType.APPROVAL_RESOLVED,
                            trace_id=history[0].trace_id,
                            payload={"decision": decision.model_dump(mode="json")},
                        )
                    )
                except CheckpointConflictError as exc:
                    # Another process won the append race. Forget transient state so a
                    # retry projects the authoritative event history and fails closed.
                    self._approvals.pop(run_id, None)
                    raise ValueError("approval was decided concurrently") from exc
                except Exception:
                    self._approvals.pop(run_id, None)
                    raise
            runtime = await self._runtime(run_id)
            await runtime.resume(build_demo(current.demo), run_id)
            return await self.get(run_id)


def _terminal(event: RunEvent) -> bool:
    status = event.payload.get("status")
    if isinstance(status, str):
        try:
            return RunStatus(status).terminal
        except ValueError:
            return False
    return event.type.value in {"run.completed", "run.failed"}


def as_camel(view: RunView) -> dict[str, JsonValue]:
    """Serialize one run using the console's stable public shape."""

    duration = None
    if view.ended_at is not None:
        duration = int((view.ended_at - view.started_at).total_seconds() * 1000)
    return {
        "id": str(view.run_id),
        "demo": view.demo,
        "input": view.input,
        "status": view.status.value,
        "createdAt": view.started_at.astimezone(UTC).isoformat(),
        "updatedAt": (view.ended_at or datetime.now(UTC)).astimezone(UTC).isoformat(),
        "output": view.output,
        "error": view.error,
        "metrics": {
            "steps": view.metrics.steps,
            "modelCalls": view.metrics.model_calls,
            "toolCalls": view.metrics.tool_calls,
            "inputTokens": view.metrics.input_tokens,
            "outputTokens": view.metrics.output_tokens,
            "costUsd": view.metrics.cost_usd,
            "durationMs": duration,
        },
    }
