from __future__ import annotations

import asyncio
from uuid import UUID

import pytest

from admires_harness.contracts import (
    EventType,
    ModelUsage,
    RunBudget,
    RunEvent,
    RunStatus,
)
from admires_harness.runtime import (
    GraphRuntime,
    InMemoryCheckpointStore,
    InMemoryEventStore,
    NodeContext,
    NodeResult,
    StateGraph,
)


def make_runtime(
    *, dependencies: dict[str, object] | None = None
) -> tuple[GraphRuntime, InMemoryEventStore, InMemoryCheckpointStore]:
    events = InMemoryEventStore()
    checkpoints = InMemoryCheckpointStore()
    runtime = GraphRuntime(
        event_store=events,
        checkpoint_store=checkpoints,
        dependencies=dependencies or {},
    )
    return runtime, events, checkpoints


def child_graph(name: str, node: object) -> object:
    return StateGraph(name).add_node("work", node).set_entry_point("work").compile()


def parent_graph(node: object) -> object:
    return StateGraph("parent").add_node("parent", node).set_entry_point("parent").compile()


def background_tasks() -> set[asyncio.Task[object]]:
    current = asyncio.current_task()
    return {task for task in asyncio.all_tasks() if task is not current and not task.done()}


@pytest.mark.asyncio
async def test_delegated_child_has_identity_linkage_and_events() -> None:
    async def child(context: NodeContext) -> NodeResult:
        assert context.parent_run_id is not None
        return NodeResult(
            finish=True,
            usage=ModelUsage(input_tokens=4, output_tokens=2, cost_usd=0.05),
            model_calls=1,
        )

    captured: dict[str, object] = {}

    async def parent(context: NodeContext) -> NodeResult:
        captured["result"] = await context.delegate(child_graph("kid", child), {"seed": 1})
        return NodeResult(finish=True)

    runtime, events, checkpoints = make_runtime()
    result = await runtime.run(parent_graph(parent))
    child_result = captured["result"]  # type: ignore[assignment]

    assert result.status == RunStatus.SUCCEEDED
    assert child_result.run_id != result.run_id
    assert child_result.trace_id != result.trace_id
    assert child_result.run_id != child_result.trace_id
    assert child_result.parent_run_id == result.run_id

    parent_events = await events.list(result.run_id)
    child_events = await events.list(child_result.run_id)
    delegated = next(event for event in parent_events if event.type == EventType.AGENT_DELEGATED)
    joined = next(event for event in parent_events if event.type == EventType.AGENT_JOINED)
    assert delegated.sequence < joined.sequence
    assert delegated.payload["child_run_id"] == str(child_result.run_id)
    assert delegated.payload["child_trace_id"] == str(child_result.trace_id)
    assert joined.payload["child_run_id"] == str(child_result.run_id)
    assert joined.payload["child_trace_id"] == str(child_result.trace_id)
    assert joined.payload["status"] == "succeeded"
    assert joined.payload["metrics"]["input_tokens"] == 4
    for event in child_events:
        assert event.run_id == child_result.run_id
        assert event.trace_id == child_result.trace_id
        assert event.parent_run_id == result.run_id
    assert child_events[0].type == EventType.RUN_CREATED
    assert child_events[-1].type == EventType.RUN_COMPLETED
    assert (await checkpoints.latest(child_result.run_id)).status == RunStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_child_capabilities_are_downgraded_and_unknown_names_rejected() -> None:
    seen: set[str] = set()

    async def child(context: NodeContext) -> NodeResult:
        seen.update(context.dependencies)
        return NodeResult(finish=True)

    async def parent(context: NodeContext) -> NodeResult:
        assert set(context.dependencies) == {"model", "secret"}
        result = await context.delegate(child_graph("kid", child), {}, capabilities=("model",))
        assert result.status == RunStatus.SUCCEEDED
        with pytest.raises(ValueError, match="unknown child capabilities"):
            await context.delegate(child_graph("kid", child), {}, capabilities=("missing",))
        return NodeResult(finish=True)

    runtime, events, _ = make_runtime(dependencies={"model": object(), "secret": object()})
    result = await runtime.run(parent_graph(parent))

    print("PARENT ERR", result.error)
    assert result.status == RunStatus.SUCCEEDED
    assert seen == {"model"}
    parent_events = await events.list(result.run_id)
    delegated = next(event for event in parent_events if event.type == EventType.AGENT_DELEGATED)
    assert delegated.payload["capabilities"] == ["model"]
    assert [event.type for event in parent_events].count(EventType.AGENT_DELEGATED) == 1
    assert [event.type for event in parent_events].count(EventType.AGENT_JOINED) == 1


@pytest.mark.asyncio
async def test_unbounded_child_budget_clamps_to_parent_remaining() -> None:
    captured: dict[str, object] = {}

    async def child(context: NodeContext) -> NodeResult:
        captured["budget"] = context.budget
        return NodeResult(
            finish=True,
            usage=ModelUsage(input_tokens=2, output_tokens=1, cost_usd=0.01),
        )

    async def parent(context: NodeContext) -> NodeResult:
        captured["result"] = await context.delegate(child_graph("kid", child), {})
        return NodeResult(finish=True)

    runtime, events, _ = make_runtime()
    result = await runtime.run(
        parent_graph(parent),
        budget=RunBudget(max_steps=4, max_tokens=30, max_cost_usd=0.5, timeout_seconds=2),
    )

    assert result.status == RunStatus.SUCCEEDED
    child_budget = captured["budget"]  # type: ignore[assignment]
    assert child_budget.max_steps == 4
    assert child_budget.max_tokens == 30
    assert child_budget.max_cost_usd == 0.5
    assert child_budget.timeout_seconds == pytest.approx(2, rel=0.05)
    delegated = next(
        event
        for event in await events.list(result.run_id)
        if event.type == EventType.AGENT_DELEGATED
    )
    assert delegated.payload["budget"]["max_tokens"] == 30


@pytest.mark.asyncio
async def test_oversized_child_budget_is_rejected_without_event_or_task_leak() -> None:
    async def child(context: NodeContext) -> NodeResult:
        return NodeResult(finish=True)  # pragma: no cover

    async def parent(context: NodeContext) -> NodeResult:
        with pytest.raises(Exception, match="parent's remaining budget"):
            await context.delegate(
                child_graph("kid", child),
                {},
                budget=RunBudget(max_steps=99, max_tokens=99, max_cost_usd=9, timeout_seconds=1),
            )
        return NodeResult(finish=True)

    runtime, events, _ = make_runtime()
    before = background_tasks()
    result = await runtime.run(
        parent_graph(parent),
        budget=RunBudget(max_steps=2, max_tokens=10, max_cost_usd=0.1, timeout_seconds=1),
    )

    assert result.status == RunStatus.SUCCEEDED
    assert not (background_tasks() - before)
    parent_events = await events.list(result.run_id)
    assert EventType.AGENT_DELEGATED not in [event.type for event in parent_events]
    assert EventType.AGENT_JOINED not in [event.type for event in parent_events]


@pytest.mark.asyncio
async def test_two_children_count_usage_once_each() -> None:
    async def child(context: NodeContext) -> NodeResult:
        return NodeResult(
            finish=True,
            usage=ModelUsage(input_tokens=3, output_tokens=2, cost_usd=0.05),
            model_calls=1,
        )

    async def parent(context: NodeContext) -> NodeResult:
        first = await context.delegate(
            child_graph("kid-a", child),
            {},
            budget=RunBudget(max_steps=1, max_tokens=5, max_cost_usd=0.05, timeout_seconds=1),
        )
        second = await context.delegate(
            child_graph("kid-b", child),
            {},
            budget=RunBudget(max_steps=1, max_tokens=5, max_cost_usd=0.05, timeout_seconds=1),
        )
        assert first.status == second.status == RunStatus.SUCCEEDED
        assert first.run_id != second.run_id
        return NodeResult(finish=True)

    runtime, _, _ = make_runtime()
    result = await runtime.run(
        parent_graph(parent),
        budget=RunBudget(max_steps=5, max_tokens=20, max_cost_usd=0.5, timeout_seconds=5),
    )

    assert result.status == RunStatus.SUCCEEDED
    assert (result.input_tokens, result.output_tokens) == (6, 4)
    assert result.cost_usd == pytest.approx(0.1)
    assert result.model_calls == 2
    # Child steps fold into the parent totals (each child run executes one node).
    assert result.steps == 3


@pytest.mark.asyncio
async def test_child_infrastructure_failure_still_joins_and_releases() -> None:
    class _FailingStore(InMemoryEventStore):
        async def append(self, event: RunEvent) -> None:  # type: ignore[override]
            if event.type == EventType.RUN_CREATED and event.payload.get("graph_name") == "broken":
                raise OSError("store unavailable")
            await super().append(event)

    async def child(context: NodeContext) -> NodeResult:
        return NodeResult(finish=True)  # pragma: no cover

    async def parent(context: NodeContext) -> NodeResult:
        with pytest.raises(OSError, match="store unavailable"):
            await context.delegate(child_graph("broken", child), {})
        # The reservation was released, so a second healthy delegation still works.
        follow_up = await context.delegate(child_graph("healthy", child), {})
        assert follow_up.status == RunStatus.SUCCEEDED
        return NodeResult(finish=True)

    events = _FailingStore()
    runtime = GraphRuntime(event_store=events, checkpoint_store=InMemoryCheckpointStore())
    result = await runtime.run(parent_graph(parent))

    assert result.status == RunStatus.SUCCEEDED
    parent_events = await events.list(result.run_id)
    delegated = [event.type for event in parent_events if event.type == EventType.AGENT_DELEGATED]
    joined = [event for event in parent_events if event.type == EventType.AGENT_JOINED]
    assert delegated == [EventType.AGENT_DELEGATED, EventType.AGENT_DELEGATED]
    assert len(joined) == 2
    failed_join = joined[0]
    assert failed_join.payload["status"] == "failed"
    assert "store unavailable" in str(failed_join.payload["error"])
    assert joined[1].payload["status"] == "succeeded"


@pytest.mark.asyncio
async def test_delegation_record_failure_releases_reservation() -> None:
    class _FailingDelegationStore(InMemoryEventStore):
        async def append(self, event: RunEvent) -> None:  # type: ignore[override]
            if event.type == EventType.AGENT_DELEGATED:
                raise OSError("cannot record delegation")
            await super().append(event)

    async def child(context: NodeContext) -> NodeResult:
        return NodeResult(finish=True)  # pragma: no cover

    async def parent(context: NodeContext) -> NodeResult:
        with pytest.raises(OSError, match="cannot record delegation"):
            await context.delegate(
                child_graph("kid", child),
                {},
                budget=RunBudget(max_steps=1, max_tokens=5, max_cost_usd=0.1, timeout_seconds=1),
            )
        return NodeResult(finish=True)

    events = _FailingDelegationStore()
    runtime = GraphRuntime(event_store=events, checkpoint_store=InMemoryCheckpointStore())
    result = await runtime.run(parent_graph(parent))

    assert result.status == RunStatus.SUCCEEDED
    # The reservation was released and no child run started, so there is no join.
    parent_events = await events.list(result.run_id)
    assert EventType.AGENT_JOINED not in [event.type for event in parent_events]


@pytest.mark.asyncio
async def test_failed_child_converges_as_data_without_parent_failure() -> None:
    async def child(context: NodeContext) -> NodeResult:
        raise RuntimeError("child exploded")

    statuses: list[RunStatus] = []

    async def parent(context: NodeContext) -> NodeResult:
        result = await context.delegate(child_graph("kid", child), {})
        statuses.append(result.status)
        assert result.error == "child exploded"
        return NodeResult(update={"child_status": result.status.value}, finish=True)

    runtime, events, checkpoints = make_runtime()
    result = await runtime.run(parent_graph(parent))

    assert result.status == RunStatus.SUCCEEDED
    assert statuses == [RunStatus.FAILED]
    parent_events = await events.list(result.run_id)
    assert [event.type for event in parent_events].count(EventType.AGENT_DELEGATED) == 1
    joined = next(event for event in parent_events if event.type == EventType.AGENT_JOINED)
    assert [event.type for event in parent_events].count(EventType.AGENT_JOINED) == 1
    assert joined.payload["status"] == "failed"
    assert joined.payload["error"] == "child exploded"
    child_id = UUID(str(joined.payload["child_run_id"]))
    child_events = await events.list(child_id)
    assert child_events[-1].type == EventType.RUN_FAILED
    assert (await checkpoints.latest(child_id)).status == RunStatus.FAILED


@pytest.mark.asyncio
async def test_child_timeout_converges_and_joins_once() -> None:
    async def child(context: NodeContext) -> NodeResult:
        await asyncio.sleep(5)
        return NodeResult(finish=True)  # pragma: no cover

    statuses: list[RunStatus] = []

    async def parent(context: NodeContext) -> NodeResult:
        result = await context.delegate(
            child_graph("slow", child),
            {},
            budget=RunBudget(max_steps=1, max_tokens=10, max_cost_usd=1, timeout_seconds=0.05),
        )
        statuses.append(result.status)
        return NodeResult(finish=True)

    runtime, events, checkpoints = make_runtime()
    before = background_tasks()
    result = await runtime.run(
        parent_graph(parent),
        budget=RunBudget(max_steps=2, max_tokens=10, max_cost_usd=1, timeout_seconds=5),
    )

    assert result.status == RunStatus.SUCCEEDED
    assert statuses == [RunStatus.TIMED_OUT]
    assert not (background_tasks() - before)
    parent_events = await events.list(result.run_id)
    joined = next(event for event in parent_events if event.type == EventType.AGENT_JOINED)
    assert joined.payload["status"] == "timed_out"
    child_id = UUID(str(joined.payload["child_run_id"]))
    assert (await checkpoints.latest(child_id)).status == RunStatus.TIMED_OUT
    assert (await events.list(child_id))[-1].type == EventType.RUN_STATUS


@pytest.mark.asyncio
async def test_parent_timeout_cancels_running_child_durably() -> None:
    child_started = asyncio.Event()

    async def child(context: NodeContext) -> NodeResult:
        child_started.set()
        await asyncio.sleep(5)
        return NodeResult(finish=True)  # pragma: no cover

    async def parent(context: NodeContext) -> NodeResult:
        # An explicit child timeout strictly inside the parent's deadline makes
        # the parent/child timer race deterministic: the parent must still cancel
        # and durably converge the child instead of leaking it.
        await context.delegate(
            child_graph("slow", child),
            {},
            budget=RunBudget(max_steps=1, max_tokens=10, max_cost_usd=1, timeout_seconds=0.2),
        )
        return NodeResult(finish=True)  # pragma: no cover

    runtime, events, checkpoints = make_runtime()
    before = background_tasks()
    task = asyncio.create_task(
        runtime.run(
            parent_graph(parent),
            budget=RunBudget(max_steps=2, max_tokens=10, max_cost_usd=1, timeout_seconds=0.2),
        )
    )
    await child_started.wait()
    result = await task

    assert result.status == RunStatus.TIMED_OUT
    assert not (background_tasks() - before)
    parent_events = await events.list(result.run_id)
    types_ = [event.type for event in parent_events]
    assert types_.count(EventType.AGENT_DELEGATED) == 1
    assert types_.count(EventType.AGENT_JOINED) == 1
    joined = next(event for event in parent_events if event.type == EventType.AGENT_JOINED)
    assert joined.payload["status"] in {"cancelled", "timed_out"}
    child_id = UUID(str(joined.payload["child_run_id"]))
    assert (await checkpoints.latest(child_id)).status in {
        RunStatus.CANCELLED,
        RunStatus.TIMED_OUT,
    }


@pytest.mark.asyncio
async def test_external_parent_cancellation_converges_child() -> None:
    child_started = asyncio.Event()
    captured: dict[str, object] = {}

    async def child(context: NodeContext) -> NodeResult:
        captured["parent_run_id"] = context.parent_run_id
        captured["child_run_id"] = context.run_id
        child_started.set()
        await asyncio.sleep(5)
        return NodeResult(finish=True)  # pragma: no cover

    async def parent(context: NodeContext) -> NodeResult:
        await context.delegate(child_graph("slow", child), {})
        return NodeResult(finish=True)  # pragma: no cover

    runtime, events, checkpoints = make_runtime()
    before = background_tasks()
    task = asyncio.create_task(runtime.run(parent_graph(parent)))
    await child_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    assert not (background_tasks() - before)
    parent_id = captured["parent_run_id"]  # type: ignore[assignment]
    child_id = captured["child_run_id"]  # type: ignore[assignment]
    assert child_id != parent_id
    assert (await checkpoints.latest(child_id)).status == RunStatus.CANCELLED
    parent_events = await events.list(parent_id)
    assert [event.type for event in parent_events].count(EventType.AGENT_DELEGATED) == 1
    joined = next(event for event in parent_events if event.type == EventType.AGENT_JOINED)
    assert [event.type for event in parent_events].count(EventType.AGENT_JOINED) == 1
    assert joined.payload["child_run_id"] == str(child_id)
    assert joined.payload["status"] == "cancelled"
    child_events = await events.list(child_id)
    assert child_events[-1].payload.get("status") == "cancelled"


@pytest.mark.asyncio
async def test_concurrent_children_get_unique_identity_and_all_join() -> None:
    seen_ids: set[UUID] = set()
    seen_traces: set[UUID] = set()
    gate = asyncio.Event()

    async def child(context: NodeContext) -> NodeResult:
        seen_ids.add(context.run_id)
        seen_traces.add(context.trace_id)
        await gate.wait()
        return NodeResult(finish=True)

    async def parent(context: NodeContext) -> NodeResult:
        results = await asyncio.gather(
            *[context.delegate(child_graph(f"kid-{index}", child), {}) for index in range(3)]
        )
        assert all(result.status == RunStatus.SUCCEEDED for result in results)
        return NodeResult(finish=True)

    async def release() -> None:
        await asyncio.sleep(0)
        gate.set()

    runtime, events, _ = make_runtime()
    release_task = asyncio.create_task(release())
    before = background_tasks()
    result = await runtime.run(parent_graph(parent))
    await release_task

    assert result.status == RunStatus.SUCCEEDED
    assert len(seen_ids) == 3
    assert len(seen_traces) == 3
    assert seen_ids.isdisjoint(seen_traces)
    assert result.run_id not in seen_ids
    assert not (background_tasks() - before - {release_task})
    parent_events = await events.list(result.run_id)
    delegated = [event for event in parent_events if event.type == EventType.AGENT_DELEGATED]
    joined = [event for event in parent_events if event.type == EventType.AGENT_JOINED]
    assert len(delegated) == len(joined) == 3
    delegated_ids = {event.payload["child_run_id"] for event in delegated}
    joined_ids = {event.payload["child_run_id"] for event in joined}
    assert delegated_ids == joined_ids
    assert len(delegated_ids) == 3
    for child_id in delegated_ids:
        child_events = await events.list(UUID(str(child_id)))
        assert child_events[0].parent_run_id == result.run_id
        assert child_events[0].trace_id == UUID(
            str(
                next(
                    event for event in delegated if event.payload["child_run_id"] == child_id
                ).payload["child_trace_id"]
            )
        )


@pytest.mark.asyncio
async def test_plan_batch_failure_cancels_sibling_child_without_leak() -> None:
    from admires_harness.runtime import Plan, PlanStep, execute_plan

    slow_started = asyncio.Event()

    async def slow_node(context: NodeContext) -> NodeResult:
        slow_started.set()
        await asyncio.sleep(5)
        return NodeResult(finish=True)  # pragma: no cover

    async def broken_node(context: NodeContext) -> NodeResult:
        return NodeResult(finish=True)  # pragma: no cover

    slow_graph = child_graph("slow-worker", slow_node)
    broken_graph = child_graph("broken-worker", broken_node)

    class _BrokenChildStore(InMemoryEventStore):
        async def append(self, event: RunEvent) -> None:  # type: ignore[override]
            if (
                event.type == EventType.RUN_CREATED
                and event.payload.get("graph_name") == "broken-worker"
            ):
                await slow_started.wait()
                raise RuntimeError("boom from child store")
            await super().append(event)

    plan = Plan(
        objective="fan out",
        steps=(
            PlanStep(id="slow", objective="slow", agent="slow"),
            PlanStep(id="broken", objective="broken", agent="broken"),
        ),
    )

    async def supervisor(context: NodeContext) -> NodeResult:
        return await execute_plan(context, plan, {"slow": slow_graph, "broken": broken_graph})

    events = _BrokenChildStore()
    checkpoints = InMemoryCheckpointStore()
    runtime = GraphRuntime(event_store=events, checkpoint_store=checkpoints)
    before = background_tasks()
    result = await runtime.run(
        parent_graph(supervisor),
        budget=RunBudget(max_steps=10, max_tokens=100, max_cost_usd=1, timeout_seconds=5),
    )

    assert result.status == RunStatus.FAILED
    assert "boom from child store" in (result.error or "")
    assert not (background_tasks() - before)
    parent_events = await events.list(result.run_id)
    delegated = [event for event in parent_events if event.type == EventType.AGENT_DELEGATED]
    joined = [event for event in parent_events if event.type == EventType.AGENT_JOINED]
    assert len(delegated) == 2
    assert len(joined) == 2
    statuses = {event.payload["status"] for event in joined}
    assert statuses == {"cancelled", "failed"}
    cancelled_join = next(event for event in joined if event.payload["status"] == "cancelled")
    slow_id = UUID(str(cancelled_join.payload["child_run_id"]))
    assert (await checkpoints.latest(slow_id)).status == RunStatus.CANCELLED


@pytest.mark.asyncio
async def test_resume_does_not_double_count_joined_child_usage() -> None:
    async def child(context: NodeContext) -> NodeResult:
        return NodeResult(
            finish=True,
            usage=ModelUsage(input_tokens=4, output_tokens=1, cost_usd=0.02),
            model_calls=1,
        )

    async def gate(context: NodeContext) -> NodeResult:
        if not context.state.get("approved"):
            await context.delegate(
                child_graph("kid", child),
                {},
                budget=RunBudget(max_steps=1, max_tokens=5, max_cost_usd=0.02, timeout_seconds=1),
            )
            return NodeResult(update={"asked": True}, pause=RunStatus.WAITING_APPROVAL)
        return NodeResult(update={"done": True}, finish=True)

    gated_graph = StateGraph("gate").add_node("gate", gate).set_entry_point("gate").compile()
    runtime, events, _ = make_runtime()

    paused = await runtime.run(
        gated_graph,
        budget=RunBudget(max_steps=10, max_tokens=50, max_cost_usd=1, timeout_seconds=5),
    )
    assert paused.status == RunStatus.WAITING_APPROVAL
    assert (paused.input_tokens, paused.output_tokens) == (4, 1)

    resumed = await runtime.resume(gated_graph, paused.run_id, state_update={"approved": True})
    assert resumed.status == RunStatus.SUCCEEDED
    # The joined child usage is rebuilt once from agent.joined, not duplicated.
    assert (resumed.input_tokens, resumed.output_tokens) == (4, 1)
    assert resumed.cost_usd == pytest.approx(0.02)
    assert resumed.model_calls == 1
    # One folded child step plus the two gate node executions, counted once each.
    assert resumed.steps == 3
    parent_events = await events.list(resumed.run_id)
    assert [event.type for event in parent_events].count(EventType.AGENT_JOINED) == 1


@pytest.mark.asyncio
async def test_repeated_delegation_preserves_budget_for_later_children() -> None:
    completed = 0

    async def child(context: NodeContext) -> NodeResult:
        nonlocal completed
        completed += 1
        return NodeResult(
            finish=True,
            usage=ModelUsage(input_tokens=1, output_tokens=0),
        )

    async def parent(context: NodeContext) -> NodeResult:
        for index in range(3):
            result = await context.delegate(
                child_graph(f"kid-{index}", child),
                {},
                budget=RunBudget(max_steps=1, max_tokens=2, max_cost_usd=0.1, timeout_seconds=1),
            )
            assert result.status == RunStatus.SUCCEEDED
        return NodeResult(finish=True)

    runtime, events, _ = make_runtime()
    result = await runtime.run(
        parent_graph(parent),
        budget=RunBudget(max_steps=10, max_tokens=10, max_cost_usd=1, timeout_seconds=5),
    )

    assert result.status == RunStatus.SUCCEEDED
    assert completed == 3
    assert result.input_tokens == 3
    parent_events = await events.list(result.run_id)
    assert [event.type for event in parent_events].count(EventType.AGENT_DELEGATED) == 3
    assert [event.type for event in parent_events].count(EventType.AGENT_JOINED) == 3
