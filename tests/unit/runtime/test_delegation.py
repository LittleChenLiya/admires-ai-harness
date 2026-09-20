from __future__ import annotations

import asyncio
from uuid import UUID

import pytest

from admires_harness.contracts import EventType, ModelUsage, RunBudget, RunEvent, RunStatus
from admires_harness.runtime import (
    END,
    GraphRuntime,
    InMemoryCheckpointStore,
    InMemoryEventStore,
    NodeResult,
    StateGraph,
)


def make_runtime(
    dependencies: dict[str, object] | None = None,
) -> tuple[GraphRuntime, InMemoryEventStore, InMemoryCheckpointStore]:
    events = InMemoryEventStore()
    checkpoints = InMemoryCheckpointStore()
    return (
        GraphRuntime(event_store=events, checkpoint_store=checkpoints, dependencies=dependencies),
        events,
        checkpoints,
    )


def single_node_graph(name, node):  # type: ignore[no-untyped-def]
    return StateGraph(name).add_node("a", node).set_entry_point("a").add_edge("a", END).compile()


def delegation_pairs(events: list[RunEvent]) -> list[tuple[RunEvent, RunEvent]]:
    """Pair every agent.delegated with exactly one later agent.joined per child."""

    delegated = [event for event in events if event.type == EventType.AGENT_DELEGATED]
    joined = [event for event in events if event.type == EventType.AGENT_JOINED]
    assert len(delegated) == len(joined)
    pairs = []
    for start in delegated:
        matches = [
            event
            for event in joined
            if event.payload["child_run_id"] == start.payload["child_run_id"]
            and event.sequence > start.sequence
        ]
        assert len(matches) == 1
        pairs.append((start, matches[0]))
    return pairs


def live_extra_tasks(before: set[asyncio.Task]) -> set[asyncio.Task]:  # type: ignore[type-arg]
    return {task for task in asyncio.all_tasks() if task not in before and not task.done()}


@pytest.mark.asyncio
async def test_child_run_has_independent_identity_and_paired_events() -> None:
    before = set(asyncio.all_tasks())

    async def child(context):  # type: ignore[no-untyped-def]
        return NodeResult(
            finish=True,
            output="child-done",
            usage=ModelUsage(input_tokens=3, output_tokens=2, cost_usd=0.05),
            model_calls=1,
        )

    child_graph = single_node_graph("child", child)

    async def parent(context):  # type: ignore[no-untyped-def]
        result = await context.delegate(
            child_graph,
            {"seed": 1},
            label="only-child",
            budget=RunBudget(max_steps=2, max_tokens=10, max_cost_usd=0.05, timeout_seconds=5),
        )
        assert result.status == RunStatus.SUCCEEDED
        assert result.output == "child-done"
        assert result.run_id != context.run_id
        assert result.trace_id != context.trace_id
        assert result.parent_run_id == context.run_id
        return NodeResult(finish=True, output="parent-done")

    parent_graph = single_node_graph("parent", parent)
    runtime, events, _ = make_runtime()
    result = await runtime.run(parent_graph, budget=RunBudget(max_steps=5, max_cost_usd=0.10))

    assert result.status == RunStatus.SUCCEEDED, result.error
    assert result.output == "parent-done"
    assert (result.steps, result.model_calls) == (2, 1)
    assert (result.input_tokens, result.output_tokens, result.cost_usd) == (3, 2, 0.05)

    history = await events.list(result.run_id)
    pairs = delegation_pairs(history)
    assert len(pairs) == 1
    delegated, joined = pairs[0]
    assert delegated.payload["graph_name"] == "child"
    assert delegated.payload["label"] == "only-child"
    assert delegated.payload["capabilities"] == []
    assert joined.payload["status"] == RunStatus.SUCCEEDED.value
    assert joined.payload["label"] == "only-child"
    assert joined.payload["metrics"]["cost_usd"] == 0.05

    child_run_id = UUID(delegated.payload["child_run_id"])
    child_trace_id = UUID(delegated.payload["child_trace_id"])
    assert child_run_id != result.run_id
    assert child_trace_id != result.trace_id

    child_events = await events.list(child_run_id)
    assert child_events
    assert all(event.trace_id == child_trace_id for event in child_events)
    assert all(event.parent_run_id == result.run_id for event in child_events)
    assert child_events[0].type == EventType.RUN_CREATED
    assert child_events[0].payload["budget"]["max_cost_usd"] == 0.05
    assert child_events[-1].type == EventType.RUN_COMPLETED
    assert live_extra_tasks(before) == set()


@pytest.mark.asyncio
async def test_concurrent_child_runs_are_isolated_and_counted_once() -> None:
    before = set(asyncio.all_tasks())
    barrier = asyncio.Barrier(2)

    async def child(context):  # type: ignore[no-untyped-def]
        await barrier.wait()
        return NodeResult(
            finish=True,
            output=context.state["who"],
            usage=ModelUsage(input_tokens=1, output_tokens=1, cost_usd=0.02),
        )

    child_graph = single_node_graph("child", child)

    async def parent(context):  # type: ignore[no-untyped-def]
        first, second = await asyncio.gather(
            context.delegate(
                child_graph,
                {"who": "a"},
                label="first",
                budget=RunBudget(max_steps=2, max_tokens=20, max_cost_usd=0.05),
            ),
            context.delegate(
                child_graph,
                {"who": "b"},
                label="second",
                budget=RunBudget(max_steps=2, max_tokens=20, max_cost_usd=0.05),
            ),
        )
        assert {first.output, second.output} == {"a", "b"}
        assert first.run_id != second.run_id
        assert first.trace_id != second.trace_id
        return NodeResult(finish=True)

    parent_graph = single_node_graph("parent", parent)
    runtime, events, _ = make_runtime()
    result = await runtime.run(
        parent_graph, budget=RunBudget(max_steps=6, max_tokens=50, max_cost_usd=0.10)
    )

    assert result.status == RunStatus.SUCCEEDED
    assert result.cost_usd == 0.04
    assert result.steps == 3

    history = await events.list(result.run_id)
    pairs = delegation_pairs(history)
    assert len(pairs) == 2
    child_ids = {pair[0].payload["child_run_id"] for pair in pairs}
    assert len(child_ids) == 2
    assert {pair[0].payload["label"] for pair in pairs} == {"first", "second"}
    for delegated, joined in pairs:
        assert joined.payload["status"] == RunStatus.SUCCEEDED.value
        assert delegated.payload["child_trace_id"] != str(result.trace_id)
    assert live_extra_tasks(before) == set()


@pytest.mark.asyncio
async def test_child_receives_only_explicitly_delegated_capabilities() -> None:
    observed: list[set[str]] = []

    async def child(context):  # type: ignore[no-untyped-def]
        observed.append(set(context.dependencies))
        return NodeResult(finish=True)

    child_graph = single_node_graph("child", child)

    async def parent(context):  # type: ignore[no-untyped-def]
        await context.delegate(child_graph, {}, capabilities=["allowed"])
        return NodeResult(finish=True)

    parent_graph = single_node_graph("parent", parent)
    runtime, _, _ = make_runtime({"allowed": object(), "secret": object()})
    result = await runtime.run(parent_graph)

    assert result.status == RunStatus.SUCCEEDED
    assert observed == [{"allowed"}]


@pytest.mark.asyncio
async def test_unknown_child_capability_fails_closed_without_delegation() -> None:
    async def child(context):  # type: ignore[no-untyped-def]
        return NodeResult(finish=True)

    child_graph = single_node_graph("child", child)

    async def parent(context):  # type: ignore[no-untyped-def]
        await context.delegate(child_graph, {}, capabilities=["missing"])
        return NodeResult(finish=True)

    parent_graph = single_node_graph("parent", parent)
    runtime, events, _ = make_runtime({"allowed": object()})
    result = await runtime.run(parent_graph)

    assert result.status == RunStatus.FAILED
    assert "unknown child capabilities" in (result.error or "")
    history = await events.list(result.run_id)
    assert EventType.AGENT_DELEGATED not in {event.type for event in history}
    assert EventType.AGENT_JOINED not in {event.type for event in history}


@pytest.mark.asyncio
async def test_child_budget_is_capped_by_parent_remaining_budget() -> None:
    child_budgets: list[RunBudget] = []

    async def child(context):  # type: ignore[no-untyped-def]
        child_budgets.append(context.budget)
        return NodeResult(finish=True)

    child_graph = single_node_graph("child", child)

    async def parent(context):  # type: ignore[no-untyped-def]
        await context.delegate(child_graph, {})
        with pytest.raises(Exception, match="remaining budget"):
            await context.delegate(
                child_graph,
                {},
                budget=RunBudget(max_steps=100, max_tokens=100, max_cost_usd=1, timeout_seconds=5),
            )
        return NodeResult(finish=True)

    parent_graph = single_node_graph("parent", parent)
    runtime, events, _ = make_runtime()
    result = await runtime.run(
        parent_graph,
        budget=RunBudget(max_steps=4, max_tokens=50, max_cost_usd=0.25, timeout_seconds=30),
    )

    assert result.status == RunStatus.SUCCEEDED
    assert len(child_budgets) == 1
    defaulted = child_budgets[0]
    assert defaulted.max_steps <= 4
    assert defaulted.max_tokens <= 50
    assert defaulted.max_cost_usd <= 0.25
    assert defaulted.timeout_seconds <= 30

    history = await events.list(result.run_id)
    pairs = delegation_pairs(history)
    assert len(pairs) == 1


@pytest.mark.asyncio
async def test_child_budget_exhaustion_is_a_stable_parent_failure() -> None:
    async def child(context):  # type: ignore[no-untyped-def]
        return NodeResult(finish=True, usage=ModelUsage(cost_usd=0.06))

    child_graph = single_node_graph("child", child)

    async def parent(context):  # type: ignore[no-untyped-def]
        await context.delegate(
            child_graph,
            {},
            budget=RunBudget(max_steps=2, max_tokens=20, max_cost_usd=0.06),
        )
        # The first child folded 0.06 into the parent, so this reservation
        # exceeds the remaining budget and no second child run is created.
        await context.delegate(
            child_graph,
            {},
            budget=RunBudget(max_steps=2, max_tokens=20, max_cost_usd=0.06),
        )
        return NodeResult(finish=True)

    parent_graph = single_node_graph("parent", parent)
    runtime, events, _ = make_runtime()
    result = await runtime.run(parent_graph, budget=RunBudget(max_cost_usd=0.10))

    assert result.status == RunStatus.BUDGET_EXCEEDED
    # The joined child is charged exactly once; the rejected reservation never
    # consumed budget.
    assert result.cost_usd == 0.06
    history = await events.list(result.run_id)
    pairs = delegation_pairs(history)
    assert len(pairs) == 1
    assert pairs[0][1].payload["status"] == RunStatus.SUCCEEDED.value
    assert pairs[0][1].payload["metrics"]["cost_usd"] == 0.06


@pytest.mark.asyncio
async def test_child_failure_is_data_and_does_not_finish_parent() -> None:
    async def broken(context):  # type: ignore[no-untyped-def]
        raise ValueError("child boom")

    child_graph = single_node_graph("child", broken)

    async def parent(context):  # type: ignore[no-untyped-def]
        result = await context.delegate(child_graph, {}, label="doomed")
        assert result.status == RunStatus.FAILED
        assert result.error == "child boom"
        return NodeResult(finish=True, output="recovered")

    parent_graph = single_node_graph("parent", parent)
    runtime, events, _ = make_runtime()
    result = await runtime.run(parent_graph)

    assert result.status == RunStatus.SUCCEEDED
    assert result.output == "recovered"
    history = await events.list(result.run_id)
    pairs = delegation_pairs(history)
    assert len(pairs) == 1
    assert pairs[0][1].payload["status"] == RunStatus.FAILED.value

    child_run_id = UUID(pairs[0][0].payload["child_run_id"])
    child_events = await events.list(child_run_id)
    assert child_events[-1].type == EventType.RUN_FAILED


@pytest.mark.asyncio
async def test_child_timeout_converges_and_parent_continues() -> None:
    async def slow(context):  # type: ignore[no-untyped-def]
        await asyncio.sleep(5)
        return NodeResult(finish=True)

    child_graph = single_node_graph("child", slow)

    async def parent(context):  # type: ignore[no-untyped-def]
        result = await context.delegate(
            child_graph,
            {},
            budget=RunBudget(max_steps=2, max_tokens=10, max_cost_usd=0.1, timeout_seconds=0.01),
        )
        assert result.status == RunStatus.TIMED_OUT
        return NodeResult(finish=True)

    parent_graph = single_node_graph("parent", parent)
    runtime, events, _ = make_runtime()
    result = await runtime.run(parent_graph)

    assert result.status == RunStatus.SUCCEEDED
    history = await events.list(result.run_id)
    pairs = delegation_pairs(history)
    assert len(pairs) == 1
    assert pairs[0][1].payload["status"] == RunStatus.TIMED_OUT.value


@pytest.mark.asyncio
async def test_cancelling_parent_cancels_child_and_pairs_terminal_join() -> None:
    before = set(asyncio.all_tasks())
    child_entered = asyncio.Event()

    async def child(context):  # type: ignore[no-untyped-def]
        child_entered.set()
        await asyncio.Event().wait()
        return NodeResult(finish=True)

    child_graph = single_node_graph("child", child)

    async def parent(context):  # type: ignore[no-untyped-def]
        await context.delegate(child_graph, {})
        return NodeResult(finish=True)

    parent_graph = single_node_graph("parent", parent)
    runtime, events, _ = make_runtime()
    parent_run_id = UUID(int=0)
    task = asyncio.create_task(runtime.run(parent_graph, run_id=parent_run_id))
    await child_entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    # The parent stream pairs the delegation with a cancelled join and ends in
    # its own cancelled terminal status; the child stream converges to cancelled
    # on its own identity.
    history = await events.list(parent_run_id)
    pairs = delegation_pairs(history)
    assert len(pairs) == 1
    assert pairs[0][1].payload["status"] == RunStatus.CANCELLED.value
    assert history[-1].type == EventType.RUN_STATUS
    assert history[-1].payload["status"] == RunStatus.CANCELLED.value

    child_run_id = UUID(pairs[0][0].payload["child_run_id"])
    child_trace_id = UUID(pairs[0][0].payload["child_trace_id"])
    child_events = await events.list(child_run_id)
    assert child_events
    assert all(event.trace_id == child_trace_id for event in child_events)
    assert all(event.parent_run_id == parent_run_id for event in child_events)
    assert child_events[-1].type == EventType.RUN_STATUS
    assert child_events[-1].payload["status"] == RunStatus.CANCELLED.value
    assert live_extra_tasks(before) == set()


@pytest.mark.asyncio
async def test_repeated_delegation_releases_budget_and_mints_fresh_ids() -> None:
    async def child(context):  # type: ignore[no-untyped-def]
        return NodeResult(finish=True, usage=ModelUsage(cost_usd=0.02))

    child_graph = single_node_graph("child", child)

    async def parent(context):  # type: ignore[no-untyped-def]
        first = await context.delegate(child_graph, {}, label="one")
        second = await context.delegate(child_graph, {}, label="two")
        assert first.run_id != second.run_id
        assert first.trace_id != second.trace_id
        return NodeResult(finish=True)

    parent_graph = single_node_graph("parent", parent)
    runtime, events, _ = make_runtime()
    result = await runtime.run(parent_graph, budget=RunBudget(max_cost_usd=0.05))

    assert result.status == RunStatus.SUCCEEDED
    assert result.cost_usd == 0.04
    history = await events.list(result.run_id)
    pairs = delegation_pairs(history)
    assert len(pairs) == 2
    assert len({pair[0].payload["child_run_id"] for pair in pairs}) == 2
    assert len({pair[0].payload["child_trace_id"] for pair in pairs}) == 2


@pytest.mark.asyncio
async def test_resumed_parent_does_not_spend_child_budget_twice() -> None:
    async def child(context):  # type: ignore[no-untyped-def]
        return NodeResult(
            finish=True,
            usage=ModelUsage(input_tokens=2, output_tokens=1, cost_usd=0.05),
        )

    child_graph = single_node_graph("child", child)

    async def parent(context):  # type: ignore[no-untyped-def]
        if not context.state.get("joined"):
            await context.delegate(child_graph, {})
            return NodeResult(update={"joined": True}, pause=RunStatus.PAUSED)
        return NodeResult(finish=True)

    parent_graph = single_node_graph("parent", parent)
    runtime, events, _ = make_runtime()
    paused = await runtime.run(parent_graph, budget=RunBudget(max_cost_usd=0.10))
    assert paused.status == RunStatus.PAUSED
    assert paused.cost_usd == 0.05

    resumed = await runtime.resume(parent_graph, paused.run_id)

    assert resumed.status == RunStatus.SUCCEEDED
    # Child usage is folded exactly once: from the durable agent.joined metrics,
    # not again from the reservation that was released before the pause.
    assert resumed.cost_usd == 0.05
    assert (resumed.input_tokens, resumed.output_tokens) == (2, 1)
    assert resumed.steps == 3
    history = await events.list(paused.run_id)
    pairs = delegation_pairs(history)
    assert len(pairs) == 1
