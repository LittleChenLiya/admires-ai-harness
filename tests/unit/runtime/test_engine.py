from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest

from admires_harness.contracts import (
    Checkpoint,
    EventType,
    ModelUsage,
    RunBudget,
    RunEvent,
    RunStatus,
)
from admires_harness.runtime import (
    END,
    GraphRuntime,
    IncompatibleCheckpointError,
    InMemoryCheckpointStore,
    InMemoryEventStore,
    NodeResult,
    RunAlreadyExistsError,
    RunLeaseActiveError,
    RunNotFoundError,
    StateGraph,
    TerminalRunError,
)


def make_runtime() -> tuple[GraphRuntime, InMemoryEventStore, InMemoryCheckpointStore]:
    events = InMemoryEventStore()
    checkpoints = InMemoryCheckpointStore()
    return (
        GraphRuntime(event_store=events, checkpoint_store=checkpoints),
        events,
        checkpoints,
    )


class _StaleReadCheckpointStore(InMemoryCheckpointStore):
    """Let two resume callers deterministically observe the same revision."""

    def __init__(self) -> None:
        super().__init__()
        self.read_barrier: asyncio.Barrier | None = None

    async def latest(self, run_id: UUID) -> Checkpoint | None:
        checkpoint = await super().latest(run_id)
        if self.read_barrier is not None:
            await self.read_barrier.wait()
        return checkpoint


@pytest.mark.asyncio
async def test_runs_graph_and_records_metrics_and_monotonic_events() -> None:
    async def first(context):  # type: ignore[no-untyped-def]
        assert context.state == {}
        return NodeResult(
            update={"value": 2},
            usage=ModelUsage(input_tokens=5, output_tokens=3, cost_usd=0.01),
            model_calls=1,
        )

    def second(context):  # type: ignore[no-untyped-def]
        return NodeResult(output=context.state["value"], tool_calls=1)

    graph = (
        StateGraph("basic")
        .add_node("first", first)
        .add_node("second", second)
        .set_entry_point("first")
        .add_edge("first", "second")
        .add_edge("second", END)
        .compile()
    )
    runtime, events, checkpoints = make_runtime()

    result = await runtime.run(graph)

    assert result.status == RunStatus.SUCCEEDED
    assert result.output == 2
    assert (result.steps, result.model_calls, result.tool_calls) == (2, 1, 1)
    assert (result.input_tokens, result.output_tokens, result.cost_usd) == (5, 3, 0.01)
    history = await events.list(result.run_id)
    assert [event.sequence for event in history] == list(range(len(history)))
    assert history[0].payload["initial_state"] == {}
    assert history[-1].type == EventType.RUN_COMPLETED
    assert (await checkpoints.latest(result.run_id)).status == RunStatus.SUCCEEDED  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_pause_resume_and_replay_are_durable() -> None:
    async def gate(context):  # type: ignore[no-untyped-def]
        if not context.state.get("approved"):
            return NodeResult(update={"asked": True}, pause=RunStatus.WAITING_APPROVAL)
        return NodeResult(update={"passed": True})

    graph = (
        StateGraph("approval", version="7")
        .add_node("gate", gate)
        .set_entry_point("gate")
        .add_edge("gate", END)
        .compile()
    )
    runtime, _, _ = make_runtime()

    paused = await runtime.run(graph)
    assert paused.status == RunStatus.WAITING_APPROVAL
    assert paused.next_node == "gate"
    snapshot = await runtime.replay(paused.run_id)
    assert snapshot.state == {"asked": True}
    assert snapshot.next_node == "gate"

    completed = await runtime.resume(graph, paused.run_id, state_update={"approved": True})
    assert completed.status == RunStatus.SUCCEEDED
    assert completed.state == {"asked": True, "approved": True, "passed": True}
    assert completed.steps == 2
    with pytest.raises(TerminalRunError):
        await runtime.resume(graph, completed.run_id)


@pytest.mark.asyncio
async def test_concurrent_resume_acquires_lease_before_node_side_effect() -> None:
    executions = 0

    async def gate(context):  # type: ignore[no-untyped-def]
        nonlocal executions
        if not context.state.get("resume"):
            return NodeResult(pause=RunStatus.PAUSED)
        executions += 1
        await asyncio.sleep(0.01)
        return NodeResult(finish=True)

    graph = (
        StateGraph("leased")
        .add_node("gate", gate)
        .set_entry_point("gate")
        .add_edge("gate", END)
        .compile()
    )
    events = InMemoryEventStore()
    checkpoints = _StaleReadCheckpointStore()
    runtime = GraphRuntime(event_store=events, checkpoint_store=checkpoints)
    paused = await runtime.run(graph)
    checkpoints.read_barrier = asyncio.Barrier(2)
    outcomes = await asyncio.gather(
        runtime.resume(graph, paused.run_id, state_update={"resume": True}),
        runtime.resume(graph, paused.run_id, state_update={"resume": True}),
        return_exceptions=True,
    )

    assert executions == 1
    assert sum(isinstance(outcome, RunLeaseActiveError) for outcome in outcomes) == 1
    successful = [outcome for outcome in outcomes if not isinstance(outcome, BaseException)]
    assert len(successful) == 1
    assert successful[0].status == RunStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_resume_rejects_missing_or_incompatible_checkpoint() -> None:
    async def pause(context):  # type: ignore[no-untyped-def]
        return NodeResult(pause=RunStatus.PAUSED)

    graph = StateGraph("one").add_node("a", pause).set_entry_point("a").add_edge("a", END).compile()
    other = StateGraph("two").add_node("a", pause).set_entry_point("a").add_edge("a", END).compile()
    runtime, _, _ = make_runtime()

    with pytest.raises(RunNotFoundError):
        await runtime.resume(graph, uuid4())
    paused = await runtime.run(graph)
    with pytest.raises(IncompatibleCheckpointError):
        await runtime.resume(other, paused.run_id)


@pytest.mark.asyncio
async def test_failure_and_loop_guard_return_controlled_results() -> None:
    async def broken(context):  # type: ignore[no-untyped-def]
        raise ValueError("boom")

    broken_graph = StateGraph("broken").add_node("a", broken).set_entry_point("a").compile()
    runtime, events, _ = make_runtime()
    failed = await runtime.run(broken_graph)
    assert failed.status == RunStatus.FAILED
    assert failed.error == "boom"
    assert (await events.list(failed.run_id))[-1].type == EventType.RUN_FAILED

    async def repeat(context):  # type: ignore[no-untyped-def]
        return NodeResult(goto="again", action_key="same")

    loop = (
        StateGraph("loop")
        .add_node("again", repeat)
        .set_entry_point("again")
        .add_edge("again", "again")
        .compile()
    )
    guarded = await runtime.run(loop, budget=RunBudget(max_repeated_actions=2))
    assert guarded.status == RunStatus.BUDGET_EXCEEDED
    assert guarded.steps == 3


@pytest.mark.asyncio
async def test_existing_run_id_is_rejected() -> None:
    async def done(context):  # type: ignore[no-untyped-def]
        return NodeResult(finish=True)

    graph = StateGraph("done").add_node("a", done).set_entry_point("a").compile()
    runtime, _, _ = make_runtime()
    run_id = uuid4()
    await runtime.run(graph, run_id=run_id)
    with pytest.raises(RunAlreadyExistsError):
        await runtime.run(graph, run_id=run_id)


@pytest.mark.asyncio
async def test_live_stream_follows_run_to_terminal_event() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def wait(context):  # type: ignore[no-untyped-def]
        entered.set()
        await release.wait()
        return NodeResult(finish=True)

    graph = StateGraph("stream").add_node("a", wait).set_entry_point("a").compile()
    runtime, _, _ = make_runtime()
    run_id = uuid4()
    task = asyncio.create_task(runtime.run(graph, run_id=run_id))
    await entered.wait()

    async def collect():  # type: ignore[no-untyped-def]
        return [event async for event in runtime.events(run_id, follow=True)]

    collector = asyncio.create_task(collect())
    await asyncio.sleep(0)
    release.set()
    await task
    streamed = await collector
    assert streamed[0].type == EventType.RUN_CREATED
    assert streamed[-1].type == EventType.RUN_COMPLETED


@pytest.mark.asyncio
async def test_mapping_result_custom_event_and_nested_state_are_isolated() -> None:
    nested = {"items": ["original"]}

    async def mutate_copy(context):  # type: ignore[no-untyped-def]
        items = context.state["nested"]["items"]
        items.append("node-only")
        await context.emit(EventType.LOG, {"message": "custom"})
        return {"done": True}

    graph = (
        StateGraph("isolation")
        .add_node("a", mutate_copy)
        .set_entry_point("a")
        .add_edge("a", END)
        .compile()
    )
    runtime, events, _ = make_runtime()
    result = await runtime.run(graph, {"nested": nested})

    assert result.state["nested"] == {"items": ["original"]}
    assert nested == {"items": ["original"]}
    assert result.state["done"] is True
    assert EventType.LOG in {event.type for event in await events.list(result.run_id)}


@pytest.mark.asyncio
async def test_timeout_token_and_cost_limits_are_terminal() -> None:
    async def slow(context):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0.05)
        return NodeResult()

    slow_graph = StateGraph("slow").add_node("a", slow).set_entry_point("a").compile()
    runtime, _, _ = make_runtime()
    timed_out = await runtime.run(slow_graph, budget=RunBudget(timeout_seconds=0.001))
    assert timed_out.status == RunStatus.TIMED_OUT

    async def expensive(context):  # type: ignore[no-untyped-def]
        return NodeResult(
            usage=ModelUsage(input_tokens=2, output_tokens=2, cost_usd=0.5),
            finish=True,
        )

    expensive_graph = (
        StateGraph("expensive").add_node("a", expensive).set_entry_point("a").compile()
    )
    token_limited = await runtime.run(
        expensive_graph, budget=RunBudget(max_tokens=3, max_cost_usd=1)
    )
    assert token_limited.status == RunStatus.BUDGET_EXCEEDED
    cost_limited = await runtime.run(
        expensive_graph, budget=RunBudget(max_tokens=10, max_cost_usd=0.1)
    )
    assert cost_limited.status == RunStatus.BUDGET_EXCEEDED

    async def free(context):  # type: ignore[no-untyped-def]
        return NodeResult(finish=True)

    free_graph = StateGraph("free").add_node("a", free).set_entry_point("a").compile()
    free_run = await runtime.run(free_graph, budget=RunBudget(max_cost_usd=0))
    assert free_run.status == RunStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_events_support_history_cursor_and_missing_replay() -> None:
    async def done(context):  # type: ignore[no-untyped-def]
        return None

    graph = StateGraph("history").add_node("a", done).set_entry_point("a").compile()
    runtime, _, _ = make_runtime()
    result = await runtime.run(graph)
    tail = [event async for event in runtime.events(result.run_id, after=1)]
    assert tail and all(event.sequence > 1 for event in tail)
    followed = [event async for event in runtime.events(result.run_id, follow=True)]
    assert followed[-1].type == EventType.RUN_COMPLETED
    with pytest.raises(RunNotFoundError):
        await runtime.replay(uuid4())


@pytest.mark.asyncio
async def test_replay_does_not_publish_terminal_checkpoint_before_lifecycle_event() -> None:
    runtime, events, checkpoints = make_runtime()
    run_id, trace_id = uuid4(), uuid4()
    await events.append(
        RunEvent(
            run_id=run_id,
            trace_id=trace_id,
            sequence=0,
            type=EventType.RUN_CREATED,
            payload={"graph_name": "partial", "graph_version": "1"},
        )
    )
    await events.append(
        RunEvent(
            run_id=run_id,
            trace_id=trace_id,
            sequence=1,
            type=EventType.RUN_STARTED,
        )
    )
    await checkpoints.save(
        Checkpoint(
            run_id=run_id,
            revision=0,
            graph_name="partial",
            graph_version="1",
            next_node=None,
            status=RunStatus.SUCCEEDED,
            state={"answer": 42},
        ),
        expected_revision=None,
    )
    await events.append(
        RunEvent(
            run_id=run_id,
            trace_id=trace_id,
            sequence=2,
            type=EventType.CHECKPOINT_SAVED,
            payload={"status": "succeeded", "revision": 0},
        )
    )

    snapshot = await runtime.replay(run_id)

    assert snapshot.status is RunStatus.RUNNING
    assert snapshot.state == {"answer": 42}


@pytest.mark.asyncio
async def test_child_is_least_privilege_and_usage_counts_against_parent() -> None:
    async def child(context):  # type: ignore[no-untyped-def]
        assert "secret" not in context.dependencies
        return NodeResult(
            finish=True,
            usage=ModelUsage(input_tokens=3, output_tokens=2, cost_usd=0.09),
            model_calls=1,
        )

    child_graph = StateGraph("child").add_node("a", child).set_entry_point("a").compile()

    async def parent(context):  # type: ignore[no-untyped-def]
        first = await context.delegate(
            child_graph,
            {},
            budget=RunBudget(max_steps=1, max_tokens=5, max_cost_usd=0.09, timeout_seconds=1),
        )
        assert first.status == RunStatus.SUCCEEDED
        await context.delegate(
            child_graph,
            {},
            budget=RunBudget(max_steps=1, max_tokens=5, max_cost_usd=0.09, timeout_seconds=1),
        )
        return NodeResult(finish=True)

    parent_graph = StateGraph("parent").add_node("a", parent).set_entry_point("a").compile()
    runtime, _, _ = make_runtime()
    runtime.dependencies["secret"] = object()
    result = await runtime.run(
        parent_graph,
        budget=RunBudget(max_steps=5, max_tokens=20, max_cost_usd=0.10),
    )

    assert result.status == RunStatus.BUDGET_EXCEEDED
    assert result.cost_usd == 0.09
    assert result.model_calls == 1
