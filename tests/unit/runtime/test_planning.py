from __future__ import annotations

import pytest
from pydantic import ValidationError

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
