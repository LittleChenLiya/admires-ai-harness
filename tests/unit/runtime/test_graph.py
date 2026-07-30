from __future__ import annotations

import pytest
from pydantic import ValidationError

from admires_harness.contracts import RunStatus
from admires_harness.runtime import (
    END,
    GraphDefinitionError,
    GraphRoutingError,
    NodeResult,
    StateGraph,
)


async def noop(context):  # type: ignore[no-untyped-def]
    return None


def test_compile_rejects_missing_entry_and_unreachable_nodes() -> None:
    with pytest.raises(GraphDefinitionError, match="entry point"):
        StateGraph("bad").add_node("a", noop).compile()

    graph = StateGraph("bad").add_node("a", noop).add_node("b", noop)
    graph.set_entry_point("a")
    with pytest.raises(GraphDefinitionError, match="unreachable nodes: b"):
        graph.compile()


@pytest.mark.asyncio
async def test_conditional_route_precedes_fallback() -> None:
    graph = (
        StateGraph("route")
        .add_node("choose", noop)
        .add_node("yes", noop)
        .add_node("no", noop)
        .set_entry_point("choose")
        .add_edge("choose", "yes", condition=lambda state: state["answer"] is True)
        .add_edge("choose", "no")
        .add_edge("yes", END)
        .add_edge("no", END)
        .compile()
    )

    assert await graph.next_node("choose", {"answer": True}, NodeResult()) == "yes"
    assert await graph.next_node("choose", {"answer": False}, NodeResult()) == "no"


@pytest.mark.asyncio
async def test_explicit_goto_must_be_a_declared_edge() -> None:
    graph = (
        StateGraph("route")
        .add_node("a", noop)
        .add_node("b", noop)
        .set_entry_point("a")
        .add_edge("a", "b")
        .add_edge("b", END)
        .compile()
    )

    with pytest.raises(GraphRoutingError, match="undeclared"):
        await graph.next_node("a", {}, NodeResult(goto=END))


def test_graph_builder_and_node_result_reject_invalid_definitions() -> None:
    with pytest.raises(GraphDefinitionError, match="name"):
        StateGraph(" ")
    with pytest.raises(GraphDefinitionError, match="version"):
        StateGraph("x", version=" ")
    graph = StateGraph("x").add_node("a", noop)
    with pytest.raises(GraphDefinitionError, match="duplicate"):
        graph.add_node("a", noop)
    with pytest.raises(GraphDefinitionError, match="fallback"):
        graph.add_edge("a", END).add_edge("a", END)
    with pytest.raises(GraphDefinitionError, match="condition"):
        StateGraph("condition").add_edge("a", END, condition=True)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="mutually exclusive"):
        NodeResult(finish=True, goto=END)
    with pytest.raises(ValidationError, match="cannot pause"):
        NodeResult(finish=True, pause=RunStatus.PAUSED)
    with pytest.raises(ValidationError, match="pause must"):
        NodeResult(pause=RunStatus.RUNNING)


@pytest.mark.asyncio
async def test_route_predicate_cannot_mutate_nested_runtime_state() -> None:
    def attempts_mutation(state):  # type: ignore[no-untyped-def]
        state["nested"]["items"].append("predicate-only")
        return True

    graph = (
        StateGraph("pure-route")
        .add_node("a", noop)
        .add_node("b", noop)
        .set_entry_point("a")
        .add_edge("a", "b", condition=attempts_mutation)
        .add_edge("b", END)
        .compile()
    )
    state = {"nested": {"items": []}}
    assert await graph.next_node("a", state, NodeResult()) == "b"
    assert state == {"nested": {"items": []}}
