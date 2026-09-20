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


def route_graph() -> StateGraph:
    return (
        StateGraph("route")
        .add_node("a", noop)
        .add_node("b", noop)
        .set_entry_point("a")
        .add_edge("a", "b")
        .add_edge("b", END)
    )


def test_compile_rejects_missing_entry_and_unreachable_nodes() -> None:
    with pytest.raises(GraphDefinitionError, match="entry point"):
        StateGraph("bad").add_node("a", noop).compile()

    graph = StateGraph("bad").add_node("a", noop).add_node("b", noop)
    graph.set_entry_point("a")
    with pytest.raises(GraphDefinitionError, match="unreachable nodes: b"):
        graph.compile()


def test_compile_rejects_empty_graph_and_unknown_entry_point() -> None:
    with pytest.raises(GraphDefinitionError, match="at least one node"):
        StateGraph("empty").compile()

    graph = StateGraph("bad").add_node("a", noop).set_entry_point("ghost")
    with pytest.raises(GraphDefinitionError, match="unknown entry point"):
        graph.compile()


def test_compile_rejects_invalid_end_connections() -> None:
    with pytest.raises(GraphDefinitionError, match="invalid node id"):
        StateGraph("bad").add_node(END, noop)
    with pytest.raises(GraphDefinitionError, match="invalid node id"):
        StateGraph("bad").set_entry_point(END)
    with pytest.raises(GraphDefinitionError, match="invalid node id"):
        StateGraph("bad").add_edge(END, "a")

    graph = StateGraph("bad").add_node("a", noop).set_entry_point("a")
    with pytest.raises(GraphDefinitionError, match="unknown node"):
        graph.add_edge("ghost", "a").compile()
    with pytest.raises(GraphDefinitionError, match="unknown node"):
        StateGraph("bad").add_node("a", noop).set_entry_point("a").add_edge("a", "ghost").compile()


def test_compile_rejects_duplicate_edges_and_fallbacks() -> None:
    graph = StateGraph("dup").add_node("a", noop).add_node("b", noop)
    graph.add_edge("a", "b", condition=lambda state: True)
    with pytest.raises(GraphDefinitionError, match="duplicate edge"):
        graph.add_edge("a", "b", condition=lambda state: False)

    same_condition = lambda state: True  # noqa: E731
    graph = StateGraph("dup").add_node("a", noop).add_node("b", noop)
    graph.add_edge("a", "b", condition=same_condition)
    with pytest.raises(GraphDefinitionError, match="duplicate edge"):
        graph.add_edge("a", "b", condition=same_condition)

    graph = StateGraph("dup").add_node("a", noop).add_node("b", noop)
    with pytest.raises(GraphDefinitionError, match="fallback"):
        graph.add_edge("a", "b").add_edge("a", END)


def test_compile_rejects_conditional_edges_without_fallback() -> None:
    graph = (
        StateGraph("dangling")
        .add_node("a", noop)
        .add_node("b", noop)
        .set_entry_point("a")
        .add_edge("a", "b", condition=lambda state: True)
        .add_edge("b", END)
    )
    with pytest.raises(GraphDefinitionError, match="no fallback route"):
        graph.compile()


def test_compile_accepts_cyclic_graphs_when_reachable() -> None:
    graph = (
        StateGraph("cycle")
        .add_node("a", noop)
        .add_node("b", noop)
        .set_entry_point("a")
        .add_edge("a", "b")
        .add_edge("b", "a", condition=lambda state: not state["done"])
        .add_edge("b", END)
        .compile()
    )
    assert set(graph.nodes) == {"a", "b"}


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
async def test_conditional_edges_are_evaluated_in_declaration_order() -> None:
    graph = (
        StateGraph("priority")
        .add_node("choose", noop)
        .add_node("first", noop)
        .add_node("second", noop)
        .set_entry_point("choose")
        .add_edge("choose", "first", condition=lambda state: True)
        .add_edge("choose", "second", condition=lambda state: True)
        .add_edge("choose", END)
        .add_edge("first", END)
        .add_edge("second", END)
        .compile()
    )

    assert await graph.next_node("choose", {}, NodeResult()) == "first"


@pytest.mark.asyncio
async def test_async_predicate_and_conditional_end_route() -> None:
    async def done(state):  # type: ignore[no-untyped-def]
        return state["done"] is True

    graph = (
        StateGraph("async-route")
        .add_node("a", noop)
        .add_node("b", noop)
        .set_entry_point("a")
        .add_edge("a", END, condition=done)
        .add_edge("a", "b")
        .add_edge("b", END)
        .compile()
    )

    assert await graph.next_node("a", {"done": True}, NodeResult()) is None
    assert await graph.next_node("a", {"done": False}, NodeResult()) == "b"


@pytest.mark.asyncio
async def test_predicate_failure_and_non_bool_result_raise_routing_error() -> None:
    def broken(state):  # type: ignore[no-untyped-def]
        raise ValueError("predicate exploded")

    graph = (
        StateGraph("broken-route")
        .add_node("a", noop)
        .add_node("b", noop)
        .set_entry_point("a")
        .add_edge("a", "b", condition=broken)
        .add_edge("a", END)
        .add_edge("b", END)
        .compile()
    )
    with pytest.raises(GraphRoutingError, match="predicate exploded") as exc_info:
        await graph.next_node("a", {}, NodeResult())
    assert isinstance(exc_info.value.__cause__, ValueError)

    not_bool = (
        StateGraph("not-bool")
        .add_node("a", noop)
        .add_node("b", noop)
        .set_entry_point("a")
        .add_edge("a", "b", condition=lambda state: "yes")
        .add_edge("a", END)
        .add_edge("b", END)
        .compile()
    )
    with pytest.raises(GraphRoutingError, match="expected bool"):
        await not_bool.next_node("a", {}, NodeResult())


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


@pytest.mark.asyncio
async def test_explicit_goto_routes_to_declared_target_and_end() -> None:
    graph = (
        StateGraph("goto")
        .add_node("a", noop)
        .add_node("b", noop)
        .add_node("c", noop)
        .set_entry_point("a")
        .add_edge("a", "b")
        .add_edge("b", "c", condition=lambda state: state["go"] is True)
        .add_edge("b", END)
        .add_edge("c", END)
        .compile()
    )

    assert await graph.next_node("b", {"go": False}, NodeResult(goto="c")) == "c"
    assert await graph.next_node("b", {"go": False}, NodeResult(goto=END)) is None
    assert await graph.next_node("a", {}, NodeResult(finish=True)) is None


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
        .add_edge("a", END)
        .add_edge("b", END)
        .compile()
    )
    state = {"nested": {"items": []}}
    assert await graph.next_node("a", state, NodeResult()) == "b"
    assert state == {"nested": {"items": []}}
