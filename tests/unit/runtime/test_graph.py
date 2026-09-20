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


def test_compile_rejects_empty_graph() -> None:
    with pytest.raises(GraphDefinitionError, match="at least one node"):
        StateGraph("empty").compile()


def test_compile_rejects_missing_entry_and_unreachable_nodes() -> None:
    with pytest.raises(GraphDefinitionError, match="entry point"):
        StateGraph("bad").add_node("a", noop).compile()

    graph = StateGraph("bad").add_node("a", noop).add_node("b", noop)
    graph.set_entry_point("a")
    with pytest.raises(GraphDefinitionError, match="unreachable nodes: b"):
        graph.compile()


def test_compile_rejects_edges_to_or_from_unknown_nodes() -> None:
    unknown_source = StateGraph("bad").add_node("a", noop).set_entry_point("a")
    unknown_source.add_edge("ghost", "a")
    with pytest.raises(GraphDefinitionError, match="edge starts at unknown node"):
        unknown_source.compile()

    unknown_target = StateGraph("bad").add_node("a", noop).set_entry_point("a")
    unknown_target.add_edge("a", "ghost")
    with pytest.raises(GraphDefinitionError, match="edge targets unknown node"):
        unknown_target.compile()


def test_compile_rejects_conditional_edges_without_fallback() -> None:
    graph = (
        StateGraph("bad")
        .add_node("a", noop)
        .add_node("b", noop)
        .set_entry_point("a")
        .add_edge("a", "b", condition=lambda state: True)
        .add_edge("b", END)
    )
    with pytest.raises(GraphDefinitionError, match="no fallback edge"):
        graph.compile()


def test_builder_rejects_invalid_end_connections() -> None:
    with pytest.raises(GraphDefinitionError, match="invalid node id"):
        StateGraph("bad").add_node(END, noop)
    graph = StateGraph("bad").add_node("a", noop)
    with pytest.raises(GraphDefinitionError, match="invalid node id"):
        graph.set_entry_point(END)
    with pytest.raises(GraphDefinitionError, match="invalid node id"):
        graph.add_edge(END, "a")


def test_builder_rejects_duplicate_edges() -> None:
    graph = StateGraph("bad").add_node("a", noop).add_node("b", noop)
    graph.add_edge("a", "b")
    with pytest.raises(GraphDefinitionError, match="duplicate edge"):
        graph.add_edge("a", "b")
    conditional = StateGraph("bad").add_node("a", noop).add_node("b", noop)
    conditional.add_edge("a", "b", condition=lambda state: True)
    with pytest.raises(GraphDefinitionError, match="duplicate edge"):
        conditional.add_edge("a", "b", condition=lambda state: False)


def test_compile_accepts_reachable_cycle_and_rejects_isolated_cycle() -> None:
    cyclic = (
        StateGraph("cyclic")
        .add_node("a", noop)
        .add_node("b", noop)
        .set_entry_point("a")
        .add_edge("a", "b")
        .add_edge("b", "a")
        .compile()
    )
    assert set(cyclic.nodes) == {"a", "b"}

    isolated = (
        StateGraph("isolated")
        .add_node("a", noop)
        .add_node("b", noop)
        .add_node("c", noop)
        .set_entry_point("a")
        .add_edge("a", END)
        .add_edge("b", "c")
        .add_edge("c", "b")
    )
    with pytest.raises(GraphDefinitionError, match="unreachable nodes: b, c"):
        isolated.compile()


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
async def test_first_matching_condition_wins_in_declaration_order() -> None:
    graph = (
        StateGraph("priority")
        .add_node("choose", noop)
        .add_node("first", noop)
        .add_node("second", noop)
        .add_node("other", noop)
        .set_entry_point("choose")
        .add_edge("choose", "first", condition=lambda state: state["n"] > 0)
        .add_edge("choose", "second", condition=lambda state: state["n"] > 10)
        .add_edge("choose", "other")
        .add_edge("first", END)
        .add_edge("second", END)
        .add_edge("other", END)
        .compile()
    )

    assert await graph.next_node("choose", {"n": 50}, NodeResult()) == "first"
    assert await graph.next_node("choose", {"n": 0}, NodeResult()) == "other"


@pytest.mark.asyncio
async def test_async_predicate_is_awaited() -> None:
    async def predicate(state):  # type: ignore[no-untyped-def]
        return state["ready"] is True

    graph = (
        StateGraph("async-route")
        .add_node("a", noop)
        .add_node("b", noop)
        .set_entry_point("a")
        .add_edge("a", "b", condition=predicate)
        .add_edge("a", END)
        .add_edge("b", END)
        .compile()
    )

    assert await graph.next_node("a", {"ready": True}, NodeResult()) == "b"
    assert await graph.next_node("a", {"ready": False}, NodeResult()) is None


@pytest.mark.asyncio
async def test_conditional_route_to_end_terminates() -> None:
    graph = (
        StateGraph("end-route")
        .add_node("a", noop)
        .add_node("b", noop)
        .set_entry_point("a")
        .add_edge("a", END, condition=lambda state: state["done"] is True)
        .add_edge("a", "b")
        .add_edge("b", END)
        .compile()
    )

    assert await graph.next_node("a", {"done": True}, NodeResult()) is None
    assert await graph.next_node("a", {"done": False}, NodeResult()) == "b"


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
async def test_explicit_goto_routes_to_declared_targets() -> None:
    graph = (
        StateGraph("goto")
        .add_node("a", noop)
        .add_node("b", noop)
        .set_entry_point("a")
        .add_edge("a", "b")
        .add_edge("b", END)
        .compile()
    )

    assert await graph.next_node("a", {}, NodeResult(goto="b")) == "b"
    assert await graph.next_node("b", {}, NodeResult(goto=END)) is None
    assert await graph.next_node("a", {}, NodeResult(finish=True)) is None


@pytest.mark.asyncio
async def test_predicate_errors_surface_as_routing_errors_with_cause() -> None:
    def broken(state):  # type: ignore[no-untyped-def]
        raise KeyError("missing-key")

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

    state = {"k": "v"}
    with pytest.raises(GraphRoutingError, match="condition on edge 'a' -> 'b' failed") as excinfo:
        await graph.next_node("a", state, NodeResult())
    assert isinstance(excinfo.value.__cause__, KeyError)
    assert state == {"k": "v"}


@pytest.mark.asyncio
async def test_async_predicate_errors_surface_as_routing_errors() -> None:
    async def broken(state):  # type: ignore[no-untyped-def]
        raise RuntimeError("async boom")

    graph = (
        StateGraph("broken-async-route")
        .add_node("a", noop)
        .add_node("b", noop)
        .set_entry_point("a")
        .add_edge("a", "b", condition=broken)
        .add_edge("a", END)
        .add_edge("b", END)
        .compile()
    )

    with pytest.raises(GraphRoutingError, match="async boom"):
        await graph.next_node("a", {}, NodeResult())


def test_graph_builder_and_node_result_reject_invalid_definitions() -> None:
    with pytest.raises(GraphDefinitionError, match="name"):
        StateGraph(" ")
    with pytest.raises(GraphDefinitionError, match="version"):
        StateGraph("x", version=" ")
    graph = StateGraph("x").add_node("a", noop)
    with pytest.raises(GraphDefinitionError, match="duplicate"):
        graph.add_node("a", noop)
    with pytest.raises(GraphDefinitionError, match="duplicate edge"):
        graph.add_edge("a", END).add_edge("a", END)
    with pytest.raises(GraphDefinitionError, match="fallback"):
        StateGraph("fallback").add_node("a", noop).add_edge("a", END).add_edge("a", "b")
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
