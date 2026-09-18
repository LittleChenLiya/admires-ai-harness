"""Focused coverage for the custom conditional graph walkthrough.

These tests load ``examples/custom_conditional_graph.py`` as a standalone script (the
examples directory is intentionally outside the packaged import tree) and exercise the
public runtime contracts end to end: both conditional outcomes, the fallback edge,
and an invalid-definition / node-failure path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from admires_harness.contracts import EventType, RunStatus
from admires_harness.runtime import (
    GraphDefinitionError,
    GraphRuntime,
    InMemoryCheckpointStore,
    InMemoryEventStore,
    NodeResult,
    StateGraph,
)

EXAMPLE_PATH = Path(__file__).resolve().parents[3] / "examples" / "custom_conditional_graph.py"


def _load_example() -> Any:
    spec = importlib.util.spec_from_file_location("custom_conditional_graph", EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example() -> Any:
    return _load_example()


def test_example_file_exists() -> None:
    assert EXAMPLE_PATH.is_file()


@pytest.mark.asyncio
async def test_positive_input_routes_to_positive_branch(example: Any) -> None:
    outcome = await example.run_triage(example.POSITIVE_INPUT)

    assert outcome.tone == "positive"
    assert outcome.routed_to == "positive"
    assert outcome.output.startswith("[positive branch]")
    assert outcome.steps == 2


@pytest.mark.asyncio
async def test_negative_input_routes_to_negative_branch(example: Any) -> None:
    outcome = await example.run_triage(example.NEGATIVE_INPUT)

    assert outcome.tone == "negative"
    assert outcome.routed_to == "negative"
    assert outcome.output.startswith("[negative branch]")
    assert outcome.steps == 2


@pytest.mark.asyncio
async def test_unclear_input_uses_fallback_clarify_branch(example: Any) -> None:
    outcome = await example.run_triage(example.UNCLEAR_INPUT)

    assert outcome.tone == "unclear"
    assert outcome.routed_to == "clarify"
    assert outcome.output.startswith("[clarify branch]")
    assert outcome.steps == 2


@pytest.mark.asyncio
async def test_routes_are_deterministic_across_repeated_runs(example: Any) -> None:
    first = await example.run_triage(example.NEGATIVE_INPUT)
    second = await example.run_triage(example.NEGATIVE_INPUT)

    assert first == second


@pytest.mark.asyncio
async def test_runtime_emits_node_events_for_the_selected_branch(example: Any) -> None:
    graph = example.build_triage_graph().compile()
    events = InMemoryEventStore()
    runtime = GraphRuntime(
        event_store=events,
        checkpoint_store=InMemoryCheckpointStore(),
        dependencies={"model": example.MockModelProvider(responder=example.triage_responder)},
    )

    result = await runtime.run(graph, {"input": example.POSITIVE_INPUT})

    assert result.status == RunStatus.SUCCEEDED
    history = await events.list(result.run_id)
    started = [event for event in history if event.type == EventType.NODE_STARTED]
    assert len(started) == 2
    node_ids = {
        event.node_id
        for event in history
        if event.type == EventType.NODE_COMPLETED and event.node_id is not None
    }
    assert node_ids == {"classify", "positive_reply"}


@pytest.mark.asyncio
async def test_missing_model_dependency_fails_the_run_without_routing(example: Any) -> None:
    graph = example.build_triage_graph().compile()
    runtime = GraphRuntime(
        event_store=InMemoryEventStore(),
        checkpoint_store=InMemoryCheckpointStore(),
        # Deliberately omit the "model" dependency: classify cannot run.
        dependencies={},
    )

    result = await runtime.run(graph, {"input": example.POSITIVE_INPUT})

    assert result.status == RunStatus.FAILED
    assert result.next_node == "classify"
    assert "model" in (result.error or "")


def test_compile_rejects_graph_with_unreachable_branch(example: Any) -> None:
    # A graph that declares a branch node without any incoming edge must fail
    # structural validation, matching the documented compile-time guarantees.
    broken = (
        StateGraph("broken-triage")
        .add_node("classify", example.classify)
        .add_node("orphan_reply", example.positive_reply)
        .set_entry_point("classify")
    )

    with pytest.raises(GraphDefinitionError, match="unreachable nodes: orphan_reply"):
        broken.compile()


@pytest.mark.asyncio
async def test_walkthrough_builds_a_real_conditional_graph(example: Any) -> None:
    graph = example.build_triage_graph().compile()

    edges = graph.edges["classify"]
    conditions = [edge.condition is not None for edge in edges]
    assert conditions == [True, True, False]
    assert await graph.next_node("classify", {"tone": "negative"}, NodeResult()) == "negative_reply"
    assert await graph.next_node("classify", {"tone": "weird"}, NodeResult()) == ("clarify_request")
