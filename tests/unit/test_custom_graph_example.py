from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from admires_harness.contracts import RunStatus
from admires_harness.runtime import (
    GraphDefinitionError,
    GraphRuntime,
    InMemoryCheckpointStore,
    InMemoryEventStore,
    StateGraph,
)

EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "examples" / "custom_graph.py"
spec = importlib.util.spec_from_file_location("custom_graph_example", EXAMPLE_PATH)
assert spec is not None and spec.loader is not None
custom_graph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom_graph)


@pytest.mark.asyncio
async def test_refund_question_takes_the_escalate_branch() -> None:
    result = await custom_graph.run_question("Where is my refund?")

    assert result.status is RunStatus.SUCCEEDED
    assert result.state["category"] == "refund"
    assert result.state["resolution"] == "escalated"
    assert result.steps == 2
    assert result.model_calls == 1
    assert result.output == "Escalated refund request as ticket REFUND-where-is-my-refund."


@pytest.mark.asyncio
async def test_general_question_takes_the_answer_branch() -> None:
    result = await custom_graph.run_question("How do I change my password?")

    assert result.status is RunStatus.SUCCEEDED
    assert result.state["category"] == "general"
    assert result.state["resolution"] == "answered"
    assert result.steps == 2
    assert result.model_calls == 2
    assert result.output == "Support answer: How do I change my password?"


@pytest.mark.asyncio
async def test_runs_are_deterministic() -> None:
    first = await custom_graph.run_question("Where is my refund?")
    second = await custom_graph.run_question("Where is my refund?")

    assert first.state == second.state
    assert first.output == second.output


def test_invalid_definitions_are_rejected_at_compile_time() -> None:
    with pytest.raises(GraphDefinitionError, match="entry point"):
        StateGraph("broken").add_node("classify", custom_graph.classify).compile()

    graph = (
        StateGraph("broken")
        .add_node("classify", custom_graph.classify)
        .set_entry_point("classify")
        .add_edge("classify", "missing")
    )
    with pytest.raises(GraphDefinitionError, match="unknown node"):
        graph.compile()


@pytest.mark.asyncio
async def test_missing_model_dependency_fails_the_run() -> None:
    runtime = GraphRuntime(
        event_store=InMemoryEventStore(),
        checkpoint_store=InMemoryCheckpointStore(),
    )

    result = await runtime.run(custom_graph.build_graph(), {"input": "Where is my refund?"})

    assert result.status is RunStatus.FAILED
    assert "model" in str(result.error)


def test_main_runs_both_branches_offline(capsys: pytest.CaptureFixture[str]) -> None:
    custom_graph.main()

    out = capsys.readouterr().out
    assert "category=refund" in out
    assert "category=general" in out
