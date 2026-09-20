"""Offline coverage for the executable custom-graph example."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from admires_harness.contracts import EventType, Message, RunStatus
from admires_harness.runtime import (
    END,
    GraphDefinitionError,
    GraphRuntime,
    InMemoryCheckpointStore,
    InMemoryEventStore,
    StateGraph,
)

EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "examples" / "custom_graph.py"


def _load_example() -> Any:
    spec = importlib.util.spec_from_file_location("custom_graph_example", EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


example = _load_example()


async def _run_with_events(question: str) -> tuple[Any, list[str]]:
    events = InMemoryEventStore()
    runtime = GraphRuntime(
        event_store=events,
        checkpoint_store=InMemoryCheckpointStore(),
        dependencies={"model": example.build_model()},
    )
    result = await runtime.run(example.build_graph(), {"question": question})
    node_order = [
        str(event.node_id)
        for event in await events.list(result.run_id)
        if event.type is EventType.NODE_STARTED
    ]
    return result, node_order


async def test_technical_question_takes_technical_branch() -> None:
    result, node_order = await _run_with_events("How do I fix a Python KeyError?")

    assert result.status is RunStatus.SUCCEEDED
    assert result.state["category"] == "technical"
    assert result.output == "Technical answer: reproduce, isolate the failing line, then patch it."
    assert node_order == ["classify", "technical"]
    assert result.steps == 2
    assert result.model_calls == 2


async def test_general_question_takes_fallback_branch() -> None:
    result, node_order = await _run_with_events("What is the meaning of life?")

    assert result.status is RunStatus.SUCCEEDED
    assert result.state["category"] == "general"
    assert result.output == "General answer: here is a friendly, high-level explanation."
    assert node_order == ["classify", "general"]
    assert result.steps == 2


async def test_model_failure_fails_the_run() -> None:
    class FailingModel:
        name = "failing-mock"

        async def complete(
            self,
            messages: Sequence[Message],
            *,
            tools: Sequence[Mapping[str, Any]] = (),
        ) -> Any:
            raise RuntimeError("deterministic model failure")

        async def stream(
            self,
            messages: Sequence[Message],
            *,
            tools: Sequence[Mapping[str, Any]] = (),
        ) -> AsyncIterator[str]:
            raise RuntimeError("deterministic model failure")
            yield ""  # pragma: no cover

    result = await example.run("How do I fix a Python KeyError?", model=FailingModel())

    assert result.status is RunStatus.FAILED
    assert result.error is not None
    assert "deterministic model failure" in result.error


def test_invalid_custom_definitions_still_fail_at_compile() -> None:
    async def noop(context: Any) -> None:
        return None

    with pytest.raises(GraphDefinitionError, match="unknown node"):
        (
            StateGraph("broken")
            .add_node("a", noop)
            .set_entry_point("a")
            .add_edge("a", "missing")
            .compile()
        )
    with pytest.raises(GraphDefinitionError, match="unreachable"):
        (
            StateGraph("broken")
            .add_node("a", noop)
            .add_node("b", noop)
            .set_entry_point("a")
            .add_edge("a", END)
            .compile()
        )
