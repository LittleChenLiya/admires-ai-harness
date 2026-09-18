"""Focused coverage for the custom conditional-graph walkthrough.

The tests import the public example module directly so they exercise the same code the
guide tells readers to run; they rely solely on public runtime contracts.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from admires_harness.adapters.models import MockModelProvider
from admires_harness.contracts import RunStatus
from admires_harness.runtime import (
    END,
    GraphDefinitionError,
    GraphRuntime,
    InMemoryCheckpointStore,
    InMemoryEventStore,
    NodeContext,
    NodeResult,
    StateGraph,
)

_EXAMPLE_PATH = Path(__file__).resolve().parents[3] / "examples" / "custom_graph_triage.py"
_SPEC = importlib.util.spec_from_file_location("custom_graph_triage_example", _EXAMPLE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


URGENT_TICKET = "[T-1] Production checkout is down for every customer in EU-West."
ROUTINE_TICKET = "[T-2] Question: how do I export my runs to a CSV file?"
AMBIGUOUS_TICKET = "[T-3] Please look at my account when possible."


def _runtime() -> GraphRuntime:
    return GraphRuntime(
        event_store=InMemoryEventStore(),
        checkpoint_store=InMemoryCheckpointStore(),
        dependencies={"model": MockModelProvider(responder=example.triage_responder)},
    )


@pytest.mark.asyncio
async def test_urgent_ticket_takes_escalation_branch() -> None:
    result = await _runtime().run(example.build_graph().compile(), {"input": URGENT_TICKET})

    assert result.status is RunStatus.SUCCEEDED
    assert result.state["priority"] == "urgent"
    assert result.state["route"] == "escalate"
    assert result.output == f"ESCALATED (urgent): {URGENT_TICKET}"
    assert result.model_calls == 1
    assert result.steps == 2


@pytest.mark.asyncio
async def test_routine_ticket_takes_reply_branch() -> None:
    result = await _runtime().run(example.build_graph().compile(), {"input": ROUTINE_TICKET})

    assert result.status is RunStatus.SUCCEEDED
    assert result.state["priority"] == "routine"
    assert result.state["route"] == "reply"
    assert result.output is not None and result.output.startswith("ROUTINE REPLY:")
    assert result.steps == 2


@pytest.mark.asyncio
async def test_ambiguous_ticket_fails_closed_to_escalation() -> None:
    result = await _runtime().run(example.build_graph().compile(), {"input": AMBIGUOUS_TICKET})

    assert result.status is RunStatus.SUCCEEDED
    assert result.state["priority"] == "urgent"
    assert result.state["route"] == "escalate"


def test_route_predicate_matches_each_branch() -> None:
    assert example.is_urgent({"priority": "urgent"}) is True
    assert example.is_urgent({"priority": "routine"}) is False
    assert example.is_urgent({}) is False


def test_compile_rejects_edge_targeting_unknown_node() -> None:
    async def noop(context: NodeContext) -> NodeResult:
        del context
        return NodeResult()

    with pytest.raises(GraphDefinitionError, match="unknown node"):
        (
            StateGraph("broken-route")
            .add_node("triage", noop)
            .set_entry_point("triage")
            .add_edge("triage", "missing", condition=example.is_urgent)
            .add_edge("triage", END)
            .compile()
        )


@pytest.mark.asyncio
async def test_run_fails_when_branch_node_signals_undeclared_destination() -> None:
    async def rogue(context: NodeContext) -> NodeResult:
        del context
        return NodeResult(goto="nowhere")

    graph = (
        StateGraph("rogue-route")
        .add_node("triage", rogue)
        .set_entry_point("triage")
        .add_edge("triage", END)
        .compile()
    )

    result = await _runtime().run(graph, {"input": URGENT_TICKET})

    assert result.status is RunStatus.FAILED
    assert result.error is not None and "undeclared destination" in result.error
