"""Build, compile, and run a custom conditional graph with the deterministic mock model.

The graph classifies a support question with the offline MockModelProvider and then
branches: refund questions are escalated to a deterministic ticket summary, everything
else receives a drafted answer. Run it offline with:

    uv run python examples/custom_graph.py

See docs/custom-graph.md for the step-by-step guide.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping, Sequence
from typing import Any

from admires_harness.adapters.models import MockModelProvider
from admires_harness.contracts import JsonValue, Message, ModelResponse, ModelUsage, Role
from admires_harness.runtime import (
    END,
    CompiledGraph,
    GraphRuntime,
    InMemoryCheckpointStore,
    InMemoryEventStore,
    NodeContext,
    NodeResult,
    RunResult,
    StateGraph,
)

GRAPH_NAME = "support-router"
GRAPH_VERSION = "1"
CATEGORIES = ("refund", "general")


def support_responder(
    messages: Sequence[Message], tools: Sequence[Mapping[str, Any]]
) -> ModelResponse:
    """Deterministic rule-based replies: no network, credentials, or wall clock."""

    del tools
    prompt = next(
        (message.content for message in reversed(messages) if message.role == Role.USER), ""
    )
    if prompt.startswith("Classify:"):
        content = "refund" if "refund" in prompt.casefold() else "general"
    else:
        content = f"Support answer: {prompt}"
    return ModelResponse(
        content=content,
        model="mock-support-v1",
        usage=ModelUsage(
            input_tokens=sum(len(message.content.split()) for message in messages),
            output_tokens=len(content.split()),
        ),
    )


def _question(state: Mapping[str, JsonValue]) -> str:
    value = state.get("input", "")
    return value if isinstance(value, str) else ""


def _model(context: NodeContext) -> MockModelProvider:
    model = context.dependencies.get("model")
    if not isinstance(model, MockModelProvider):
        raise RuntimeError("the support-router graph requires a 'model' dependency")
    return model


async def classify(context: NodeContext) -> NodeResult:
    """Ask the mock model for a one-word category and store it in state."""

    response = await _model(context).complete(
        [
            Message(role=Role.SYSTEM, content="Reply with exactly one category word."),
            Message(role=Role.USER, content=f"Classify: {_question(context.state)}"),
        ]
    )
    category = response.content.strip().casefold()
    if category not in CATEGORIES:
        raise RuntimeError(f"model returned an unknown category: {category!r}")
    return NodeResult(
        update={"category": category},
        usage=response.usage,
        model_calls=1,
        action_key=f"classify:{category}",
    )


async def answer(context: NodeContext) -> NodeResult:
    """Draft a deterministic answer for general questions and finish the run."""

    response = await _model(context).complete(
        [Message(role=Role.USER, content=_question(context.state))]
    )
    return NodeResult(
        update={"resolution": "answered"},
        output=response.content,
        finish=True,
        usage=response.usage,
        model_calls=1,
    )


async def escalate(context: NodeContext) -> NodeResult:
    """Summarize a refund escalation without another model call and finish."""

    slug = re.sub(r"[^a-z0-9]+", "-", _question(context.state).casefold()).strip("-")[:24]
    ticket = f"REFUND-{slug}"
    return NodeResult(
        update={"resolution": "escalated", "ticket": ticket},
        output=f"Escalated refund request as ticket {ticket}.",
        finish=True,
    )


def build_graph() -> CompiledGraph:
    """Define and compile the conditional support-router graph."""

    return (
        StateGraph(GRAPH_NAME, version=GRAPH_VERSION)
        .add_node("classify", classify)
        .add_node("answer", answer)
        .add_node("escalate", escalate)
        .set_entry_point("classify")
        .add_edge(
            "classify",
            "escalate",
            condition=lambda state: state.get("category") == "refund",
            label="refund",
        )
        .add_edge("classify", "answer", label="fallback")
        .add_edge("answer", END)
        .add_edge("escalate", END)
        .compile()
    )


async def run_question(question: str) -> RunResult:
    """Compile the graph and run one question fully offline."""

    runtime = GraphRuntime(
        event_store=InMemoryEventStore(),
        checkpoint_store=InMemoryCheckpointStore(),
        dependencies={"model": MockModelProvider(responder=support_responder)},
    )
    return await runtime.run(build_graph(), {"input": question})


def main() -> None:
    for question in ("Where is my refund?", "How do I change my password?"):
        result = asyncio.run(run_question(question))
        print(f"Q: {question}")
        print(f"   status={result.status.value} category={result.state.get('category')}")
        print(f"   output={result.output}")


if __name__ == "__main__":
    main()
