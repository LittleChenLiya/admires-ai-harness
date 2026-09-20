"""Executable offline example: define, compile, and run a custom conditional graph.

Run it from the repository root:

    uv run python examples/custom_graph.py
    uv run python examples/custom_graph.py --question "How do I fix a Python KeyError?"

The example is deterministic: it uses the rule-based ``MockModelProvider`` and
in-memory stores, so it needs no network, credentials, or wall-clock reads. See
``docs/custom-graphs.md`` for the full walkthrough.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from admires_harness.adapters.models import MockModelProvider
from admires_harness.contracts import Message, ModelResponse, ModelUsage, Role, RunStatus
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

GRAPH_NAME = "custom-support-router"
TECHNICAL_KEYWORDS = ("python", "code", "error", "bug", "exception", "stack trace")


def build_model() -> MockModelProvider:
    """Deterministic offline model: rules instead of network calls."""

    def responder(messages: Sequence[Message], tools: Sequence[Mapping[str, Any]]) -> ModelResponse:
        del tools
        system = messages[0].content if messages else ""
        user = next(
            (message.content for message in reversed(messages) if message.role == Role.USER),
            "",
        )
        if "classifier" in system:
            category = (
                "technical"
                if any(keyword in user.casefold() for keyword in TECHNICAL_KEYWORDS)
                else "general"
            )
            content = category
        elif "technical" in system:
            content = "Technical answer: reproduce, isolate the failing line, then patch it."
        else:
            content = "General answer: here is a friendly, high-level explanation."
        return ModelResponse(
            content=content,
            model="mock-custom-graph-v1",
            usage=ModelUsage(
                input_tokens=sum(len(message.content.split()) for message in messages),
                output_tokens=len(content.split()),
            ),
        )

    return MockModelProvider(responder=responder)


async def _classify(context: NodeContext) -> NodeResult:
    model = context.dependencies["model"]
    question = str(context.state.get("question", ""))
    response = await model.complete(
        [
            Message(role=Role.SYSTEM, content="You are a classifier. Reply with one category."),
            Message(role=Role.USER, content=question),
        ]
    )
    category = response.content.strip()
    if category not in {"technical", "general"}:
        raise RuntimeError(f"classifier returned an unknown category: {category!r}")
    return NodeResult(
        update={"category": category},
        usage=response.usage,
        model_calls=1,
    )


async def _answer(context: NodeContext, *, system: str, key: str) -> NodeResult:
    model = context.dependencies["model"]
    question = str(context.state.get("question", ""))
    response = await model.complete(
        [Message(role=Role.SYSTEM, content=system), Message(role=Role.USER, content=question)]
    )
    return NodeResult(
        update={key: response.content},
        output=response.content,
        finish=True,
        usage=response.usage,
        model_calls=1,
    )


async def _answer_technical(context: NodeContext) -> NodeResult:
    return await _answer(
        context,
        system="You are a technical support specialist.",
        key="technical_answer",
    )


async def _answer_general(context: NodeContext) -> NodeResult:
    return await _answer(
        context,
        system="You are a general support specialist.",
        key="general_answer",
    )


def build_graph() -> CompiledGraph:
    """A question router: classify, then branch to a specialist answer node."""

    return (
        StateGraph(GRAPH_NAME)
        .add_node("classify", _classify)
        .add_node("technical", _answer_technical)
        .add_node("general", _answer_general)
        .set_entry_point("classify")
        .add_edge(
            "classify",
            "technical",
            condition=lambda state: state.get("category") == "technical",
            label="category is technical",
        )
        .add_edge("classify", "general")
        .add_edge("technical", END)
        .add_edge("general", END)
        .compile()
    )


async def run(question: str, *, model: MockModelProvider | None = None) -> RunResult:
    """Compile the graph and execute one run against in-memory stores."""

    runtime = GraphRuntime(
        event_store=InMemoryEventStore(),
        checkpoint_store=InMemoryCheckpointStore(),
        dependencies={"model": model or build_model()},
    )
    return await runtime.run(build_graph(), {"question": question})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--question",
        default="How do I fix a Python KeyError?",
        help="Question routed through the conditional graph.",
    )
    args = parser.parse_args()
    result = asyncio.run(run(args.question))
    print(f"status:   {result.status.value}")
    print(f"category: {result.state.get('category')}")
    print(f"steps:    {result.steps}")
    print(f"output:   {result.output}")
    if result.status is not RunStatus.SUCCEEDED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
