"""Offline walkthrough: define, compile, and run a custom conditional graph.

This script accompanies ``docs/custom-graph-walkthrough.md``. It uses only the public
runtime API (``StateGraph``, ``GraphRuntime``, ``NodeContext``/``NodeResult``) and the
public ``MockModelProvider``. Everything is deterministic: no network, no credentials,
and no wall-clock assumptions.

The triage graph has a real conditional branch:

    classify -> positive_reply      (state["tone"] == "positive")
             -> negative_reply      (state["tone"] == "negative")
             -> clarify_request     (fallback when the mock cannot decide)
    positive_reply / negative_reply / clarify_request -> END

Run it with::

    uv run python examples/custom_conditional_graph.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, cast

import typer

from admires_harness.adapters.models import MockModelProvider
from admires_harness.contracts import Message, ModelResponse, Role
from admires_harness.runtime import (
    END,
    GraphRuntime,
    InMemoryCheckpointStore,
    InMemoryEventStore,
    NodeContext,
    NodeResult,
    StateGraph,
)

POSITIVE_KEYWORDS = ("thanks", "great", "love", "good", "happy", "works")
NEGATIVE_KEYWORDS = ("broken", "bad", "fail", "error", "hate", "crash")

POSITIVE_INPUT = "Great, the deploy works, thanks for the quick fix!"
NEGATIVE_INPUT = "The build is broken again and the new dashboard started to crash."
UNCLEAR_INPUT = "I am writing to follow up about the thing from yesterday."


def triage_responder(messages: Sequence[Message], tools: Sequence[object]) -> ModelResponse:
    """Deterministic rule-based responder standing in for a remote model."""

    del tools
    user_prompt = next(
        (message.content for message in reversed(messages) if message.role == Role.USER),
        "",
    )
    if any(keyword in user_prompt.casefold() for keyword in POSITIVE_KEYWORDS):
        reply = "Acknowledged, thank you for the positive feedback!"
    elif any(keyword in user_prompt.casefold() for keyword in NEGATIVE_KEYWORDS):
        reply = "We are sorry for the trouble; a support engineer will investigate."
    else:
        reply = "Could you share a bit more detail about what you need?"
    return ModelResponse(content=reply, model="mock-triage-v1")


async def classify(context: NodeContext) -> NodeResult:
    """Call the deterministic mock model, then derive the routing key from state."""

    raw_input = context.state.get("input", "")
    user_input = raw_input if isinstance(raw_input, str) else ""
    model = cast(MockModelProvider, context.dependencies["model"])
    response = await model.complete(
        [
            Message(
                role=Role.SYSTEM,
                content="You triage inbound messages. Decide tone: positive, negative, or unclear.",
            ),
            Message(role=Role.USER, content=user_input),
        ]
    )
    if any(keyword in user_input.casefold() for keyword in POSITIVE_KEYWORDS):
        tone = "positive"
    elif any(keyword in user_input.casefold() for keyword in NEGATIVE_KEYWORDS):
        tone = "negative"
    else:
        tone = "unclear"
    return NodeResult(
        update={"tone": tone, "draft_reply": response.content},
        model_calls=1,
        usage=response.usage,
        action_key=f"classify:{tone}",
    )


async def positive_reply(context: NodeContext) -> NodeResult:
    output = f"[positive branch] {context.state.get('draft_reply', '')}".strip()
    return NodeResult(update={"routed_to": "positive"}, output=output, finish=True)


async def negative_reply(context: NodeContext) -> NodeResult:
    output = f"[negative branch] {context.state.get('draft_reply', '')}".strip()
    return NodeResult(update={"routed_to": "negative"}, output=output, finish=True)


async def clarify_request(context: NodeContext) -> NodeResult:
    output = f"[clarify branch] {context.state.get('draft_reply', '')}".strip()
    return NodeResult(update={"routed_to": "clarify"}, output=output, finish=True)


def build_triage_graph() -> StateGraph:
    """Return the triage graph definition; compile() performs structural validation."""

    return (
        StateGraph("inbox-triage", version="1")
        .add_node("classify", classify)
        .add_node("positive_reply", positive_reply)
        .add_node("negative_reply", negative_reply)
        .add_node("clarify_request", clarify_request)
        .set_entry_point("classify")
        .add_edge(
            "classify",
            "positive_reply",
            condition=lambda state: state.get("tone") == "positive",
        )
        .add_edge(
            "classify",
            "negative_reply",
            condition=lambda state: state.get("tone") == "negative",
        )
        .add_edge("classify", "clarify_request")
        .add_edge("positive_reply", END)
        .add_edge("negative_reply", END)
        .add_edge("clarify_request", END)
    )


@dataclass(frozen=True, slots=True)
class TriageOutcome:
    tone: str
    routed_to: str
    output: str
    steps: int


async def run_triage(user_input: str) -> TriageOutcome:
    """Compile the graph and execute one offline run with in-memory stores."""

    graph = build_triage_graph().compile()
    runtime = GraphRuntime(
        event_store=InMemoryEventStore(),
        checkpoint_store=InMemoryCheckpointStore(),
        dependencies={"model": MockModelProvider(responder=triage_responder)},
    )
    result = await runtime.run(graph, {"input": user_input})
    tone = result.state.get("tone")
    routed_to = result.state.get("routed_to")
    return TriageOutcome(
        tone=tone if isinstance(tone, str) else "unknown",
        routed_to=routed_to if isinstance(routed_to, str) else "unknown",
        output=result.output if isinstance(result.output, str) else "",
        steps=result.steps,
    )


app = typer.Typer(add_completion=False)


@app.command()
def main(
    message: Annotated[
        str | None, typer.Option("--message", "-m", help="Custom message to triage")
    ] = None,
) -> None:
    """Run the positive, negative, and fallback branches of the offline triage graph."""

    samples = (
        (message,)
        if message is not None
        else (
            POSITIVE_INPUT,
            NEGATIVE_INPUT,
            UNCLEAR_INPUT,
        )
    )

    async def _main() -> None:
        for sample in samples:
            outcome = await run_triage(sample)
            typer.echo(f"input:    {sample}")
            typer.echo(f"tone:     {outcome.tone} -> branch {outcome.routed_to}")
            typer.echo(f"output:   {outcome.output}")
            typer.echo(f"steps:    {outcome.steps}")
            typer.echo("-" * 60)

    asyncio.run(_main())


if __name__ == "__main__":
    app()
