"""Offline custom conditional graph: a deterministic support-ticket triager.

This file is a walkthrough companion to ``docs/custom-graph.md``. It uses only the
public runtime surface (``admires_harness.runtime`` and ``admires_harness.adapters``):
there are no private test helpers, network calls, credentials, or wall-clock dependent
assertions. ``python examples/custom_graph_triage.py`` runs a real conditional branch
through the event-sourced engine and prints a deterministic report.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from admires_harness.adapters.models import MockModelProvider
from admires_harness.contracts import (
    EventType,
    JsonValue,
    Message,
    ModelResponse,
    ModelUsage,
    Role,
)
from admires_harness.ports import ModelProvider
from admires_harness.runtime import (
    END,
    GraphRuntime,
    InMemoryCheckpointStore,
    InMemoryEventStore,
    NodeContext,
    NodeResult,
    StateGraph,
)

PRIORITY_URGENT = "urgent"
PRIORITY_ROUTINE = "routine"

URGENT_MARKERS = ("outage", "down", "cannot log in", "can't log in", "breach")
ROUTINE_MARKERS = ("question", "how do i", "feature request", "documentation")

SYSTEM_PROMPT = "You are a support triage classifier for an offline demo."


def triage_responder(
    messages: Sequence[Message], tools: Sequence[Mapping[str, Any]]
) -> ModelResponse:
    """Classify a ticket from deterministic keywords; no randomness, no network."""

    del tools
    prompt = next(
        (message.content for message in reversed(messages) if message.role == Role.USER), ""
    )
    normalized = prompt.casefold()
    if any(marker in normalized for marker in URGENT_MARKERS):
        priority, rationale = PRIORITY_URGENT, "describes a service outage or security event"
    elif any(marker in normalized for marker in ROUTINE_MARKERS):
        priority, rationale = PRIORITY_ROUTINE, "is a how-to or documentation question"
    else:
        priority, rationale = PRIORITY_URGENT, "needs human review because intent is unclear"
    content = f"priority: {priority}    reason: {rationale}"
    return ModelResponse(
        content=content,
        model="mock-triage-v1",
        usage=ModelUsage(
            input_tokens=sum(len(message.content.split()) for message in messages),
            output_tokens=len(content.split()),
            cost_usd=0.0,
        ),
    )


async def _classify(context: NodeContext, model: ModelProvider) -> ModelResponse:
    prompt = f"Ticket: {context.state['input']}"
    await context.emit(EventType.MODEL_STARTED, {"provider": str(model.name)})
    response = await model.complete(
        [
            Message(role=Role.SYSTEM, content=SYSTEM_PROMPT),
            Message(role=Role.USER, content=prompt),
        ],
    )
    await context.emit(EventType.MODEL_COMPLETED, {"model": response.model})
    return response


async def triage(context: NodeContext) -> NodeResult:
    """Call the mock classifier and record its decision in graph state."""

    model = context.dependencies["model"]
    response = await _classify(context, model)
    fields = dict(line.split(": ", 1) for line in response.content.split("    ") if ": " in line)
    return NodeResult(
        update={
            "priority": fields.get("priority", PRIORITY_URGENT),
            "triage_reason": fields.get("reason", "unparsed classifier output"),
        },
        model_calls=1,
        usage=response.usage,
    )


async def escalate(context: NodeContext) -> NodeResult:
    """Urgent branch: page the on-call engineer."""

    output = f"ESCALATED ({context.state['priority']}): {context.state['input']}"
    return NodeResult(update={"route": "escalate", "answer": output}, output=output, finish=True)


async def draft_reply(context: NodeContext) -> NodeResult:
    """Routine branch: draft a canned self-service answer."""

    answer = (
        f"ROUTINE REPLY: Thanks, the team will answer: {context.state['input']} "
        f"(reason: {context.state['triage_reason']})"
    )
    return NodeResult(update={"route": "reply", "answer": answer}, output=answer, finish=True)


def is_urgent(state: Mapping[str, JsonValue]) -> bool:
    """Conditional-edge predicate, evaluated against the post-update state."""

    return state.get("priority") == PRIORITY_URGENT


def build_graph() -> StateGraph:
    """Define the triage graph with one real conditional branch."""

    return (
        StateGraph("support-triage", version="1")
        .add_node("triage", triage)
        .add_node("escalate", escalate)
        .add_node("draft_reply", draft_reply)
        .set_entry_point("triage")
        .add_edge("triage", "escalate", condition=is_urgent, label="urgent")
        .add_edge("triage", "draft_reply", label="routine")
        .add_edge("escalate", END)
        .add_edge("draft_reply", END)
    )


async def run_ticket(model: MockModelProvider, ticket: str) -> dict[str, JsonValue]:
    """Compile the graph and execute one ticket through the in-memory runtime."""

    runtime = GraphRuntime(
        event_store=InMemoryEventStore(),
        checkpoint_store=InMemoryCheckpointStore(),
        dependencies={"model": model},
    )
    result = await runtime.run(build_graph().compile(), {"input": ticket})
    return {
        "status": result.status.value,
        "route": result.state["route"],
        "priority": result.state["priority"],
        "model_calls": result.model_calls,
        "output": result.output,
    }


TICKETS = (
    "[T-1] Production checkout is down for every customer in EU-West.",
    "[T-2] Question: how do I export my runs to a CSV file?",
)


async def main() -> None:
    for ticket in TICKETS:
        report = await run_ticket(MockModelProvider(responder=triage_responder), ticket)
        print(
            f"{report['status']:<9} route={report['route']:<8} "
            f"priority={report['priority']:<7} model_calls={report['model_calls']}"
        )
        print(f"  {report['output']}")


if __name__ == "__main__":
    asyncio.run(main())
