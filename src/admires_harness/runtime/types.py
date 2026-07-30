"""Runtime-only contracts for graph nodes, results, and replay snapshots."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import Field, model_validator

from admires_harness.contracts import (
    EventType,
    JsonValue,
    ModelUsage,
    RunBudget,
    RunEvent,
    RunStatus,
    StrictModel,
)

if TYPE_CHECKING:
    from admires_harness.runtime.graph import CompiledGraph

END = "__end__"


class NodeResult(StrictModel):
    """The sole mutation and routing surface available to a graph node.

    ``update`` is merged into a fresh state mapping. A node may select an explicit
    destination with ``goto``, finish immediately, or let graph edges route it.
    Paused results are checkpointed before control returns to the caller.
    """

    update: dict[str, JsonValue] = Field(default_factory=dict)
    goto: str | None = None
    finish: bool = False
    pause: RunStatus | None = None
    output: JsonValue = None
    usage: ModelUsage = Field(default_factory=ModelUsage)
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    action_key: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_control_flow(self) -> NodeResult:
        if self.finish and self.goto is not None:
            raise ValueError("finish and goto are mutually exclusive")
        if self.finish and self.pause is not None:
            raise ValueError("a finished node cannot pause")
        if self.pause not in {None, RunStatus.PAUSED, RunStatus.WAITING_APPROVAL}:
            raise ValueError("pause must be paused or waiting_approval")
        return self


class RunResult(StrictModel):
    """Materialized result returned for successful and controlled terminal runs."""

    run_id: UUID
    trace_id: UUID
    parent_run_id: UUID | None = None
    graph_name: str
    graph_version: str
    status: RunStatus
    state: dict[str, JsonValue]
    output: JsonValue = None
    next_node: str | None = None
    checkpoint_revision: int | None = None
    steps: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)
    started_at: datetime
    ended_at: datetime
    error: str | None = None


class ReplaySnapshot(StrictModel):
    """Read-only reconstruction from persisted events and the latest checkpoint."""

    run_id: UUID
    status: RunStatus
    state: dict[str, JsonValue]
    next_node: str | None
    checkpoint_revision: int | None
    events: tuple[RunEvent, ...]


type EmitCallback = Callable[[EventType, Mapping[str, JsonValue]], Awaitable[RunEvent]]
type DelegateCallback = Callable[
    [
        "CompiledGraph",
        Mapping[str, JsonValue],
        str | None,
        RunBudget | None,
        Sequence[str],
    ],
    Awaitable[RunResult],
]


@dataclass(frozen=True, slots=True)
class NodeContext:
    """Capability-limited context passed to every node invocation."""

    run_id: UUID
    trace_id: UUID
    parent_run_id: UUID | None
    node_id: str
    checkpoint_revision: int | None
    state: Mapping[str, JsonValue]
    budget: RunBudget
    dependencies: Mapping[str, Any]
    _emit: EmitCallback
    _delegate: DelegateCallback

    @classmethod
    def create(
        cls,
        *,
        run_id: UUID,
        trace_id: UUID,
        parent_run_id: UUID | None,
        node_id: str,
        checkpoint_revision: int | None,
        state: dict[str, JsonValue],
        budget: RunBudget,
        dependencies: Mapping[str, Any],
        emit: EmitCallback,
        delegate: DelegateCallback,
    ) -> NodeContext:
        return cls(
            run_id=run_id,
            trace_id=trace_id,
            parent_run_id=parent_run_id,
            node_id=node_id,
            checkpoint_revision=checkpoint_revision,
            # Detach nested JSON containers so a node cannot mutate runtime state by
            # retaining or changing a list/dict reached through the read-only mapping.
            state=MappingProxyType(deepcopy(state)),
            budget=budget,
            dependencies=MappingProxyType(dict(dependencies)),
            _emit=emit,
            _delegate=delegate,
        )

    async def emit(
        self, event_type: EventType, payload: Mapping[str, JsonValue] | None = None
    ) -> RunEvent:
        """Append a domain event within the current node span."""

        return await self._emit(event_type, payload or {})

    async def delegate(
        self,
        graph: CompiledGraph,
        state: Mapping[str, JsonValue],
        *,
        label: str | None = None,
        budget: RunBudget | None = None,
        capabilities: Sequence[str] = (),
    ) -> RunResult:
        """Run a child graph with an explicit budget and least-privilege capabilities."""

        return await self._delegate(graph, state, label, budget, capabilities)


type NodeReturn = NodeResult | Mapping[str, JsonValue] | None
type NodeCallable = Callable[[NodeContext], NodeReturn | Awaitable[NodeReturn]]
