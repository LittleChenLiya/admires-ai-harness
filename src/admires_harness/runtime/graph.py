"""A small, explicit, framework-independent state graph."""

from __future__ import annotations

import inspect
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from admires_harness.contracts import JsonValue
from admires_harness.runtime.errors import GraphDefinitionError, GraphRoutingError
from admires_harness.runtime.types import END, NodeCallable, NodeResult

type Predicate = Callable[[Mapping[str, JsonValue]], bool | Awaitable[bool]]


@dataclass(frozen=True, slots=True)
class Edge:
    source: str
    target: str
    condition: Predicate | None = None
    label: str | None = None


@dataclass(frozen=True, slots=True)
class CompiledGraph:
    """Validated immutable graph safe to share between concurrent runs."""

    name: str
    version: str
    entry_point: str
    nodes: Mapping[str, NodeCallable]
    edges: Mapping[str, tuple[Edge, ...]]

    def node(self, node_id: str) -> NodeCallable:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise GraphRoutingError(f"unknown node: {node_id}") from exc

    async def next_node(
        self, node_id: str, state: Mapping[str, JsonValue], result: NodeResult
    ) -> str | None:
        """Resolve explicit or conditional routing after a node completes."""

        outgoing = self.edges.get(node_id, ())
        if result.finish:
            return None
        if result.goto is not None:
            target = result.goto
            declared = {edge.target for edge in outgoing}
            if target not in declared:
                raise GraphRoutingError(
                    f"node {node_id!r} selected undeclared destination {target!r}"
                )
            return None if target == END else target

        fallback: str | None = None
        predicate_state = MappingProxyType(deepcopy(dict(state)))
        for edge in outgoing:
            if edge.condition is None:
                fallback = edge.target
                continue
            selected = edge.condition(predicate_state)
            if inspect.isawaitable(selected):
                selected = await selected
            if selected:
                return None if edge.target == END else edge.target
        if fallback is not None:
            return None if fallback == END else fallback
        return None


class StateGraph:
    """Fluent graph builder with eager structural validation on ``compile``."""

    def __init__(self, name: str, *, version: str = "1") -> None:
        if not name.strip():
            raise GraphDefinitionError("graph name cannot be empty")
        if not version.strip():
            raise GraphDefinitionError("graph version cannot be empty")
        self.name = name
        self.version = version
        self._nodes: dict[str, NodeCallable] = {}
        self._edges: dict[str, list[Edge]] = {}
        self._entry_point: str | None = None

    def add_node(self, node_id: str, node: NodeCallable) -> StateGraph:
        self._validate_node_id(node_id)
        if node_id in self._nodes:
            raise GraphDefinitionError(f"duplicate node: {node_id}")
        if not callable(node):
            raise GraphDefinitionError(f"node {node_id!r} is not callable")
        self._nodes[node_id] = node
        return self

    def set_entry_point(self, node_id: str) -> StateGraph:
        self._validate_node_id(node_id)
        self._entry_point = node_id
        return self

    def add_edge(
        self,
        source: str,
        target: str,
        *,
        condition: Predicate | None = None,
        label: str | None = None,
    ) -> StateGraph:
        self._validate_node_id(source)
        if target != END:
            self._validate_node_id(target)
        if condition is not None and not callable(condition):
            raise GraphDefinitionError("edge condition must be callable")
        edges = self._edges.setdefault(source, [])
        if any(edge.condition is None for edge in edges) and condition is None:
            raise GraphDefinitionError(f"node {source!r} has more than one fallback edge")
        edges.append(Edge(source, target, condition, label))
        return self

    def compile(self) -> CompiledGraph:
        if not self._nodes:
            raise GraphDefinitionError("a graph requires at least one node")
        if self._entry_point is None:
            raise GraphDefinitionError("entry point is not set")
        if self._entry_point not in self._nodes:
            raise GraphDefinitionError(f"unknown entry point: {self._entry_point}")

        for source, edges in self._edges.items():
            if source not in self._nodes:
                raise GraphDefinitionError(f"edge starts at unknown node: {source}")
            for edge in edges:
                if edge.target != END and edge.target not in self._nodes:
                    raise GraphDefinitionError(f"edge targets unknown node: {edge.target}")

        reachable = self._reachable_from_entry()
        unreachable = sorted(set(self._nodes) - reachable)
        if unreachable:
            raise GraphDefinitionError(f"unreachable nodes: {', '.join(unreachable)}")

        return CompiledGraph(
            name=self.name,
            version=self.version,
            entry_point=self._entry_point,
            nodes=cast(Mapping[str, NodeCallable], MappingProxyType(dict(self._nodes))),
            edges=MappingProxyType({key: tuple(value) for key, value in self._edges.items()}),
        )

    def _reachable_from_entry(self) -> set[str]:
        assert self._entry_point is not None
        seen: set[str] = set()
        pending = deque([self._entry_point])
        while pending:
            node_id = pending.popleft()
            if node_id in seen:
                continue
            seen.add(node_id)
            pending.extend(
                edge.target for edge in self._edges.get(node_id, ()) if edge.target != END
            )
        return seen

    @staticmethod
    def _validate_node_id(node_id: str) -> None:
        if not node_id or not node_id.strip() or node_id == END:
            raise GraphDefinitionError(f"invalid node id: {node_id!r}")
