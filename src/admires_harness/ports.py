"""Dependency-inversion ports used by the core runtime."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Protocol
from uuid import UUID

from admires_harness.contracts import (
    Checkpoint,
    Message,
    ModelResponse,
    RunEvent,
    ToolCall,
    ToolResult,
)


class ModelProvider(Protocol):
    name: str

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> ModelResponse: ...

    def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> AsyncIterator[str]: ...


class ToolExecutor(Protocol):
    async def execute(
        self,
        call: ToolCall,
        *,
        run_id: UUID,
        idempotency_key: str,
    ) -> ToolResult: ...

    def schemas(self) -> Sequence[Mapping[str, Any]]: ...


class EventStore(Protocol):
    async def append(self, event: RunEvent) -> None: ...

    async def list(self, run_id: UUID, *, after: int = -1) -> list[RunEvent]: ...


class CheckpointStore(Protocol):
    async def save(self, checkpoint: Checkpoint, *, expected_revision: int | None) -> None: ...

    async def latest(self, run_id: UUID) -> Checkpoint | None: ...
