"""Reference in-memory stores for tests, examples, and offline execution."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from uuid import UUID

from admires_harness.contracts import Checkpoint, RunEvent
from admires_harness.runtime.errors import CheckpointConflictError


class InMemoryEventStore:
    """Concurrency-safe append-only event store with sequence validation."""

    def __init__(self) -> None:
        self._events: dict[UUID, list[RunEvent]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def append(self, event: RunEvent) -> None:
        async with self._lock:
            events = self._events[event.run_id]
            expected = len(events)
            if event.sequence != expected:
                raise CheckpointConflictError(
                    f"event sequence conflict for {event.run_id}: "
                    f"expected {expected}, got {event.sequence}"
                )
            events.append(event)

    async def list(self, run_id: UUID, *, after: int = -1) -> list[RunEvent]:
        async with self._lock:
            return [event for event in self._events.get(run_id, ()) if event.sequence > after]


class InMemoryCheckpointStore:
    """Optimistically versioned checkpoint store."""

    def __init__(self) -> None:
        self._checkpoints: dict[UUID, Checkpoint] = {}
        self._lock = asyncio.Lock()

    async def save(self, checkpoint: Checkpoint, *, expected_revision: int | None) -> None:
        async with self._lock:
            current = self._checkpoints.get(checkpoint.run_id)
            current_revision = current.revision if current is not None else None
            if current_revision != expected_revision:
                raise CheckpointConflictError(
                    f"checkpoint conflict for {checkpoint.run_id}: "
                    f"expected revision {expected_revision}, found {current_revision}"
                )
            next_revision = 0 if current is None else current.revision + 1
            if checkpoint.revision != next_revision:
                raise CheckpointConflictError(
                    f"checkpoint revision must be {next_revision}, got {checkpoint.revision}"
                )
            self._checkpoints[checkpoint.run_id] = checkpoint

    async def latest(self, run_id: UUID) -> Checkpoint | None:
        async with self._lock:
            return self._checkpoints.get(run_id)
