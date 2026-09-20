"""Adapter boundary for pluggable long-term memory stores."""

from __future__ import annotations

from typing import Protocol

from admires_harness.memory.models import MemoryRecord, RetrievalHit


class MemoryStore(Protocol):
    """Durable memory port.

    Implementations must keep namespaces isolated, reject cross-dimension writes
    within a namespace, and surface storage failures as MemoryStoreError subclasses
    rather than raw driver exceptions.
    """

    async def put(self, record: MemoryRecord) -> None: ...

    async def search(
        self,
        query_embedding: tuple[float, ...],
        *,
        namespace: str,
        limit: int = 5,
        min_score: float = 0.0,
        metadata: dict[str, object] | None = None,
    ) -> list[RetrievalHit]: ...
