"""Data contracts for persistent and retrieved memories.

Namespaces are mandatory.  This makes the isolation boundary explicit at every call
site instead of relying on a dangerous process-wide default collection.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import Field, StringConstraints, model_validator

from admires_harness.contracts import JsonValue, StrictModel, utc_now

Namespace = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]


class MemoryRecord(StrictModel):
    """A durable, namespace-isolated piece of knowledge."""

    id: UUID = Field(default_factory=uuid4)
    namespace: Namespace
    content: str = Field(min_length=1, max_length=100_000)
    source: str = Field(default="memory", min_length=1, max_length=2_048)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    embedding: tuple[float, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def embedding_must_be_finite(self) -> MemoryRecord:
        if not all(math.isfinite(value) for value in self.embedding):
            raise ValueError("embedding values must be finite")
        return self


class RetrievalHit(StrictModel):
    """A scored record returned by a memory query."""

    record: MemoryRecord
    score: float = Field(ge=0.0, le=1.0)
