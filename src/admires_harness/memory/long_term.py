"""SQLite-backed durable memory."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from uuid import UUID

import aiosqlite

from admires_harness.memory.models import MemoryRecord, RetrievalHit


class SQLiteMemoryStore:
    """A small durable vector store suitable for local harness runs.

    It intentionally favors inspectability over index sophistication. Embeddings are
    JSON arrays in SQLite and scored in process, which keeps the no-key demo portable.
    The embedding dimension is persisted and checked so corrupt/mixed indexes fail
    closed rather than returning misleading matches.
    """

    def __init__(self, path: Path, *, max_candidates: int = 10_000) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        self.path = path
        self.max_candidates = max_candidates
        self._write_lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with self._write_lock, aiosqlite.connect(self.path) as database:
            await database.execute("PRAGMA journal_mode=WAL")
            await database.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    embedding_dimensions INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (namespace, id)
                )
                """
            )
            await database.execute(
                "CREATE INDEX IF NOT EXISTS ix_memories_namespace ON memories(namespace)"
            )
            await database.commit()
        self._initialized = True

    async def put(self, record: MemoryRecord) -> None:
        self._ensure_initialized()
        if not record.embedding:
            raise ValueError("persistent memories require a non-empty embedding")
        values = (
            str(record.id),
            record.namespace,
            record.content,
            record.source,
            json.dumps(record.metadata, ensure_ascii=False, separators=(",", ":")),
            json.dumps(record.embedding, separators=(",", ":")),
            len(record.embedding),
            record.created_at.isoformat(),
            record.updated_at.isoformat(),
        )
        async with self._write_lock, aiosqlite.connect(self.path) as database:
            await database.execute("BEGIN IMMEDIATE")
            dimension_cursor = await database.execute(
                "SELECT embedding_dimensions FROM memories WHERE namespace = ? LIMIT 1",
                (record.namespace,),
            )
            dimension_row = await dimension_cursor.fetchone()
            if dimension_row is not None and dimension_row[0] != len(record.embedding):
                await database.rollback()
                raise ValueError("embedding dimension differs from existing namespace records")
            await database.execute(
                """
                INSERT INTO memories (
                    id, namespace, content, source, metadata_json, embedding_json,
                    embedding_dimensions, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, id) DO UPDATE SET
                    content=excluded.content, source=excluded.source,
                    metadata_json=excluded.metadata_json,
                    embedding_json=excluded.embedding_json,
                    embedding_dimensions=excluded.embedding_dimensions,
                    updated_at=excluded.updated_at
                """,
                values,
            )
            await database.commit()

    async def get(self, memory_id: UUID, *, namespace: str) -> MemoryRecord | None:
        self._ensure_initialized()
        self._validate_namespace(namespace)
        async with aiosqlite.connect(self.path) as database:
            database.row_factory = aiosqlite.Row
            cursor = await database.execute(
                "SELECT * FROM memories WHERE id = ? AND namespace = ?",
                (str(memory_id), namespace),
            )
            row = await cursor.fetchone()
        return self._from_row(row) if row is not None else None

    async def delete(self, memory_id: UUID, *, namespace: str) -> bool:
        self._ensure_initialized()
        self._validate_namespace(namespace)
        async with self._write_lock, aiosqlite.connect(self.path) as database:
            cursor = await database.execute(
                "DELETE FROM memories WHERE id = ? AND namespace = ?",
                (str(memory_id), namespace),
            )
            await database.commit()
            return cursor.rowcount == 1

    async def search(
        self,
        query_embedding: tuple[float, ...],
        *,
        namespace: str,
        limit: int = 5,
        min_score: float = 0.0,
        metadata: dict[str, object] | None = None,
    ) -> list[RetrievalHit]:
        self._ensure_initialized()
        self._validate_namespace(namespace)
        if not query_embedding:
            raise ValueError("query_embedding must not be empty")
        if not all(math.isfinite(value) for value in query_embedding):
            raise ValueError("query_embedding values must be finite")
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")

        async with aiosqlite.connect(self.path) as database:
            database.row_factory = aiosqlite.Row
            cursor = await database.execute(
                """
                SELECT * FROM memories
                WHERE namespace = ? AND embedding_dimensions = ?
                ORDER BY updated_at DESC, id ASC LIMIT ?
                """,
                (namespace, len(query_embedding), self.max_candidates),
            )
            rows = await cursor.fetchall()

        hits: list[RetrievalHit] = []
        for row in rows:
            try:
                record = self._from_row(row)
            except (KeyError, ValueError):
                # A corrupt or anomalous row must not fail the entire query.
                continue
            if len(record.embedding) != len(query_embedding):
                continue
            if metadata is not None and not all(
                record.metadata.get(key) == value for key, value in metadata.items()
            ):
                continue
            # Negative correlation and orthogonal/zero vectors are not useful matches.
            # Clamp so floating-point noise cannot push a perfect match above 1.0.
            score = min(1.0, max(0.0, self._cosine(query_embedding, record.embedding)))
            if score >= min_score:
                hits.append(RetrievalHit(record=record, score=score))
        hits.sort(key=lambda hit: (-hit.score, str(hit.record.id)))
        return hits[:limit]

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("memory store is not initialized; call initialize() first")

    @staticmethod
    def _validate_namespace(namespace: str) -> None:
        if not namespace.strip():
            raise ValueError("namespace must not be empty")

    @staticmethod
    def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        if len(left) != len(right):
            raise ValueError("embedding dimension mismatch")
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)

    @staticmethod
    def _from_row(row: aiosqlite.Row) -> MemoryRecord:
        return MemoryRecord(
            id=UUID(row["id"]),
            namespace=row["namespace"],
            content=row["content"],
            source=row["source"],
            metadata=json.loads(row["metadata_json"]),
            embedding=tuple(json.loads(row["embedding_json"])),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
