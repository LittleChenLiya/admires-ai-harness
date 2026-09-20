"""SQLite-backed durable memory."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import aiosqlite

from admires_harness.memory.models import MemoryRecord, RetrievalHit


class MemoryStoreError(RuntimeError):
    """Base class for stable, diagnosable memory store failures."""


class MemoryStoreLockedError(MemoryStoreError):
    """The database stayed locked by another connection past the busy timeout."""


class MemoryCorruptionError(MemoryStoreError):
    """A persisted row cannot be decoded into a valid memory record."""


class MemoryStoreClosedError(MemoryStoreError):
    """An operation was attempted on a closed store."""


class SQLiteMemoryStore:
    """A small durable vector store suitable for local harness runs.

    It intentionally favors inspectability over index sophistication. Embeddings are
    JSON arrays in SQLite and scored in process, which keeps the no-key demo portable.
    The embedding dimension is persisted and checked so corrupt/mixed indexes fail
    closed rather than returning misleading matches.

    Concurrency and lifecycle contract:

    - initialize() is idempotent and safe to run from many instances against the
      same database file; schema creation uses WAL mode and IF NOT EXISTS.
    - Writes are serialized per instance and guarded cross-instance by a SQLite busy
      timeout; lock contention that outlasts it raises MemoryStoreLockedError.
    - Every operation opens and closes its own connection and cursors, so failed or
      cancelled calls never leak handles or leave half-committed transactions.
    - Rows that fail to decode raise MemoryCorruptionError instead of returning
      misleading matches.
    - close() releases the store; further operations raise MemoryStoreClosedError
      until initialize() reopens it.
    """

    def __init__(
        self,
        path: Path,
        *,
        max_candidates: int = 10_000,
        busy_timeout_ms: int = 5_000,
    ) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must not be negative")
        self.path = path
        self.max_candidates = max_candidates
        self._busy_timeout_ms = busy_timeout_ms
        self._write_lock = asyncio.Lock()
        self._initialized = False
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def __aenter__(self) -> SQLiteMemoryStore:
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()

    async def initialize(self) -> None:
        async with self._write_lock:
            if self._initialized and not self._closed:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # The WAL mode transition can report SQLITE_BUSY without honoring the
            # busy timeout when many instances race to create the same database,
            # so schema setup retries briefly before surfacing a locked error.
            for attempt in range(5):
                try:
                    async with self._connect() as database:
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
                            "CREATE INDEX IF NOT EXISTS ix_memories_namespace "
                            "ON memories(namespace)"
                        )
                        await database.commit()
                    break
                except MemoryStoreLockedError:
                    if attempt == 4:
                        raise
                    await asyncio.sleep(0.05 * (attempt + 1))
            self._initialized = True
            self._closed = False

    async def close(self) -> None:
        """Mark the store closed; connections are per-operation, so this is cheap."""
        self._closed = True
        self._initialized = False

    async def put(self, record: MemoryRecord) -> None:
        self._ensure_open()
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
        async with self._write_lock, self._connect() as database:
            await database.execute("BEGIN IMMEDIATE")
            try:
                async with database.execute(
                    "SELECT embedding_dimensions FROM memories WHERE namespace = ? LIMIT 1",
                    (record.namespace,),
                ) as dimension_cursor:
                    dimension_row = await dimension_cursor.fetchone()
                if dimension_row is not None and dimension_row[0] != len(record.embedding):
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
            except BaseException:
                with contextlib.suppress(sqlite3.Error):
                    await database.rollback()
                raise
            await database.commit()

    async def get(self, memory_id: UUID, *, namespace: str) -> MemoryRecord | None:
        self._ensure_open()
        self._validate_namespace(namespace)
        async with (
            self._connect() as database,
            database.execute(
                "SELECT * FROM memories WHERE id = ? AND namespace = ?",
                (str(memory_id), namespace),
            ) as cursor,
        ):
            row = await cursor.fetchone()
        return self._from_row(row) if row is not None else None

    async def delete(self, memory_id: UUID, *, namespace: str) -> bool:
        self._ensure_open()
        self._validate_namespace(namespace)
        async with self._write_lock, self._connect() as database:
            async with database.execute(
                "DELETE FROM memories WHERE id = ? AND namespace = ?",
                (str(memory_id), namespace),
            ) as cursor:
                deleted = cursor.rowcount == 1
            await database.commit()
            return deleted

    async def search(
        self,
        query_embedding: tuple[float, ...],
        *,
        namespace: str,
        limit: int = 5,
        min_score: float = 0.0,
        metadata: dict[str, object] | None = None,
    ) -> list[RetrievalHit]:
        self._ensure_open()
        self._validate_namespace(namespace)
        if not query_embedding:
            raise ValueError("query_embedding must not be empty")
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")

        async with (
            self._connect() as database,
            database.execute(
                """
                SELECT * FROM memories
                WHERE namespace = ? AND embedding_dimensions = ?
                ORDER BY updated_at DESC LIMIT ?
                """,
                (namespace, len(query_embedding), self.max_candidates),
            ) as cursor,
        ):
            rows = await cursor.fetchall()

        hits: list[RetrievalHit] = []
        for row in rows:
            record = self._from_row(row)
            if metadata is not None and not all(
                record.metadata.get(key) == value for key, value in metadata.items()
            ):
                continue
            # Negative correlation and orthogonal/zero vectors are not useful matches.
            score = max(0.0, self._cosine(query_embedding, record.embedding))
            if score >= min_score:
                hits.append(RetrievalHit(record=record, score=score))
        hits.sort(key=lambda hit: (-hit.score, str(hit.record.id)))
        return hits[:limit]

    @contextlib.asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        try:
            async with aiosqlite.connect(self.path) as database:
                database.row_factory = aiosqlite.Row
                await database.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
                yield database
        except sqlite3.OperationalError as exc:
            raise self._translate_operational_error(exc) from exc

    def _translate_operational_error(self, exc: sqlite3.OperationalError) -> MemoryStoreError:
        message = str(exc).lower()
        if "locked" in message or "busy" in message:
            return MemoryStoreLockedError(
                f"memory database at {self.path} is locked by another connection: {exc}"
            )
        return MemoryStoreError(f"memory database at {self.path} failed: {exc}")

    def _ensure_open(self) -> None:
        if self._closed:
            raise MemoryStoreClosedError("memory store is closed; call initialize() to reopen it")
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
        try:
            row_id = row["id"]
            namespace = row["namespace"]
        except (KeyError, IndexError, TypeError):
            row_id = namespace = "<unreadable>"
        try:
            return MemoryRecord(
                id=UUID(row_id),
                namespace=namespace,
                content=row["content"],
                source=row["source"],
                metadata=json.loads(row["metadata_json"]),
                embedding=tuple(json.loads(row["embedding_json"])),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise MemoryCorruptionError(
                f"corrupt memory row id={row_id!r} namespace={namespace!r}: {exc}"
            ) from exc
