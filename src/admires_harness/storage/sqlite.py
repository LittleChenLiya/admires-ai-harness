"""Inspectable SQLite persistence with optimistic concurrency."""

from __future__ import annotations

import asyncio
import builtins
from pathlib import Path
from uuid import UUID

import aiosqlite

from admires_harness.contracts import Checkpoint, RunEvent
from admires_harness.runtime.errors import CheckpointConflictError


class _SQLiteStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._initialized = False
        self._initialize_lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._initialize_lock:
            if self._initialized:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(self.path) as database:
                await database.execute("PRAGMA journal_mode=WAL")
                await database.execute("PRAGMA foreign_keys=ON")
                await self._create_tables(database)
                await database.commit()
            self._initialized = True

    async def _create_tables(self, database: aiosqlite.Connection) -> None:
        raise NotImplementedError

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("store is not initialized; call initialize() first")


class SQLiteEventStore(_SQLiteStore):
    """Append-only event store with a unique per-run sequence."""

    async def _create_tables(self, database: aiosqlite.Connection) -> None:
        await database.execute(
            """
            CREATE TABLE IF NOT EXISTS run_events (
                event_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                sequence INTEGER NOT NULL CHECK (sequence >= 0),
                timestamp TEXT NOT NULL,
                event_json TEXT NOT NULL,
                UNIQUE (run_id, sequence)
            )
            """
        )
        await database.execute(
            "CREATE INDEX IF NOT EXISTS ix_run_events_run ON run_events(run_id, sequence)"
        )
        await database.execute(
            "CREATE INDEX IF NOT EXISTS ix_run_events_time ON run_events(timestamp DESC)"
        )

    async def append(self, event: RunEvent) -> None:
        self._ensure_initialized()
        payload = event.model_dump_json()
        try:
            async with aiosqlite.connect(self.path) as database:
                await database.execute("BEGIN IMMEDIATE")
                cursor = await database.execute(
                    "SELECT COALESCE(MAX(sequence), -1) + 1 FROM run_events WHERE run_id = ?",
                    (str(event.run_id),),
                )
                row = await cursor.fetchone()
                expected = int(row[0]) if row is not None else 0
                if event.sequence != expected:
                    raise CheckpointConflictError(
                        f"event sequence conflict for {event.run_id}: "
                        f"expected {expected}, got {event.sequence}"
                    )
                await database.execute(
                    """
                    INSERT INTO run_events(event_id, run_id, sequence, timestamp, event_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(event.event_id),
                        str(event.run_id),
                        event.sequence,
                        event.timestamp.isoformat(),
                        payload,
                    ),
                )
                await database.commit()
        except aiosqlite.IntegrityError as exc:
            raise CheckpointConflictError(
                f"event sequence or identity conflict for {event.run_id}"
            ) from exc

    async def list(self, run_id: UUID, *, after: int = -1) -> builtins.list[RunEvent]:
        self._ensure_initialized()
        async with aiosqlite.connect(self.path) as database:
            cursor = await database.execute(
                """
                SELECT event_json FROM run_events
                WHERE run_id = ? AND sequence > ? ORDER BY sequence ASC
                """,
                (str(run_id), after),
            )
            rows = await cursor.fetchall()
        return [RunEvent.model_validate_json(row[0]) for row in rows]

    async def list_run_ids(self, *, limit: int = 100) -> builtins.list[UUID]:
        self._ensure_initialized()
        if not 1 <= limit <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        async with aiosqlite.connect(self.path) as database:
            cursor = await database.execute(
                """
                SELECT run_id, MAX(timestamp) AS latest
                FROM run_events GROUP BY run_id ORDER BY latest DESC LIMIT ?
                """,
                (limit,),
            )
            rows = await cursor.fetchall()
        return [UUID(row[0]) for row in rows]


class SQLiteCheckpointStore(_SQLiteStore):
    """Latest-checkpoint store guarded by compare-and-swap revisions."""

    async def _create_tables(self, database: aiosqlite.Connection) -> None:
        await database.execute(
            """
            CREATE TABLE IF NOT EXISTS run_checkpoints (
                run_id TEXT PRIMARY KEY,
                revision INTEGER NOT NULL CHECK (revision >= 0),
                checkpoint_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    async def save(self, checkpoint: Checkpoint, *, expected_revision: int | None) -> None:
        self._ensure_initialized()
        async with aiosqlite.connect(self.path) as database:
            await database.execute("BEGIN IMMEDIATE")
            cursor = await database.execute(
                "SELECT revision FROM run_checkpoints WHERE run_id = ?",
                (str(checkpoint.run_id),),
            )
            row = await cursor.fetchone()
            current = int(row[0]) if row is not None else None
            if current != expected_revision:
                await database.rollback()
                raise CheckpointConflictError(
                    f"checkpoint conflict for {checkpoint.run_id}: "
                    f"expected revision {expected_revision}, found {current}"
                )
            next_revision = 0 if current is None else current + 1
            if checkpoint.revision != next_revision:
                await database.rollback()
                raise CheckpointConflictError(
                    f"checkpoint revision must be {next_revision}, got {checkpoint.revision}"
                )
            if current is None:
                await database.execute(
                    """
                    INSERT INTO run_checkpoints(run_id, revision, checkpoint_json, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        str(checkpoint.run_id),
                        checkpoint.revision,
                        checkpoint.model_dump_json(),
                        checkpoint.created_at.isoformat(),
                    ),
                )
            else:
                result = await database.execute(
                    """
                    UPDATE run_checkpoints
                    SET revision = ?, checkpoint_json = ?, updated_at = ?
                    WHERE run_id = ? AND revision = ?
                    """,
                    (
                        checkpoint.revision,
                        checkpoint.model_dump_json(),
                        checkpoint.created_at.isoformat(),
                        str(checkpoint.run_id),
                        current,
                    ),
                )
                if result.rowcount != 1:
                    await database.rollback()
                    raise CheckpointConflictError("checkpoint compare-and-swap failed")
            await database.commit()

    async def latest(self, run_id: UUID) -> Checkpoint | None:
        self._ensure_initialized()
        async with aiosqlite.connect(self.path) as database:
            cursor = await database.execute(
                "SELECT checkpoint_json FROM run_checkpoints WHERE run_id = ?",
                (str(run_id),),
            )
            row = await cursor.fetchone()
        return Checkpoint.model_validate_json(row[0]) if row is not None else None
