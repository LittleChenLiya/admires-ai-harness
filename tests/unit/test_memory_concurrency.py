"""Deterministic concurrency, failure, and lifecycle tests for SQLiteMemoryStore."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest

from admires_harness.memory import (
    MemoryCorruptionError,
    MemoryRecord,
    MemoryStoreClosedError,
    MemoryStoreLockedError,
    SQLiteMemoryStore,
)


def _record(
    namespace: str,
    content: str = "fact",
    embedding: tuple[float, ...] = (1.0, 0.0),
    **overrides: object,
) -> MemoryRecord:
    return MemoryRecord(namespace=namespace, content=content, embedding=embedding, **overrides)


@pytest.mark.asyncio
async def test_concurrent_initialize_across_instances_is_safe(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    stores = [SQLiteMemoryStore(path) for _ in range(8)]

    await asyncio.gather(*(store.initialize() for store in stores))
    # Initialization is idempotent, including across instances.
    await asyncio.gather(*(store.initialize() for store in stores))

    record = _record("n")
    await stores[3].put(record)
    assert await stores[7].get(record.id, namespace="n") == record


@pytest.mark.asyncio
async def test_concurrent_upsert_delete_search_stay_consistent(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    writer, reader = SQLiteMemoryStore(path), SQLiteMemoryStore(path)
    await writer.initialize()
    await reader.initialize()

    shared = _record("shared", content="original")
    await writer.put(shared)

    async def upsert(index: int) -> None:
        await writer.put(
            _record("shared", content=f"revision-{index}", id=shared.id, embedding=(1.0, 0.0))
        )

    async def remove() -> bool:
        return await writer.delete(shared.id, namespace="shared")

    async def scan() -> int:
        return len(await reader.search((1.0, 0.0), namespace="shared", limit=100))

    # Concurrent upserts of one primary key converge to a single committed row.
    await asyncio.gather(*(upsert(index) for index in range(6)))
    fetched = await reader.get(shared.id, namespace="shared")
    assert fetched is not None
    assert fetched.content.startswith("revision-")

    # Concurrent deletes and searches never observe half-committed state.
    results = await asyncio.gather(*(remove() for _ in range(3)), *(scan() for _ in range(4)))
    deletes = [result for result in results if isinstance(result, bool)]
    assert deletes.count(True) == 1
    assert all(count in (0, 1) for count in results if isinstance(count, int))
    assert await reader.get(shared.id, namespace="shared") is None

    # Successful transactions are fully visible to every instance afterwards.
    final = _record("shared", content="final", id=shared.id)
    await writer.put(final)
    assert await reader.get(shared.id, namespace="shared") == final


@pytest.mark.asyncio
async def test_dimension_conflict_rolls_back_without_polluting_reads(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    await store.initialize()
    original = _record("n", embedding=(1.0, 0.0))
    await store.put(original)

    conflicting = _record("n", content="bad", embedding=(1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="dimension"):
        await store.put(conflicting)

    # The failed transaction left nothing behind.
    assert await store.get(conflicting.id, namespace="n") is None
    assert await store.get(original.id, namespace="n") == original
    assert [hit.record.id for hit in await store.search((1.0, 0.0), namespace="n")] == [original.id]


@pytest.mark.asyncio
async def test_concurrent_dimension_race_has_a_single_winner(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    first, second = SQLiteMemoryStore(path), SQLiteMemoryStore(path)
    await asyncio.gather(first.initialize(), second.initialize())

    outcomes = await asyncio.gather(
        first.put(_record("race", embedding=(1.0, 0.0))),
        second.put(_record("race", embedding=(1.0, 0.0, 0.0))),
        return_exceptions=True,
    )

    errors = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    hits = await first.search((1.0, 0.0, 0.0), namespace="race", limit=10)
    hits += await first.search((1.0, 0.0), namespace="race", limit=10)
    assert len(hits) == 1


@pytest.mark.asyncio
async def test_duplicate_primary_key_upserts_instead_of_failing(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    await store.initialize()
    record = _record("n", content="v1")
    await store.put(record)
    await store.put(_record("n", content="v2", id=record.id))

    fetched = await store.get(record.id, namespace="n")
    assert fetched is not None
    assert fetched.content == "v2"


@pytest.mark.asyncio
async def test_database_locked_surfaces_stable_error_and_recovers(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    store = SQLiteMemoryStore(path, busy_timeout=0.05)
    await store.initialize()

    blocker = sqlite3.connect(path)
    blocker.execute("BEGIN IMMEDIATE")
    blocker.execute(
        "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (str(uuid4()), "n", "held", "test", "{}", "[1.0]", 1, "t", "t"),
    )
    try:
        with pytest.raises(MemoryStoreLockedError, match="locked"):
            await store.put(_record("n"))
    finally:
        blocker.rollback()
        blocker.close()

    # The lock failure does not poison the store.
    record = _record("n")
    await store.put(record)
    assert await store.get(record.id, namespace="n") == record


@pytest.mark.asyncio
async def test_cancellation_leaves_no_partial_state_and_store_usable(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    await store.initialize()
    cancelled = _record("n", content="cancelled")

    task = asyncio.create_task(store.put(cancelled))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert await store.get(cancelled.id, namespace="n") is None
    record = _record("n", content="after")
    await store.put(record)
    assert await store.get(record.id, namespace="n") == record


@pytest.mark.asyncio
async def test_corrupt_rows_raise_diagnosable_error_and_are_removable(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    store = SQLiteMemoryStore(path)
    await store.initialize()
    good = _record("n", content="good")
    await store.put(good)

    corrupt_id = uuid4()
    database = sqlite3.connect(path)
    database.execute(
        "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (str(corrupt_id), "n", "broken", "test", "not-json", "[1.0, 0.0]", 2, "t", "t"),
    )
    database.commit()
    database.close()

    with pytest.raises(MemoryCorruptionError, match="corrupt memory row"):
        await store.get(corrupt_id, namespace="n")
    with pytest.raises(MemoryCorruptionError, match="corrupt memory row"):
        await store.search((1.0, 0.0), namespace="n")

    # The corrupt row can be deleted, after which reads are clean again.
    assert await store.delete(corrupt_id, namespace="n")
    assert await store.get(corrupt_id, namespace="n") is None
    assert [hit.record.id for hit in await store.search((1.0, 0.0), namespace="n")] == [good.id]


@pytest.mark.asyncio
async def test_reopen_wal_database_and_close_semantics(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    async with SQLiteMemoryStore(path) as store:
        record = _record("n", content="durable")
        await store.put(record)
    assert store.closed
    with pytest.raises(MemoryStoreClosedError, match="closed"):
        await store.get(record.id, namespace="n")
    with pytest.raises(MemoryStoreClosedError, match="closed"):
        await store.initialize()

    reopened = SQLiteMemoryStore(path)
    await reopened.initialize()
    assert await reopened.get(record.id, namespace="n") == record
    hits = await reopened.search((1.0, 0.0), namespace="n")
    assert [hit.record.id for hit in hits] == [record.id]


@pytest.mark.asyncio
async def test_namespace_cross_access_stays_isolated(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    await store.initialize()
    shared_id = uuid4()
    await store.put(_record("tenant-a", content="alpha", id=shared_id, embedding=(1.0, 0.0)))
    await store.put(_record("tenant-b", content="beta", id=shared_id, embedding=(1.0, 0.0)))

    hits = await store.search((1.0, 0.0), namespace="tenant-a")
    assert [hit.record.content for hit in hits] == ["alpha"]
    assert await store.delete(shared_id, namespace="tenant-a")
    assert await store.get(shared_id, namespace="tenant-a") is None
    remaining = await store.get(shared_id, namespace="tenant-b")
    assert remaining is not None
    assert remaining.content == "beta"


@pytest.mark.asyncio
async def test_search_ranking_is_score_descending_then_id(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    await store.initialize()
    high = _record("n", content="high", embedding=(1.0, 0.0))
    tie_a = _record("n", content="tie-a", embedding=(0.5, 0.5))
    tie_b = _record("n", content="tie-b", embedding=(0.5, 0.5))
    for record in (tie_b, high, tie_a):
        await store.put(record)

    hits = await store.search((1.0, 0.0), namespace="n", limit=10)

    assert hits[0].record.id == high.id
    assert [hit.record.id for hit in hits[1:]] == sorted([tie_a.id, tie_b.id], key=str)
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


@pytest.mark.asyncio
async def test_connections_and_cursors_converge(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    store = SQLiteMemoryStore(path)
    await store.initialize()
    record = _record("n")
    await store.put(record)
    await store.get(record.id, namespace="n")
    await store.search((1.0, 0.0), namespace="n")
    await store.delete(record.id, namespace="n")
    await store.aclose()
    await store.aclose()

    # No leaked connection or cursor keeps the database files open.
    for suffix in ("", "-wal", "-shm"):
        artifact = tmp_path / f"memory.db{suffix}"
        if artifact.exists():
            artifact.unlink()
    assert not path.exists()
