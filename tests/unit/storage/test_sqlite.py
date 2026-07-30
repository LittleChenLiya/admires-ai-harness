from uuid import uuid4

import pytest

from admires_harness.contracts import Checkpoint, EventType, RunEvent, RunStatus
from admires_harness.runtime.errors import CheckpointConflictError
from admires_harness.storage import SQLiteCheckpointStore, SQLiteEventStore


@pytest.mark.asyncio
async def test_event_store_persists_and_reopens(tmp_path):
    path = tmp_path / "harness.db"
    run_id, trace_id = uuid4(), uuid4()
    store = SQLiteEventStore(path)
    await store.initialize()
    event = RunEvent(run_id=run_id, sequence=0, type=EventType.RUN_CREATED, trace_id=trace_id)
    await store.append(event)

    reopened = SQLiteEventStore(path)
    await reopened.initialize()
    assert await reopened.list(run_id) == [event]
    assert await reopened.list_run_ids() == [run_id]


@pytest.mark.asyncio
async def test_event_store_rejects_sequence_conflict(tmp_path):
    store = SQLiteEventStore(tmp_path / "harness.db")
    await store.initialize()
    run_id, trace_id = uuid4(), uuid4()
    invalid = RunEvent(run_id=run_id, sequence=1, type=EventType.RUN_STARTED, trace_id=trace_id)
    with pytest.raises(CheckpointConflictError):
        await store.append(invalid)
    with pytest.raises(ValueError):
        await store.list_run_ids(limit=0)


@pytest.mark.asyncio
async def test_checkpoint_store_uses_compare_and_swap(tmp_path):
    path = tmp_path / "harness.db"
    store = SQLiteCheckpointStore(path)
    await store.initialize()
    run_id = uuid4()
    first = Checkpoint(
        run_id=run_id,
        revision=0,
        graph_name="demo",
        graph_version="1",
        next_node="next",
        status=RunStatus.RUNNING,
        state={"count": 1},
    )
    await store.save(first, expected_revision=None)
    assert await store.latest(run_id) == first

    second = first.model_copy(update={"revision": 1, "state": {"count": 2}})
    with pytest.raises(CheckpointConflictError):
        await store.save(second, expected_revision=None)
    await store.save(second, expected_revision=0)

    reopened = SQLiteCheckpointStore(path)
    await reopened.initialize()
    assert await reopened.latest(run_id) == second
