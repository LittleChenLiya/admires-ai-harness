from __future__ import annotations

from uuid import uuid4

import pytest

from admires_harness.contracts import Checkpoint, EventType, RunEvent, RunStatus
from admires_harness.runtime import (
    CheckpointConflictError,
    EventBroker,
    InMemoryCheckpointStore,
    InMemoryEventStore,
)


@pytest.mark.asyncio
async def test_in_memory_stores_enforce_sequences_and_revisions() -> None:
    run_id = uuid4()
    trace_id = uuid4()
    events = InMemoryEventStore()
    await events.append(
        RunEvent(run_id=run_id, sequence=0, type=EventType.RUN_CREATED, trace_id=trace_id)
    )
    with pytest.raises(CheckpointConflictError, match="sequence conflict"):
        await events.append(
            RunEvent(run_id=run_id, sequence=2, type=EventType.RUN_STARTED, trace_id=trace_id)
        )

    checkpoints = InMemoryCheckpointStore()
    first = Checkpoint(
        run_id=run_id,
        revision=0,
        graph_name="x",
        graph_version="1",
        next_node="a",
        status=RunStatus.PAUSED,
        state={},
    )
    await checkpoints.save(first, expected_revision=None)
    with pytest.raises(CheckpointConflictError, match="expected revision"):
        await checkpoints.save(first, expected_revision=None)
    with pytest.raises(CheckpointConflictError, match="must be 1"):
        await checkpoints.save(first, expected_revision=0)


@pytest.mark.asyncio
async def test_event_broker_subscribe_and_close() -> None:
    with pytest.raises(ValueError, match="positive"):
        EventBroker(queue_size=0)
    broker = EventBroker()
    run_id = uuid4()
    event = RunEvent(run_id=run_id, sequence=0, type=EventType.RUN_STARTED, trace_id=uuid4())

    async def collect():  # type: ignore[no-untyped-def]
        return [item async for item in broker.subscribe(run_id)]

    import asyncio

    task = asyncio.create_task(collect())
    await asyncio.sleep(0)
    await broker.publish(event)
    await broker.close(run_id)
    assert await task == [event]


@pytest.mark.asyncio
async def test_slow_broker_subscriber_does_not_block_publish_or_close() -> None:
    import asyncio

    broker = EventBroker(queue_size=1)
    run_id = uuid4()
    queue = await broker.attach(run_id)
    events = [
        RunEvent(run_id=run_id, sequence=index, type=EventType.LOG, trace_id=uuid4())
        for index in range(3)
    ]

    await asyncio.wait_for(broker.publish(events[0]), timeout=0.1)
    await asyncio.wait_for(broker.publish(events[1]), timeout=0.1)
    await asyncio.wait_for(broker.publish(events[2]), timeout=0.1)
    await asyncio.wait_for(broker.close(run_id), timeout=0.1)
    # Overflow replaces queued live data with a reconnect marker. Durable replay
    # remains the lossless source of truth.
    assert not isinstance(await queue.get(), RunEvent)
