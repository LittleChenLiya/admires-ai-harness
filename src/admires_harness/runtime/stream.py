"""Backpressure-aware fan-out for live run events."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import suppress
from uuid import UUID

from admires_harness.contracts import RunEvent

_CLOSED = object()


class EventBroker:
    """Fan out events to bounded per-subscriber queues.

    The durable event store is authoritative, so a slow or abandoned live subscriber is
    disconnected on overflow instead of blocking agent execution. It can reconnect with
    its last sequence cursor and replay the missing events.
    """

    def __init__(self, *, queue_size: int = 256) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self._queue_size = queue_size
        self._subscribers: dict[UUID, set[asyncio.Queue[RunEvent | object]]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def publish(self, event: RunEvent) -> None:
        async with self._lock:
            subscribers = tuple(self._subscribers.get(event.run_id, ()))
        for queue in subscribers:
            if queue.qsize() < self._queue_size:
                queue.put_nowait(event)
            else:
                # Make room for a close marker. The subscriber must reconnect to the
                # durable store rather than exerting backpressure on the run itself.
                await self.detach(event.run_id, queue)
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(_CLOSED)

    async def attach(self, run_id: UUID) -> asyncio.Queue[RunEvent | object]:
        """Register a subscriber before a caller reads durable history."""

        # Reserve one extra slot for the close marker so a normally full event
        # queue can still terminate without blocking the publisher.
        queue: asyncio.Queue[RunEvent | object] = asyncio.Queue(self._queue_size + 1)
        async with self._lock:
            self._subscribers[run_id].add(queue)
        return queue

    async def detach(self, run_id: UUID, queue: asyncio.Queue[RunEvent | object]) -> None:
        """Remove an attached subscriber and release its queue."""

        async with self._lock:
            subscribers = self._subscribers.get(run_id)
            if subscribers is not None:
                subscribers.discard(queue)
                if not subscribers:
                    with suppress(KeyError):
                        del self._subscribers[run_id]

    async def close(self, run_id: UUID) -> None:
        async with self._lock:
            subscribers = tuple(self._subscribers.pop(run_id, ()))
        for queue in subscribers:
            queue.put_nowait(_CLOSED)

    async def subscribe(self, run_id: UUID) -> AsyncIterator[RunEvent]:
        queue = await self.attach(run_id)
        try:
            while True:
                item = await queue.get()
                if item is _CLOSED:
                    break
                yield item  # type: ignore[misc]
        finally:
            await self.detach(run_id, queue)
