from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID

import httpx
import pytest

from admires_harness.api import create_app
from admires_harness.config import Settings
from admires_harness.services import RunManager
from admires_harness.services.export import MAX_EXPORT_LIMIT


def settings_for(root: Path) -> Settings:
    return Settings(
        database_path=root / "harness.db",
        workspace_root=root / "workspaces",
        model_provider="mock",
        allow_network=False,
    )


async def wait_for_status(manager: RunManager, run_id: UUID, expected: str) -> None:
    for _ in range(200):
        view = await manager.get(run_id)
        if view.status.value == expected:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"run {run_id} did not reach {expected}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_export_returns_ordered_events_with_schema_version(tmp_path: Path) -> None:
    manager = RunManager(settings_for(tmp_path))
    completed = await manager.run_sync("research", "Why use event sourcing?")
    app = create_app(manager=manager)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        response = await client.get(f"/api/runs/{completed.run_id}/export")
        assert response.status_code == 200
        assert "attachment" in response.headers["content-disposition"]
        document = response.json()
        events = (await client.get(f"/api/runs/{completed.run_id}/events")).json()

    assert document["schema_version"] == "1"
    assert document["kind"] == "admires-harness/trace-export"
    assert document["run"]["id"] == str(completed.run_id)
    assert document["run"]["status"] == "succeeded"
    assert document["truncated"] is False
    assert document["event_count"] == len(events) == document["total_events"]
    sequences = [event["sequence"] for event in document["events"]]
    assert sequences == list(range(len(events)))
    assert [event["event_id"] for event in document["events"]] == [
        event["event_id"] for event in events
    ]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_export_is_bounded_by_limit_and_validates_range(tmp_path: Path) -> None:
    manager = RunManager(settings_for(tmp_path))
    completed = await manager.run_sync("research", "Why use event sourcing?")
    total = len(await manager.list_events(completed.run_id))
    assert total > 2
    app = create_app(manager=manager)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        bounded = (
            await client.get(f"/api/runs/{completed.run_id}/export", params={"limit": 2})
        ).json()
        assert bounded["event_count"] == 2
        assert bounded["total_events"] == total
        assert bounded["truncated"] is True
        assert [event["sequence"] for event in bounded["events"]] == [0, 1]

        too_small = await client.get(f"/api/runs/{completed.run_id}/export", params={"limit": 0})
        too_large = await client.get(
            f"/api/runs/{completed.run_id}/export", params={"limit": MAX_EXPORT_LIMIT + 1}
        )
        assert too_small.status_code == 422
        assert too_large.status_code == 422


@pytest.mark.asyncio
@pytest.mark.integration
async def test_export_missing_or_empty_run_is_404(tmp_path: Path) -> None:
    manager = RunManager(settings_for(tmp_path))
    app = create_app(manager=manager)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        missing = await client.get("/api/runs/00000000-0000-0000-0000-000000000000/export")
        assert missing.status_code == 404
        not_a_uuid = await client.get("/api/runs/not-a-uuid/export")
        assert not_a_uuid.status_code == 404


@pytest.mark.asyncio
@pytest.mark.integration
async def test_export_failed_run_keeps_events_and_status(tmp_path: Path) -> None:
    manager = RunManager(settings_for(tmp_path))
    started = await manager.start("coding", "Create a slugify utility")
    for _ in range(200):
        pending = await manager.list_approvals(started.run_id)
        view = await manager.get(started.run_id)
        if pending and view.status.value == "waiting_approval":
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("coding run did not request approval")
    request_id = pending[0]["requestId"]
    assert isinstance(request_id, str)
    failed = await manager.decide_approval(
        started.run_id, UUID(request_id), approved=False, decided_by="integration-test"
    )
    assert failed.status.value == "failed"

    app = create_app(manager=manager)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        response = await client.get(f"/api/runs/{started.run_id}/export")
        assert response.status_code == 200
        document = response.json()

    assert document["run"]["status"] == "failed"
    assert document["run"]["error"]
    event_types = [event["type"] for event in document["events"]]
    assert "run.failed" in event_types


@pytest.mark.asyncio
@pytest.mark.integration
async def test_export_does_not_mutate_run_state(tmp_path: Path) -> None:
    manager = RunManager(settings_for(tmp_path))
    completed = await manager.run_sync("research", "Why use event sourcing?")
    before = [event.event_id for event in await manager.list_events(completed.run_id)]
    app = create_app(manager=manager)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        response = await client.get(f"/api/runs/{completed.run_id}/export")
        assert response.status_code == 200
    after = [event.event_id for event in await manager.list_events(completed.run_id)]
    assert after == before
    view = await manager.get(completed.run_id)
    assert view.status.value == "succeeded"
