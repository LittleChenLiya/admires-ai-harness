"""API contract tests for the run trace export endpoint."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest

from admires_harness.api import create_app
from admires_harness.config import Settings
from admires_harness.contracts import EventType, JsonValue, RunEvent
from admires_harness.services import RunManager


def settings_for(root: Path) -> Settings:
    return Settings(
        database_path=root / "harness.db",
        workspace_root=root / "workspaces",
        model_provider="mock",
        allow_network=False,
    )


def persisted_event(
    run_id: UUID,
    trace_id: UUID,
    sequence: int,
    event_type: EventType,
    payload: dict[str, JsonValue] | None = None,
) -> RunEvent:
    return RunEvent(
        run_id=run_id,
        trace_id=trace_id,
        sequence=sequence,
        type=event_type,
        payload=payload or {},
    )


async def seed_failed_run(manager: RunManager) -> UUID:
    """Persist a failed run directly; no model or tool is ever invoked."""

    await manager.initialize()
    run_id, trace_id = uuid4(), uuid4()
    events = [
        persisted_event(
            run_id,
            trace_id,
            0,
            EventType.RUN_CREATED,
            payload={
                "graph_name": "research-agent",
                "graph_version": "1",
                "initial_state": {"input": "question", "demo": "research"},
            },
        ),
        persisted_event(run_id, trace_id, 1, EventType.RUN_STARTED),
        persisted_event(
            run_id,
            trace_id,
            2,
            EventType.RUN_FAILED,
            payload={
                "status": "failed",
                "error": "boom",
                "provider": {"api_key": "sk-should-not-leak"},
            },
        ),
    ]
    for event in events:
        await manager.events.append(event)
    return run_id


@pytest.mark.integration
async def test_export_returns_ordered_versioned_bounded_events(tmp_path: Path) -> None:
    manager = RunManager(settings_for(tmp_path))
    app = create_app(manager=manager)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        created = (
            await client.post(
                "/api/runs",
                json={"demo": "research", "input": "Why use event sourcing?"},
            )
        ).json()
        run_id = created["id"]
        for _ in range(200):
            run = (await client.get(f"/api/runs/{run_id}")).json()
            if run["status"] == "succeeded":
                break
        else:
            raise AssertionError("run did not succeed")

        document = (await client.get(f"/api/runs/{run_id}/export")).json()
        assert document["schema_version"] == "trace-export/1"
        assert document["run_id"] == run_id
        assert document["status"] == "succeeded"
        sequences = [event["sequence"] for event in document["events"]]
        assert sequences == sorted(sequences)
        assert document["event_count"] == len(sequences)
        assert document["total_events"] == len(sequences)
        assert document["truncated"] is False
        assert all(event["schema_version"] == "1" for event in document["events"])

        bounded = (await client.get(f"/api/runs/{run_id}/export", params={"limit": 2})).json()
        assert bounded["event_count"] == 2
        assert bounded["total_events"] == len(sequences)
        assert bounded["truncated"] is True
        assert [event["sequence"] for event in bounded["events"]] == sequences[:2]

        rejected = await client.get(f"/api/runs/{run_id}/export", params={"limit": 0})
        assert rejected.status_code == 422


@pytest.mark.integration
async def test_export_failed_run_redacts_nested_secrets(tmp_path: Path) -> None:
    manager = RunManager(settings_for(tmp_path))
    run_id = await seed_failed_run(manager)
    app = create_app(manager=manager)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        response = await client.get(f"/api/runs/{run_id}/export")

    assert response.status_code == 200
    document = response.json()
    assert document["status"] == "failed"
    failed = document["events"][-1]
    assert failed["type"] == "run.failed"
    assert failed["payload"]["provider"] == {"api_key": "[redacted]"}
    assert failed["payload"]["error"] == "boom"
    assert "sk-should-not-leak" not in response.text


@pytest.mark.integration
async def test_export_missing_or_empty_run_is_not_found(tmp_path: Path) -> None:
    manager = RunManager(settings_for(tmp_path))
    app = create_app(manager=manager)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        missing = await client.get(f"/api/runs/{uuid4()}/export")
        invalid = await client.get("/api/runs/not-a-uuid/export")

    assert missing.status_code == 404
    assert missing.json()["detail"] == "run not found"
    assert invalid.status_code == 404


@pytest.mark.integration
async def test_export_does_not_mutate_run_state(tmp_path: Path) -> None:
    manager = RunManager(settings_for(tmp_path))
    run_id = await seed_failed_run(manager)
    app = create_app(manager=manager)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        before = (await client.get(f"/api/runs/{run_id}/events")).json()
        await client.get(f"/api/runs/{run_id}/export")
        after = (await client.get(f"/api/runs/{run_id}/events")).json()

    assert before == after
