from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from admires_harness.api import create_app
from admires_harness.config import Settings
from admires_harness.contracts import EventType, RunEvent
from admires_harness.observability import REDACTED
from admires_harness.services import RunManager


def settings_for(root: Path) -> Settings:
    return Settings(
        database_path=root / "harness.db",
        workspace_root=root / "workspaces",
        model_provider="mock",
        allow_network=False,
    )


async def wait_for_terminal(client: httpx.AsyncClient, run_id: str) -> dict[str, object]:
    for _ in range(400):
        response = await client.get(f"/api/runs/{run_id}")
        response.raise_for_status()
        run = response.json()
        if run["status"] in {"succeeded", "failed", "cancelled", "timed_out", "budget_exceeded"}:
            return run
        await asyncio.sleep(0.01)
    raise AssertionError(f"run {run_id} did not reach a terminal state")


async def started_run(client: httpx.AsyncClient) -> str:
    created = await client.post(
        "/api/runs", json={"demo": "research", "input": "export ordering probe"}
    )
    assert created.status_code == 202
    run_id = created.json()["id"]
    await wait_for_terminal(client, run_id)
    return run_id


@pytest.mark.asyncio
@pytest.mark.integration
async def test_export_preserves_sequence_and_schema_version(tmp_path: Path) -> None:
    app = create_app(manager=RunManager(settings_for(tmp_path)))
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        run_id = await started_run(client)
        response = await client.get(f"/api/runs/{run_id}/export")

    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    export = response.json()
    assert export["schema_version"] == "1"
    assert export["run_id"] == run_id
    assert export["status"] == "succeeded"
    assert export["truncated"] is False
    events = export["events"]
    assert export["event_count"] == len(events) > 0
    assert [event["sequence"] for event in events] == list(range(len(events)))
    assert {event["schema_version"] for event in events} == {"1"}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_export_is_bounded_and_marks_truncation(tmp_path: Path) -> None:
    app = create_app(manager=RunManager(settings_for(tmp_path)))
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        run_id = await started_run(client)
        full = (await client.get(f"/api/runs/{run_id}/export")).json()
        bounded = (await client.get(f"/api/runs/{run_id}/export", params={"maxEvents": 2})).json()

    assert full["event_count"] > 2
    assert bounded["event_count"] == 2
    assert bounded["truncated"] is True
    assert [event["sequence"] for event in bounded["events"]] == [0, 1]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_export_rejects_out_of_range_bounds(tmp_path: Path) -> None:
    app = create_app(manager=RunManager(settings_for(tmp_path)))
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        run_id = await started_run(client)
        zero = await client.get(f"/api/runs/{run_id}/export", params={"maxEvents": 0})
        huge = await client.get(f"/api/runs/{run_id}/export", params={"maxEvents": 99999})

    assert zero.status_code == 422
    assert huge.status_code == 422


@pytest.mark.asyncio
@pytest.mark.integration
async def test_export_redacts_sensitive_fields_at_any_depth(tmp_path: Path) -> None:
    manager = RunManager(settings_for(tmp_path))
    app = create_app(manager=manager)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        run_id, trace_id = uuid4(), uuid4()
        await manager.events.append(
            RunEvent(
                run_id=run_id,
                trace_id=trace_id,
                sequence=0,
                type=EventType.RUN_CREATED,
                payload={
                    "graph_name": "research-agent",
                    "graph_version": "1",
                    "initial_state": {"input": "secret probe", "demo": "research"},
                },
            )
        )
        await manager.events.append(
            RunEvent(
                run_id=run_id,
                trace_id=trace_id,
                sequence=1,
                type=EventType.TOOL_COMPLETED,
                payload={
                    "tool": "workspace.write",
                    "api_key": "sk-live-secret",
                    "result": {
                        "content": "visible",
                        "credentials": {"token": "nested-token", "note": "kept"},
                    },
                    "attempts": [{"authorization": "Bearer abc"}, {"ok": True}],
                },
            )
        )
        response = await client.get(f"/api/runs/{run_id}/export")

    assert response.status_code == 200
    export = response.json()
    payload = export["events"][1]["payload"]
    assert payload["api_key"] == REDACTED
    assert payload["result"]["credentials"] == REDACTED
    assert payload["result"]["content"] == "visible"
    assert payload["attempts"][0]["authorization"] == REDACTED
    assert payload["attempts"][1] == {"ok": True}
    body = response.text
    assert "sk-live-secret" not in body
    assert "nested-token" not in body
    assert "Bearer abc" not in body


@pytest.mark.asyncio
@pytest.mark.integration
async def test_export_failed_run_keeps_terminal_status(tmp_path: Path) -> None:
    manager = RunManager(settings_for(tmp_path))
    app = create_app(manager=manager)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        run_id, trace_id = uuid4(), uuid4()
        await manager.events.append(
            RunEvent(
                run_id=run_id,
                trace_id=trace_id,
                sequence=0,
                type=EventType.RUN_CREATED,
                payload={
                    "graph_name": "research-agent",
                    "graph_version": "1",
                    "initial_state": {"input": "failing probe", "demo": "research"},
                },
            )
        )
        await manager.events.append(
            RunEvent(
                run_id=run_id,
                trace_id=trace_id,
                sequence=1,
                type=EventType.RUN_FAILED,
                payload={"status": "failed", "error": "boom"},
            )
        )
        response = await client.get(f"/api/runs/{run_id}/export")

    assert response.status_code == 200
    export = response.json()
    assert export["status"] == "failed"
    assert export["event_count"] == 2
    assert export["events"][-1]["payload"]["error"] == "boom"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_export_missing_or_empty_run_returns_404(tmp_path: Path) -> None:
    app = create_app(manager=RunManager(settings_for(tmp_path)))
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        missing = await client.get(f"/api/runs/{uuid4()}/export")
        malformed = await client.get("/api/runs/not-a-uuid/export")

    assert missing.status_code == 404
    assert malformed.status_code == 404
