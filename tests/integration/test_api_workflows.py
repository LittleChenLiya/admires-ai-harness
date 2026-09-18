from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID

import httpx
import pytest

from admires_harness.api import create_app
from admires_harness.config import Settings
from admires_harness.contracts import (
    ApprovalDecision,
    EventType,
    RunEvent,
    RunStatus,
)
from admires_harness.services import RunManager


def settings_for(root: Path) -> Settings:
    return Settings(
        database_path=root / "harness.db",
        workspace_root=root / "workspaces",
        model_provider="mock",
        allow_network=False,
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_built_console_is_served_from_configured_distribution(tmp_path: Path) -> None:
    web_dist = tmp_path / "console"
    assets = web_dist / "assets"
    assets.mkdir(parents=True)
    (web_dist / "index.html").write_text("<main>packaged console</main>", encoding="utf-8")
    (assets / "app.js").write_text("console.log('ready')", encoding="utf-8")
    settings = settings_for(tmp_path).model_copy(update={"web_dist": web_dist})
    app = create_app(settings=settings, manager=RunManager(settings))
    transport = httpx.ASGITransport(app=app)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        index = await client.get("/")
        asset = await client.get("/assets/app.js")

    assert index.status_code == 200
    assert "packaged console" in index.text
    assert asset.status_code == 200
    assert "ready" in asset.text


async def wait_for_status(
    client: httpx.AsyncClient, run_id: str, expected: str
) -> dict[str, object]:
    for _ in range(200):
        response = await client.get(f"/api/runs/{run_id}")
        response.raise_for_status()
        run = response.json()
        if run["status"] == expected:
            return run
        await asyncio.sleep(0.01)
    raise AssertionError(f"run {run_id} did not reach {expected}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_research_api_events_cursor_and_replay(tmp_path: Path) -> None:
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
        run = await wait_for_status(client, created["id"], "succeeded")
        assert "Sources" in str(run["output"])
        events = (await client.get(f"/api/runs/{created['id']}/events")).json()
        assert events[0]["sequence"] == 0
        tail = (
            await client.get(
                f"/api/runs/{created['id']}/events", params={"after": events[-2]["sequence"]}
            )
        ).json()
        assert [event["sequence"] for event in tail] == [events[-1]["sequence"]]
        replay = (await client.get(f"/api/runs/{created['id']}/replay")).json()
        assert replay["status"] == "succeeded"
        assert len(replay["events"]) == len(events)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.e2e
async def test_coding_approval_survives_restart_and_is_single_use(tmp_path: Path) -> None:
    first = RunManager(settings_for(tmp_path))
    await first.initialize()
    started = await first.start("coding", "Create a slugify utility")
    for _ in range(200):
        view = await first.get(started.run_id)
        if view.status.value == "waiting_approval":
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("coding run did not request approval")
    pending = await first.list_approvals(started.run_id)
    assert len(pending) == 1
    request_id = pending[0]["requestId"]
    assert isinstance(request_id, str)
    assert not (tmp_path / "workspaces" / str(started.run_id) / "slugify.py").exists()

    await first.wait(started.run_id)
    restarted = RunManager(settings_for(tmp_path))
    completed = await restarted.decide_approval(
        started.run_id,
        UUID(request_id),
        approved=True,
        decided_by="integration-test",
    )
    assert completed.status.value == "succeeded"
    assert (tmp_path / "workspaces" / str(started.run_id) / "slugify.py").is_file()
    assert await restarted.list_approvals(started.run_id) == []
    with pytest.raises(ValueError, match="not waiting"):
        await restarted.decide_approval(
            started.run_id,
            UUID(request_id),
            approved=True,
            decided_by="replay",
        )
    event_types = {event.type for event in await restarted.list_events(started.run_id)}
    assert EventType.APPROVAL_CONSUMED in event_types
    assert EventType.TOOL_COMPLETED in event_types


@pytest.mark.asyncio
@pytest.mark.integration
async def test_denied_write_fails_without_creating_artifact(tmp_path: Path) -> None:
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
    result = await manager.decide_approval(
        started.run_id,
        UUID(request_id),
        approved=False,
        decided_by="integration-test",
    )
    assert result.status.value == "failed"
    assert "denied" in str(result.error)
    assert not (tmp_path / "workspaces" / str(started.run_id) / "slugify.py").exists()
    assert await manager.list_approvals(started.run_id) == []


@pytest.mark.asyncio
@pytest.mark.integration
async def test_approval_cannot_cross_an_extra_checkpoint_transition(tmp_path: Path) -> None:
    manager = RunManager(settings_for(tmp_path))
    started = await manager.start("coding", "Create a slugify utility")
    for _ in range(200):
        pending = await manager.list_approvals(started.run_id)
        view = await manager.get(started.run_id)
        if pending and view.status is RunStatus.WAITING_APPROVAL:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("coding run did not request approval")

    request_id = pending[0]["requestId"]
    assert isinstance(request_id, str)
    checkpoint = await manager.checkpoints.latest(started.run_id)
    assert checkpoint is not None
    advanced = checkpoint.model_copy(
        update={
            "revision": checkpoint.revision + 1,
            "status": RunStatus.WAITING_APPROVAL,
        }
    )
    await manager.checkpoints.save(advanced, expected_revision=checkpoint.revision)

    result = await manager.decide_approval(
        started.run_id,
        UUID(request_id),
        approved=True,
        decided_by="integration-test",
    )

    assert result.status is RunStatus.FAILED
    assert "stale for the current checkpoint" in str(result.error)
    assert not (tmp_path / "workspaces" / str(started.run_id) / "slugify.py").exists()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_resolved_decision_can_resume_idempotently_after_restart(tmp_path: Path) -> None:
    first = RunManager(settings_for(tmp_path))
    started = await first.start("coding", "Create a slugify utility")
    for _ in range(200):
        pending = await first.list_approvals(started.run_id)
        view = await first.get(started.run_id)
        if pending and view.status is RunStatus.WAITING_APPROVAL:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("coding run did not request approval")

    request_id = pending[0]["requestId"]
    assert isinstance(request_id, str)
    gate = await first._approval_gate(started.run_id)
    decision = ApprovalDecision(
        request_id=UUID(request_id), approved=True, decided_by="interrupted-reviewer"
    )
    await gate.resolve(decision)
    history = await first.events.list(started.run_id)
    await first.events.append(
        RunEvent(
            run_id=started.run_id,
            sequence=history[-1].sequence + 1,
            type=EventType.APPROVAL_RESOLVED,
            trace_id=history[0].trace_id,
            payload={"decision": decision.model_dump(mode="json")},
        )
    )

    restarted = RunManager(settings_for(tmp_path))
    recoverable = await restarted.list_approvals(started.run_id)
    assert len(recoverable) == 1
    assert recoverable[0]["decided"] is True
    assert recoverable[0]["approved"] is True
    with pytest.raises(ValueError, match="different decision"):
        await restarted.decide_approval(
            started.run_id,
            UUID(request_id),
            approved=False,
            decided_by="attacker",
        )

    completed = await restarted.decide_approval(
        started.run_id,
        UUID(request_id),
        approved=True,
        decided_by="retrying-reviewer",
    )
    assert completed.status is RunStatus.SUCCEEDED
    assert (tmp_path / "workspaces" / str(started.run_id) / "slugify.py").is_file()
    assert await restarted.list_approvals(started.run_id) == []
    events = await restarted.list_events(started.run_id)
    assert sum(event.type is EventType.APPROVAL_RESOLVED for event in events) == 1
    assert sum(event.type is EventType.APPROVAL_CONSUMED for event in events) == 1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_cursor_is_finite_for_terminal_run_and_missing_run_is_404(tmp_path: Path) -> None:
    manager = RunManager(settings_for(tmp_path))
    completed = await manager.run_sync("research", "Why use event sourcing?")
    app = create_app(manager=manager)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        events = await manager.list_events(completed.run_id)
        response = await client.get(
            f"/api/runs/{completed.run_id}/stream",
            params={"after": events[-2].sequence},
        )
        assert response.status_code == 200
        assert "run.completed" in response.text
        missing = await client.get("/api/runs/00000000-0000-0000-0000-000000000000/stream")
        assert missing.status_code == 404


@pytest.mark.asyncio
@pytest.mark.integration
async def test_trace_export_is_ordered_versioned_bounded_and_redacted(tmp_path: Path) -> None:
    manager = RunManager(settings_for(tmp_path))
    completed = await manager.run_sync("research", "Why use event sourcing?")
    history = await manager.list_events(completed.run_id)
    await manager.events.append(
        RunEvent(
            run_id=completed.run_id,
            sequence=history[-1].sequence + 1,
            type=EventType.LOG,
            trace_id=history[0].trace_id,
            payload={
                "message": "sanitized",
                "context": {"api_key": "sk-should-not-leak", "input_tokens": 7},
            },
        )
    )
    app = create_app(manager=manager)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        response = await client.get(f"/api/runs/{completed.run_id}/export")
        assert response.status_code == 200
        assert "attachment" in response.headers["content-disposition"]
        export = response.json()
        assert export["schema_version"] == "1"
        assert export["status"] == "succeeded"
        assert export["truncated"] is False
        assert export["total_events"] == len(export["events"])
        sequences = [event["sequence"] for event in export["events"]]
        assert sequences == sorted(sequences)
        context = export["events"][-1]["payload"]["context"]
        assert context["api_key"] == "[REDACTED]"
        assert context["input_tokens"] == 7
        assert "sk-should-not-leak" not in response.text

        bounded = (
            await client.get(f"/api/runs/{completed.run_id}/export", params={"limit": 1})
        ).json()
        assert bounded["truncated"] is True
        assert len(bounded["events"]) == 1
        assert bounded["total_events"] == export["total_events"]

        missing = await client.get("/api/runs/00000000-0000-0000-0000-000000000000/export")
        assert missing.status_code == 404


@pytest.mark.asyncio
@pytest.mark.integration
async def test_trace_export_covers_failed_runs_without_mutating_them(tmp_path: Path) -> None:
    manager = RunManager(settings_for(tmp_path))
    started = await manager.start("coding", "Create a slugify utility")
    for _ in range(200):
        pending = await manager.list_approvals(started.run_id)
        if pending:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("coding run did not request approval")
    request_id = pending[0]["requestId"]
    assert isinstance(request_id, str)
    failed = await manager.decide_approval(
        started.run_id, UUID(request_id), approved=False, decided_by="integration-test"
    )
    assert failed.status is RunStatus.FAILED
    before = await manager.list_events(started.run_id)

    app = create_app(manager=manager)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        export = (await client.get(f"/api/runs/{started.run_id}/export")).json()

    assert export["schema_version"] == "1"
    assert export["status"] == "failed"
    assert export["total_events"] == len(before)
    after = await manager.list_events(started.run_id)
    assert [event.event_id for event in after] == [event.event_id for event in before]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_cancel_paused_run_survives_restart(tmp_path: Path) -> None:
    manager = RunManager(settings_for(tmp_path))
    started = await manager.start("coding", "Create a slugify utility")
    for _ in range(200):
        view = await manager.get(started.run_id)
        pending = await manager.list_approvals(started.run_id)
        if view.status.value == "waiting_approval" and pending:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("coding run did not pause")
    request_id = pending[0]["requestId"]
    assert isinstance(request_id, str)
    restarted = RunManager(settings_for(tmp_path))
    cancelled = await restarted.cancel(started.run_id)
    assert cancelled.status.value == "cancelled"
    assert await restarted.list_approvals(started.run_id) == []
    with pytest.raises(ValueError, match="not waiting"):
        await restarted.decide_approval(
            started.run_id, UUID(request_id), approved=True, decided_by="too-late"
        )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_concurrent_approval_decisions_execute_write_once(tmp_path: Path) -> None:
    manager = RunManager(settings_for(tmp_path))
    started = await manager.start("coding", "Create a slugify utility")
    for _ in range(200):
        view = await manager.get(started.run_id)
        pending = await manager.list_approvals(started.run_id)
        if view.status.value == "waiting_approval" and pending:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("coding run did not pause")
    request_id = pending[0]["requestId"]
    assert isinstance(request_id, str)

    results = await asyncio.gather(
        manager.decide_approval(
            started.run_id, UUID(request_id), approved=True, decided_by="reviewer-a"
        ),
        manager.decide_approval(
            started.run_id, UUID(request_id), approved=True, decided_by="reviewer-b"
        ),
        return_exceptions=True,
    )
    assert sum(not isinstance(result, BaseException) for result in results) == 1
    events = await manager.list_events(started.run_id)
    writes = [
        event
        for event in events
        if event.type is EventType.TOOL_COMPLETED and event.payload.get("tool") == "workspace.write"
    ]
    assert len(writes) == 1
