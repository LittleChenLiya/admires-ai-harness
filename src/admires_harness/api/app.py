"""FastAPI surface for runs, replay, approvals, export, and live events."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, cast
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from admires_harness.agents import list_demos
from admires_harness.config import Settings
from admires_harness.contracts import JsonValue, RunEvent
from admires_harness.observability import TraceExport
from admires_harness.observability.export import DEFAULT_EXPORT_LIMIT, MAX_EXPORT_LIMIT
from admires_harness.services.runs import RunManager, as_camel


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class StartRunRequest(APIModel):
    demo: str = Field(min_length=1, max_length=64)
    input: str = Field(min_length=1, max_length=20_000)


class ApprovalDecisionRequest(APIModel):
    approved: bool
    decided_by: str = Field(alias="decidedBy", min_length=1, max_length=128)
    comment: str | None = Field(default=None, max_length=2_000)


def _manager(request: Request) -> RunManager:
    return cast(RunManager, request.app.state.manager)


ManagerDependency = Annotated[RunManager, Depends(_manager)]


def _uuid(value: str) -> UUID:
    try:
        return UUID(value)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc


def _event_json(event: RunEvent) -> dict[str, JsonValue]:
    return event.model_dump(mode="json")


def create_app(settings: Settings | None = None, *, manager: RunManager | None = None) -> FastAPI:
    resolved_settings = settings or Settings()
    resolved_manager = manager or RunManager(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.manager = resolved_manager
        await resolved_manager.initialize()
        try:
            yield
        finally:
            await resolved_manager.close()

    app = FastAPI(
        title="Admire's AI Harness API",
        version="1.0.0",
        description="Run and inspect safe-by-default reference agents.",
        lifespan=lifespan,
    )
    app.state.manager = resolved_manager
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/api/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "modelProvider": resolved_settings.model_provider}

    @app.get("/api/demos", tags=["demos"])
    async def demos() -> list[dict[str, JsonValue]]:
        return [demo.as_api() for demo in list_demos()]

    @app.get("/api/runs", tags=["runs"])
    async def runs(
        service: ManagerDependency,
        limit: int = Query(default=50, ge=1, le=200),
    ) -> list[dict[str, JsonValue]]:
        return [as_camel(view) for view in await service.list_runs(limit=limit)]

    @app.post("/api/runs", status_code=status.HTTP_202_ACCEPTED, tags=["runs"])
    async def start_run(body: StartRunRequest, service: ManagerDependency) -> dict[str, JsonValue]:
        try:
            return as_camel(await service.start(body.demo, body.input))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}", tags=["runs"])
    async def get_run(run_id: str, service: ManagerDependency) -> dict[str, JsonValue]:
        try:
            return as_camel(await service.get(_uuid(run_id)))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc

    @app.get("/api/runs/{run_id}/events", tags=["runs"])
    async def get_events(
        run_id: str,
        service: ManagerDependency,
        after: int = Query(default=-1, ge=-1),
    ) -> list[dict[str, JsonValue]]:
        events = await service.list_events(_uuid(run_id), after=after)
        if not events and not await service.list_events(_uuid(run_id)):
            raise HTTPException(status_code=404, detail="run not found")
        return [_event_json(event) for event in events]

    @app.get("/api/runs/{run_id}/replay", tags=["runs"])
    async def replay_run(run_id: str, service: ManagerDependency) -> dict[str, JsonValue]:
        try:
            return await service.replay(_uuid(run_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc

    @app.get("/api/runs/{run_id}/export", tags=["runs"])
    async def export_run(
        run_id: str,
        service: ManagerDependency,
        limit: int = Query(default=DEFAULT_EXPORT_LIMIT, ge=1, le=MAX_EXPORT_LIMIT),
    ) -> TraceExport:
        """Download a bounded, redacted JSON export of a run's persisted events.

        The document is schema-versioned and preserves event sequence order.
        Exporting is read-only: it never invokes a model and never mutates run
        state. A run with no persisted events (including a missing run) is a 404;
        failed runs export normally.
        """

        try:
            return await service.export_trace(_uuid(run_id), limit=limit)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc

    @app.get("/api/runs/{run_id}/stream", tags=["runs"])
    async def stream_run(
        run_id: str,
        service: ManagerDependency,
        after: int = Query(default=-1, ge=-1),
    ) -> StreamingResponse:
        parsed = _uuid(run_id)
        if not await service.list_events(parsed):
            raise HTTPException(status_code=404, detail="run not found")

        async def generate() -> AsyncIterator[str]:
            async for event in service.stream(parsed, after=after):
                payload = json.dumps(_event_json(event), separators=(",", ":"))
                yield f"id: {event.sequence}\ndata: {payload}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.post("/api/runs/{run_id}/cancel", tags=["runs"])
    async def cancel_run(run_id: str, service: ManagerDependency) -> dict[str, JsonValue]:
        try:
            return as_camel(await service.cancel(_uuid(run_id)))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc

    @app.post("/api/runs/{run_id}/resume", tags=["runs"])
    async def resume_run(run_id: str, service: ManagerDependency) -> dict[str, JsonValue]:
        try:
            return as_camel(await service.resume(_uuid(run_id)))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/approvals", tags=["approvals"])
    async def approvals(run_id: str, service: ManagerDependency) -> list[dict[str, JsonValue]]:
        try:
            return await service.list_approvals(_uuid(run_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc

    @app.post("/api/runs/{run_id}/approvals/{request_id}", tags=["approvals"])
    async def decide_approval(
        run_id: str,
        request_id: str,
        body: ApprovalDecisionRequest,
        service: ManagerDependency,
    ) -> dict[str, JsonValue]:
        try:
            return as_camel(
                await service.decide_approval(
                    _uuid(run_id),
                    _uuid(request_id),
                    approved=body.approved,
                    decided_by=body.decided_by,
                    comment=body.comment,
                )
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="approval request not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    web_dist = resolved_settings.web_dist.resolve()
    if web_dist.is_dir():
        assets = web_dist / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str) -> FileResponse:
            del path
            return FileResponse(web_dist / "index.html")

    return app
