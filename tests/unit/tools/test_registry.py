from uuid import uuid4

import pytest
from pydantic import BaseModel, ConfigDict

from admires_harness.contracts import RiskLevel, SideEffect, ToolCall, ToolResult
from admires_harness.safety import ToolSafetySpec, WorkspaceSandbox
from admires_harness.tools import ToolDefinition, ToolRegistry, workspace_tools


@pytest.mark.asyncio
async def test_workspace_tools_validate_and_confine_paths(tmp_path):
    registry = workspace_tools(WorkspaceSandbox(tmp_path / "workspace"))
    run_id = uuid4()
    invalid = await registry.execute(
        ToolCall(name="workspace.write", arguments={"unexpected": True}),
        run_id=run_id,
        idempotency_key="invalid",
    )
    escaped = await registry.execute(
        ToolCall(name="workspace.write", arguments={"path": "../secret", "content": "x"}),
        run_id=run_id,
        idempotency_key="escaped",
    )
    assert invalid.error_code == "invalid_arguments"
    assert escaped.error_code == "tool_execution_failed"
    assert not (tmp_path / "secret").exists()


@pytest.mark.asyncio
async def test_write_is_idempotent_and_detects_conflicts(tmp_path):
    registry = workspace_tools(WorkspaceSandbox(tmp_path / "workspace"))
    run_id = uuid4()
    first = await registry.execute(
        ToolCall(name="workspace.write", arguments={"path": "a.txt", "content": "one"}),
        run_id=run_id,
        idempotency_key="stable",
    )
    replay = await registry.execute(
        ToolCall(name="workspace.write", arguments={"path": "a.txt", "content": "one"}),
        run_id=run_id,
        idempotency_key="stable",
    )
    conflict = await registry.execute(
        ToolCall(name="workspace.write", arguments={"path": "a.txt", "content": "two"}),
        run_id=run_id,
        idempotency_key="stable",
    )
    assert first.ok and replay == first and first.receipt
    assert conflict.error_code == "idempotency_conflict"


@pytest.mark.asyncio
async def test_read_list_search_and_unknown_tool(tmp_path):
    workspace = WorkspaceSandbox(tmp_path / "workspace")
    workspace.write_text("notes.txt", "agent harness\ncheckpoints")
    registry = workspace_tools(workspace)
    run_id = uuid4()
    listed = await registry.execute(
        ToolCall(name="workspace.list", arguments={}), run_id=run_id, idempotency_key="list"
    )
    searched = await registry.execute(
        ToolCall(name="workspace.search", arguments={"query": "checkpoint"}),
        run_id=run_id,
        idempotency_key="search",
    )
    unknown = await registry.execute(
        ToolCall(name="missing"), run_id=run_id, idempotency_key="missing"
    )
    assert listed.ok and "notes.txt" in listed.content
    assert searched.ok and "checkpoints" in searched.content
    assert unknown.error_code == "unknown_tool"


class _FlakyInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    value: str = "x"


@pytest.mark.asyncio
async def test_failed_side_effect_is_not_cached_and_retry_reexecutes():
    attempts = 0

    def handler(arguments: BaseModel) -> ToolResult:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return ToolResult(
                ok=False, content="backend busy", error_code="tool_busy", retryable=True
            )
        return ToolResult(ok=True, content="done")

    registry = ToolRegistry(
        [
            ToolDefinition(
                "flaky.write",
                "Fails once, then succeeds.",
                _FlakyInput,
                ToolSafetySpec(
                    name="flaky.write", side_effect=SideEffect.WRITE, risk=RiskLevel.MEDIUM
                ),
                handler,
            )
        ]
    )
    run_id = uuid4()
    call = ToolCall(name="flaky.write", arguments={"value": "x"})

    failed = await registry.execute(call, run_id=run_id, idempotency_key="stable")
    retried = await registry.execute(call, run_id=run_id, idempotency_key="stable")
    replayed = await registry.execute(call, run_id=run_id, idempotency_key="stable")

    assert failed.error_code == "tool_busy" and failed.retryable
    assert retried.ok and retried.receipt
    # Only the successful result is cached; the retry re-executed the handler.
    assert replayed == retried
    assert attempts == 2
