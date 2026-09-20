from uuid import uuid4

import pytest
from pydantic import BaseModel

from admires_harness.contracts import SideEffect, ToolCall, ToolResult
from admires_harness.safety import ToolSafetySpec, WorkspaceSandbox
from admires_harness.tools import workspace_tools
from admires_harness.tools.registry import ToolDefinition, ToolRegistry


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


class _EmptyInput(BaseModel):
    pass


@pytest.mark.asyncio
async def test_retryable_failure_is_not_cached_and_can_be_retried():
    attempts = 0

    def flaky(arguments: BaseModel) -> ToolResult:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return ToolResult(ok=False, content="busy", error_code="busy", retryable=True)
        return ToolResult(ok=True, content="done")

    registry = ToolRegistry(
        [
            ToolDefinition(
                "flaky.write",
                "Fails once, then succeeds.",
                _EmptyInput,
                ToolSafetySpec(name="flaky.write", side_effect=SideEffect.WRITE),
                flaky,
            )
        ]
    )
    run_id = uuid4()
    call = ToolCall(name="flaky.write", arguments={})

    first = await registry.execute(call, run_id=run_id, idempotency_key="stable")
    retried = await registry.execute(call, run_id=run_id, idempotency_key="stable")
    replayed = await registry.execute(call, run_id=run_id, idempotency_key="stable")

    assert first.error_code == "busy" and first.retryable
    # A failed side effect earns no receipt, so the same key re-runs the handler.
    assert retried.ok and retried.receipt
    assert replayed == retried
    assert attempts == 2
