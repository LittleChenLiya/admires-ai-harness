from uuid import uuid4

import pytest

from admires_harness.contracts import ToolCall
from admires_harness.safety import WorkspaceSandbox
from admires_harness.tools import workspace_tools


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
    assert escaped.error_code == "path_traversal"
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
