from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

import pytest

from admires_harness.adapters.mcp import MCPToolExecutor
from admires_harness.contracts import ApprovalDecision, RiskLevel, SideEffect, ToolCall
from admires_harness.safety import (
    ApprovalGate,
    SafetyPolicy,
    SecuredToolExecutor,
    ToolSafetySpec,
)

RUN_ID = UUID("00000000-0000-0000-0000-000000000007")
ALLOWED_REMOTE_TOOLS = frozenset({"echo", "save_note", "slow", "broken"})

REMOTE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "echo",
        "description": "Echo a message.",
        "inputSchema": {"type": "object", "properties": {"message": {"type": "string"}}},
    },
    {"name": "save_note", "description": "Simulated write.", "inputSchema": {"type": "object"}},
    {"name": "slow", "inputSchema": {"type": "object"}},
    {"name": "broken", "inputSchema": {"type": "object"}},
    {"name": "hidden", "inputSchema": {"type": "object"}},
]


class TransportParseError(RuntimeError):
    """Stands in for a framing/parse failure raised by a real MCP transport."""


class FakeClient:
    def __init__(self) -> None:
        self.called: list[str] = []

    async def list_tools(self) -> object:
        return {"tools": REMOTE_TOOLS}

    async def call_tool(self, name: str, arguments: object = None) -> Any:
        self.called.append(name)
        if name == "slow":
            await asyncio.sleep(5)
        if name == "broken":
            raise TransportParseError("malformed frame")
        return {
            "content": [{"type": "text", "text": f"{name} ok"}],
            "structuredContent": {},
            "isError": False,
        }


async def _make_secured(client: FakeClient) -> tuple[SecuredToolExecutor, ApprovalGate]:
    executor = await MCPToolExecutor.connect(
        client,
        server_name="demo",
        allow_tools=ALLOWED_REMOTE_TOOLS,
        timeout_seconds=0.2,
    )
    policy = SafetyPolicy(
        specs=(
            ToolSafetySpec(name="demo__echo", side_effect=SideEffect.READ),
            ToolSafetySpec(
                name="demo__save_note", side_effect=SideEffect.WRITE, risk=RiskLevel.MEDIUM
            ),
            ToolSafetySpec(name="demo__slow", side_effect=SideEffect.READ),
            ToolSafetySpec(name="demo__broken", side_effect=SideEffect.READ),
            # Allowed by policy but never exposed by the server allowlist.
            ToolSafetySpec(name="demo__ghost", side_effect=SideEffect.READ),
        )
    )
    approvals = ApprovalGate()
    return SecuredToolExecutor(executor, policy=policy, approvals=approvals), approvals


async def test_allowlist_filters_and_namespaces_discovered_tools() -> None:
    client = FakeClient()
    executor = await MCPToolExecutor.connect(
        client, server_name="demo", allow_tools=ALLOWED_REMOTE_TOOLS
    )

    names = [schema["function"]["name"] for schema in executor.schemas()]

    assert names == ["demo__echo", "demo__save_note", "demo__slow", "demo__broken"]
    assert all(name.startswith("demo__") for name in names)
    assert "hidden" not in names and "demo__hidden" not in names


async def test_namespaced_call_maps_back_to_remote_tool_name() -> None:
    client = FakeClient()
    secured, _ = await _make_secured(client)

    result = await secured.execute(
        ToolCall(name="demo__echo", arguments={"message": "hi"}),
        run_id=RUN_ID,
        idempotency_key="echo-1",
    )

    assert result.ok
    assert client.called == ["echo"]


@pytest.mark.security
async def test_policy_denies_tool_without_registered_spec() -> None:
    client = FakeClient()
    secured, _ = await _make_secured(client)

    result = await secured.execute(
        ToolCall(name="demo__hidden", arguments={}),
        run_id=RUN_ID,
        idempotency_key="hidden-1",
    )

    assert not result.ok
    assert result.error_code == "policy_denied"
    assert client.called == []  # denial happens before any transport call


@pytest.mark.security
async def test_write_requires_and_consumes_one_shot_approval() -> None:
    client = FakeClient()
    secured, approvals = await _make_secured(client)
    call = ToolCall(name="demo__save_note", arguments={"text": "remember"})

    blocked = await secured.execute(call, run_id=RUN_ID, idempotency_key="save-1")
    assert blocked.error_code == "approval_required"
    assert client.called == []

    request = await approvals.create(
        run_id=RUN_ID,
        checkpoint_revision=0,
        tool_name="demo__save_note",
        arguments={"text": "remember"},
        reason="write requires approval",
        risk=RiskLevel.MEDIUM,
    )
    await approvals.resolve(
        ApprovalDecision(request_id=request.request_id, approved=True, decided_by="reviewer")
    )
    approved = await secured.execute(
        call, run_id=RUN_ID, idempotency_key="save-1", approval_request_id=request.request_id
    )
    assert approved.ok
    assert client.called == ["save_note"]

    replayed = await secured.execute(
        call, run_id=RUN_ID, idempotency_key="save-2", approval_request_id=request.request_id
    )
    assert replayed.error_code == "approval_already_consumed"
    assert client.called == ["save_note"]  # the replay never reached the server


@pytest.mark.security
async def test_reviewer_denial_blocks_execution() -> None:
    client = FakeClient()
    secured, approvals = await _make_secured(client)
    request = await approvals.create(
        run_id=RUN_ID,
        checkpoint_revision=0,
        tool_name="demo__save_note",
        arguments={"text": "no"},
        reason="write requires approval",
        risk=RiskLevel.MEDIUM,
    )
    await approvals.resolve(
        ApprovalDecision(request_id=request.request_id, approved=False, decided_by="reviewer")
    )

    result = await secured.execute(
        ToolCall(name="demo__save_note", arguments={"text": "no"}),
        run_id=RUN_ID,
        idempotency_key="save-3",
        approval_request_id=request.request_id,
    )

    assert result.error_code == "approval_denied"
    assert client.called == []


async def test_unknown_tool_failure_is_normalized() -> None:
    client = FakeClient()
    secured, _ = await _make_secured(client)

    result = await secured.execute(
        ToolCall(name="demo__ghost", arguments={}),
        run_id=RUN_ID,
        idempotency_key="ghost-1",
    )

    assert result.error_code == "mcp_tool_not_found"
    assert not result.retryable


async def test_timeout_failure_is_normalized_and_retryable() -> None:
    client = FakeClient()
    secured, _ = await _make_secured(client)

    result = await secured.execute(
        ToolCall(name="demo__slow", arguments={}),
        run_id=RUN_ID,
        idempotency_key="slow-1",
    )

    assert result.error_code == "mcp_timeout"
    assert result.retryable


async def test_malformed_response_is_normalized_and_retryable() -> None:
    client = FakeClient()
    secured, _ = await _make_secured(client)

    result = await secured.execute(
        ToolCall(name="demo__broken", arguments={}),
        run_id=RUN_ID,
        idempotency_key="broken-1",
    )

    assert result.error_code == "mcp_transport_error"
    assert result.retryable
