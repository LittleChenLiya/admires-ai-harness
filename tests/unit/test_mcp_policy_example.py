"""Policy, approval, allowlist, and namespacing coverage for the MCP example.

These cases drive the same public production contracts the example uses
(``MCPToolExecutor`` wrapped by ``SecuredToolExecutor``) with an in-process fake
client, so failure normalization is deterministic without spawning a subprocess.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from admires_harness.adapters.mcp import MCPToolExecutor
from admires_harness.contracts import ApprovalDecision, RiskLevel, ToolCall
from admires_harness.safety import ApprovalGate, SecuredToolExecutor

_EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "examples" / "mcp_policy" / "run_example.py"


def _load_example() -> Any:
    spec = importlib.util.spec_from_file_location("mcp_policy_run_example", _EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


example = _load_example()


class PolicyFakeClient:
    """In-process MCP client shaped like the offline stdio fixture."""

    def __init__(self) -> None:
        self.forwarded: list[str] = []

    async def list_tools(self) -> object:
        return {
            "tools": [
                {
                    "name": "echo",
                    "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
                },
                {
                    "name": "record",
                    "inputSchema": {"type": "object", "properties": {"note": {"type": "string"}}},
                },
                {"name": "fetch_url", "inputSchema": {"type": "object"}},
                {"name": "internal_only", "inputSchema": {"type": "object"}},
                {"name": "always_fails", "inputSchema": {"type": "object"}},
                {"name": "slow", "inputSchema": {"type": "object"}},
                {"name": "broken", "inputSchema": {"type": "object"}},
            ]
        }

    async def call_tool(self, name: str, arguments: object = None) -> Any:
        self.forwarded.append(name)
        if name == "echo":
            args = arguments if isinstance(arguments, dict) else {}
            return {"content": [{"type": "text", "text": str(args.get("text", ""))}]}
        if name == "record":
            args = arguments if isinstance(arguments, dict) else {}
            return {"content": [{"type": "text", "text": f"recorded: {args.get('note', '')}"}]}
        if name == "always_fails":
            return {"content": [{"type": "text", "text": "refused"}], "isError": True}
        if name == "slow":
            await asyncio.sleep(0.05)
            return {"content": [{"type": "text", "text": "late"}]}
        if name == "broken":
            return {"content": [42]}
        return {"content": [{"type": "text", "text": "unreachable"}]}


@pytest.fixture
async def secured_bundle() -> tuple[SecuredToolExecutor, ApprovalGate]:
    client = PolicyFakeClient()
    inner = await MCPToolExecutor.connect(
        client,
        server_name=example.SERVER_NAME,
        allow_tools=example.ALLOWED_TOOLS,
        timeout_seconds=0.01,
    )
    gate = ApprovalGate()
    return SecuredToolExecutor(inner, policy=example.build_policy(), approvals=gate), gate


def test_allowlist_and_namespacing_apply_at_discovery() -> None:
    async def scenario() -> None:
        client = PolicyFakeClient()
        inner = await MCPToolExecutor.connect(
            client,
            server_name=example.SERVER_NAME,
            allow_tools=example.ALLOWED_TOOLS,
        )
        names = [schema["function"]["name"] for schema in inner.schemas()]

        assert names == [
            "warehouse__echo",
            "warehouse__record",
            "warehouse__fetch_url",
            "warehouse__always_fails",
            "warehouse__slow",
            "warehouse__broken",
        ]
        assert "warehouse__internal_only" not in names
        assert all(name.startswith("warehouse__") for name in names)

    asyncio.run(scenario())


@pytest.mark.asyncio
async def test_read_tool_runs_without_approval(
    secured_bundle: tuple[SecuredToolExecutor, ApprovalGate],
) -> None:
    secured, _gate = secured_bundle
    result = await secured.execute(
        ToolCall(name="warehouse__echo", arguments={"text": "ok"}),
        run_id=example.RUN_ID,
        idempotency_key="echo",
    )
    assert result.ok
    assert result.content == "ok"


@pytest.mark.asyncio
async def test_network_tool_is_policy_denied(
    secured_bundle: tuple[SecuredToolExecutor, ApprovalGate],
) -> None:
    secured, _gate = secured_bundle
    result = await secured.execute(
        ToolCall(name="warehouse__fetch_url", arguments={"url": "https://x.invalid"}),
        run_id=example.RUN_ID,
        idempotency_key="fetch",
    )
    assert result.error_code == "policy_denied"


@pytest.mark.asyncio
async def test_non_allowlisted_and_unknown_tools_fail_closed(
    secured_bundle: tuple[SecuredToolExecutor, ApprovalGate],
) -> None:
    secured, _gate = secured_bundle
    for tool_name in ("warehouse__internal_only", "warehouse__ghost"):
        result = await secured.execute(
            ToolCall(name=tool_name),
            run_id=example.RUN_ID,
            idempotency_key=f"deny-{tool_name}",
        )
        assert result.error_code == "policy_denied", tool_name


@pytest.mark.asyncio
async def test_write_needs_one_shot_bound_approval(
    secured_bundle: tuple[SecuredToolExecutor, ApprovalGate],
) -> None:
    secured, gate = secured_bundle
    arguments = {"note": "hi"}

    blocked = await secured.execute(
        ToolCall(name="warehouse__record", arguments=arguments),
        run_id=example.RUN_ID,
        idempotency_key="record-1",
    )
    assert blocked.error_code == "approval_required"

    request = await gate.create(
        run_id=example.RUN_ID,
        checkpoint_revision=0,
        tool_name="warehouse__record",
        arguments=arguments,
        reason="offline demo approval",
        risk=RiskLevel.MEDIUM,
    )
    await gate.resolve(
        ApprovalDecision(
            request_id=request.request_id,
            approved=True,
            decided_by=example.OPERATOR,
        )
    )

    approved = await secured.execute(
        ToolCall(name="warehouse__record", arguments=arguments),
        run_id=example.RUN_ID,
        idempotency_key="record-2",
        approval_request_id=request.request_id,
    )
    assert approved.ok
    assert approved.content == "recorded: hi"

    replayed = await secured.execute(
        ToolCall(name="warehouse__record", arguments=arguments),
        run_id=example.RUN_ID,
        idempotency_key="record-3",
        approval_request_id=request.request_id,
    )
    assert replayed.error_code == "approval_already_consumed"


@pytest.mark.asyncio
async def test_normalized_remote_timeout_and_malformed_failures(
    secured_bundle: tuple[SecuredToolExecutor, ApprovalGate],
) -> None:
    secured, _gate = secured_bundle
    timed_out = await secured.execute(
        ToolCall(name="warehouse__slow"),
        run_id=example.RUN_ID,
        idempotency_key="slow",
    )
    assert timed_out.error_code == "mcp_timeout"
    assert timed_out.retryable

    broken = await secured.execute(
        ToolCall(name="warehouse__broken"),
        run_id=example.RUN_ID,
        idempotency_key="broken",
    )
    assert broken.error_code == "mcp_malformed_response"
    assert not broken.retryable

    failed = await secured.execute(
        ToolCall(name="warehouse__always_fails"),
        run_id=example.RUN_ID,
        idempotency_key="fails",
    )
    assert failed.error_code == "mcp_tool_error"


@pytest.mark.asyncio
async def test_scrubbed_environment_excludes_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/bin")
    monkeypatch.setenv("MCP_FAKE_DELAY_SECONDS", "2")
    monkeypatch.setenv("API_TOKEN", "super-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "super-secret")

    env = example.scrubbed_env()

    assert env["PATH"] == "/bin"
    assert env["MCP_FAKE_DELAY_SECONDS"] == "2"
    assert "API_TOKEN" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
