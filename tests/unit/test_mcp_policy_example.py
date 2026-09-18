"""Focused tests for the policy-wrapped MCP example in examples/mcp_policy_demo."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from admires_harness.adapters.mcp import MCPToolExecutor
from admires_harness.contracts import ApprovalDecision, RiskLevel, SideEffect, ToolCall
from admires_harness.safety import ApprovalGate, SafetyPolicy, SecuredToolExecutor, ToolSafetySpec

DEMO_DIR = Path(__file__).resolve().parents[2] / "examples" / "mcp_policy_demo"
DEMO_SCRIPT = DEMO_DIR / "run_demo.py"
RUN_ID = UUID("00000000-0000-0000-0000-0000000000d3")

spec = importlib.util.spec_from_file_location("mcp_policy_demo", DEMO_SCRIPT)
assert spec is not None and spec.loader is not None
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class FakeClient:
    """In-process MCPClient double; no SDK or subprocess required."""

    def __init__(self, *, tools: Any = None, result: Any = None, delay: float = 0.0) -> None:
        self._tools = (
            tools
            if tools is not None
            else [
                {"name": "weather", "inputSchema": {"type": "object"}},
                {"name": "write_report", "inputSchema": {"type": "object"}},
                {"name": "hidden_admin", "inputSchema": {"type": "object"}},
            ]
        )
        self._result = result if result is not None else {"content": [{"text": "ok"}]}
        self._delay = delay
        self.calls: list[str] = []

    async def list_tools(self) -> Any:
        return {"tools": self._tools}

    async def call_tool(self, name: str, arguments: Any = None) -> Any:
        self.calls.append(name)
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._result


def make_secured(executor: MCPToolExecutor) -> tuple[SecuredToolExecutor, ApprovalGate]:
    approvals = ApprovalGate()
    policy = SafetyPolicy(
        [
            ToolSafetySpec(name="demo__weather", side_effect=SideEffect.READ),
            ToolSafetySpec(
                name="demo__write_report", side_effect=SideEffect.WRITE, risk=RiskLevel.MEDIUM
            ),
            ToolSafetySpec(name="demo__retired", side_effect=SideEffect.READ),
        ]
    )
    return SecuredToolExecutor(executor, policy=policy, approvals=approvals), approvals


async def connect(client: FakeClient, **kwargs: Any) -> MCPToolExecutor:
    return await MCPToolExecutor.connect(client, server_name="demo", **kwargs)


async def test_discovery_allowlists_and_namespaces_tools() -> None:
    executor = await connect(FakeClient(), allow_tools=frozenset({"weather", "write_report"}))

    names = [schema["function"]["name"] for schema in executor.schemas()]
    assert names == ["demo__weather", "demo__write_report"]
    assert all(schema["x-mcp-server"] == "demo" for schema in executor.schemas())
    assert "demo__hidden_admin" not in executor._remote_names


async def test_policy_denies_tool_without_registered_metadata() -> None:
    executor = await connect(FakeClient())
    secured, _ = make_secured(executor)

    result = await secured.execute(
        ToolCall(name="demo__hidden_admin"), run_id=RUN_ID, idempotency_key="deny"
    )

    assert not result.ok
    assert result.error_code == "policy_denied"
    assert "security metadata" in result.content


async def test_write_requires_one_shot_approval_and_rejects_replay() -> None:
    executor = await connect(FakeClient())
    secured, approvals = make_secured(executor)
    call = ToolCall(name="demo__write_report", arguments={"title": "notes"})

    missing = await secured.execute(call, run_id=RUN_ID, idempotency_key="w0")
    assert missing.error_code == "approval_required"

    request = await approvals.create(
        run_id=RUN_ID,
        checkpoint_revision=0,
        tool_name="demo__write_report",
        arguments={"title": "notes"},
        reason="demo",
        risk=RiskLevel.MEDIUM,
    )
    await approvals.resolve(
        ApprovalDecision(request_id=request.request_id, approved=True, decided_by="tester")
    )
    approved = await secured.execute(
        call, run_id=RUN_ID, idempotency_key="w1", approval_request_id=request.request_id
    )
    assert approved.ok

    replayed = await secured.execute(
        call, run_id=RUN_ID, idempotency_key="w2", approval_request_id=request.request_id
    )
    assert replayed.error_code == "approval_already_consumed"


async def test_denied_approval_maps_to_approval_denied() -> None:
    executor = await connect(FakeClient())
    secured, approvals = make_secured(executor)
    request = await approvals.create(
        run_id=RUN_ID,
        checkpoint_revision=0,
        tool_name="demo__write_report",
        arguments={"title": "notes"},
        reason="demo",
        risk=RiskLevel.MEDIUM,
    )
    await approvals.resolve(
        ApprovalDecision(request_id=request.request_id, approved=False, decided_by="tester")
    )

    result = await secured.execute(
        ToolCall(name="demo__write_report", arguments={"title": "notes"}),
        run_id=RUN_ID,
        idempotency_key="w3",
        approval_request_id=request.request_id,
    )

    assert result.error_code == "approval_denied"


async def test_unknown_tool_passing_policy_maps_to_mcp_tool_not_found() -> None:
    executor = await connect(FakeClient())
    secured, _ = make_secured(executor)

    result = await secured.execute(
        ToolCall(name="demo__retired"), run_id=RUN_ID, idempotency_key="retired"
    )

    assert result.error_code == "mcp_tool_not_found"


async def test_timeout_maps_to_retryable_mcp_timeout() -> None:
    executor = await connect(FakeClient(delay=0.05), timeout_seconds=0.01)
    secured, _ = make_secured(executor)

    result = await secured.execute(
        ToolCall(name="demo__weather"), run_id=RUN_ID, idempotency_key="slow"
    )

    assert result.error_code == "mcp_timeout"
    assert result.retryable


async def test_malformed_discovery_and_call_responses_are_normalized() -> None:
    with pytest.raises(ValueError, match="invalid response"):
        await connect(FakeClient(tools="not-a-list"))

    executor = await connect(
        FakeClient(result={"content": "not-a-block-list", "structuredContent": 42})
    )
    result = await executor.execute(
        ToolCall(name="demo__weather"), run_id=RUN_ID, idempotency_key="malformed"
    )
    assert result.ok
    assert result.content == ""
    assert result.data == {"result": 42}


async def test_transport_error_does_not_leak_internals() -> None:
    client = FakeClient()

    async def fail(name: str, arguments: Any = None) -> Any:
        raise ConnectionError("secret-internal-token")

    client.call_tool = fail  # type: ignore[method-assign]
    executor = await connect(client)

    result = await executor.execute(
        ToolCall(name="demo__weather"), run_id=RUN_ID, idempotency_key="transport"
    )

    assert result.error_code == "mcp_transport_error"
    assert "secret-internal-token" not in result.content


@pytest.mark.integration
async def test_stdio_section_drives_fixture_server() -> None:
    pytest.importorskip("mcp")

    lines = await demo.stdio_section()

    assert lines[1] == "discovered: fixture__weather, fixture__write_report, fixture__slow"
    assert not any("hidden_admin: ok=True" in line for line in lines)
    assert any(
        line.startswith("deny-unregistered: ok=False error_code=policy_denied") for line in lines
    )
    assert any(line.startswith("read-allowed: ok=True") for line in lines)
    assert any(
        line.startswith("unknown-tool: ok=False error_code=mcp_tool_not_found") for line in lines
    )
    assert any(
        line.startswith("approval-required: ok=False error_code=approval_required")
        for line in lines
    )
    assert any(
        line.startswith("approval-denied: ok=False error_code=approval_denied") for line in lines
    )
    assert any(line.startswith("approval-approved: ok=True") for line in lines)
    assert any(
        line.startswith("timeout: ok=False error_code=mcp_timeout retryable=True") for line in lines
    )


@pytest.mark.integration
def test_example_runs_offline_with_deterministic_output() -> None:
    pytest.importorskip("mcp")
    env = {
        **os.environ,
        # Prove the example needs no network: force all proxies to a dead address.
        "HTTP_PROXY": "http://127.0.0.1:9",
        "HTTPS_PROXY": "http://127.0.0.1:9",
        "NO_PROXY": "",
    }

    runs = [
        subprocess.run(
            [sys.executable, str(DEMO_SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        for _ in range(2)
    ]

    for completed in runs:
        assert completed.returncode == 0, completed.stderr
    assert runs[0].stdout == runs[1].stdout
    assert "discovered: fixture__weather, fixture__write_report, fixture__slow" in runs[0].stdout
    assert "error_code=policy_denied" in runs[0].stdout
    assert "error_code=approval_required" in runs[0].stdout
    assert "error_code=mcp_tool_not_found" in runs[0].stdout
    assert "error_code=mcp_timeout" in runs[0].stdout
    assert "error=mcp_invalid_response" in runs[0].stdout
    assert "error_code=mcp_transport_error" in runs[0].stdout
