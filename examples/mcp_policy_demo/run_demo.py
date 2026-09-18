"""Executable offline example: a policy-wrapped MCP stdio tool integration.

Runs the production ``stdio_mcp_executor`` against a controlled local fixture
server (``fixture_server.py``) and wraps it in ``SecuredToolExecutor`` exactly
like a production deployment. No network, credentials, or external MCP server
are involved. Output is a deterministic transcript; run it with::

    uv run python examples/mcp_policy_demo/run_demo.py

Requires the optional MCP extra: ``uv sync --extra mcp``.
See ``docs/mcp-policy-example.md`` for the full walkthrough.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import UUID

from admires_harness.adapters.mcp import MCPToolExecutor, stdio_mcp_executor
from admires_harness.contracts import (
    ApprovalDecision,
    JsonValue,
    RiskLevel,
    SideEffect,
    ToolCall,
    ToolResult,
)
from admires_harness.safety import (
    ApprovalGate,
    SafetyPolicy,
    SecuredToolExecutor,
    ToolSafetySpec,
)

SERVER_NAME = "fixture"
ALLOWED_TOOLS = frozenset({"weather", "write_report", "slow"})
TIMEOUT_SECONDS = 3.0
FIXTURE_SERVER = Path(__file__).with_name("fixture_server.py")
# Fixed run id keeps receipts and the transcript byte-for-byte deterministic.
RUN_ID = UUID("11111111-2222-3333-4444-555555555555")
WRITE_ARGUMENTS: dict[str, JsonValue] = {"title": "launch-notes"}


def build_policy() -> SafetyPolicy:
    """Safety metadata for every exposed name, keyed by the namespaced tool name.

    ``fixture__retired`` is registered in policy but not exposed by the server,
    which lets the demo reach the adapter's unknown-tool normalization through
    the secured boundary. ``fixture__hidden_admin`` has no metadata on purpose:
    default-deny must reject it even if the discovery allowlist is bypassed.
    """

    return SafetyPolicy(
        [
            ToolSafetySpec(name="fixture__weather", side_effect=SideEffect.READ),
            ToolSafetySpec(
                name="fixture__write_report",
                side_effect=SideEffect.WRITE,
                risk=RiskLevel.MEDIUM,
            ),
            ToolSafetySpec(name="fixture__slow", side_effect=SideEffect.READ),
            ToolSafetySpec(name="fixture__retired", side_effect=SideEffect.READ),
        ]
    )


def _fixture_env() -> dict[str, str]:
    """Minimal explicit environment for the server process (no inherited secrets)."""

    env = {"FIXTURE_MODE": "offline-demo"}
    if sys.platform == "win32":  # the Windows loader needs SYSTEMROOT to start Python
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
    return env


def _line(case: str, result: ToolResult, *, extra: str = "") -> str:
    receipt = result.receipt or "-"
    suffix = f" {extra}" if extra else ""
    return (
        f"{case}: ok={result.ok} error_code={result.error_code or '-'} "
        f"retryable={result.retryable} content={result.content!r} receipt={receipt}{suffix}"
    )


async def _approved_write(secured: SecuredToolExecutor, approvals: ApprovalGate) -> ToolResult:
    request = await approvals.create(
        run_id=RUN_ID,
        checkpoint_revision=0,
        tool_name="fixture__write_report",
        arguments=WRITE_ARGUMENTS,
        reason="demo write requires one-shot human approval",
        risk=RiskLevel.MEDIUM,
    )
    await approvals.resolve(
        ApprovalDecision(request_id=request.request_id, approved=True, decided_by="demo-operator")
    )
    return await secured.execute(
        ToolCall(name="fixture__write_report", arguments=WRITE_ARGUMENTS),
        run_id=RUN_ID,
        idempotency_key="write-approved",
        approval_request_id=request.request_id,
    )


async def _denied_write(secured: SecuredToolExecutor, approvals: ApprovalGate) -> ToolResult:
    request = await approvals.create(
        run_id=RUN_ID,
        checkpoint_revision=0,
        tool_name="fixture__write_report",
        arguments=WRITE_ARGUMENTS,
        reason="demo write the reviewer rejects",
        risk=RiskLevel.MEDIUM,
    )
    await approvals.resolve(
        ApprovalDecision(request_id=request.request_id, approved=False, decided_by="demo-operator")
    )
    return await secured.execute(
        ToolCall(name="fixture__write_report", arguments=WRITE_ARGUMENTS),
        run_id=RUN_ID,
        idempotency_key="write-denied",
        approval_request_id=request.request_id,
    )


async def stdio_section() -> list[str]:
    """Drive the real stdio transport against the local fixture server."""

    lines = ["== stdio section: production stdio_mcp_executor against the local fixture =="]
    async with stdio_mcp_executor(
        server_name=SERVER_NAME,
        command=sys.executable,
        args=[str(FIXTURE_SERVER)],
        env=_fixture_env(),
        timeout_seconds=TIMEOUT_SECONDS,
        allow_tools=ALLOWED_TOOLS,
    ) as executor:
        approvals = ApprovalGate()
        secured = SecuredToolExecutor(executor, policy=build_policy(), approvals=approvals)
        discovered = ", ".join(str(schema["function"]["name"]) for schema in secured.schemas())
        lines.append(f"discovered: {discovered}")
        lines.append(
            _line(
                "deny-unregistered",
                await secured.execute(
                    ToolCall(name="fixture__hidden_admin"),
                    run_id=RUN_ID,
                    idempotency_key="hidden-admin",
                ),
            )
        )
        lines.append(
            _line(
                "read-allowed",
                await secured.execute(
                    ToolCall(name="fixture__weather", arguments={"city": "Hangzhou"}),
                    run_id=RUN_ID,
                    idempotency_key="weather-1",
                ),
            )
        )
        lines.append(
            _line(
                "unknown-tool",
                await secured.execute(
                    ToolCall(name="fixture__retired"),
                    run_id=RUN_ID,
                    idempotency_key="retired-1",
                ),
            )
        )
        lines.append(
            _line(
                "approval-required",
                await secured.execute(
                    ToolCall(name="fixture__write_report", arguments=WRITE_ARGUMENTS),
                    run_id=RUN_ID,
                    idempotency_key="write-no-approval",
                ),
            )
        )
        lines.append(_line("approval-denied", await _denied_write(secured, approvals)))
        lines.append(_line("approval-approved", await _approved_write(secured, approvals)))
        lines.append(
            _line(
                "timeout",
                await secured.execute(
                    ToolCall(name="fixture__slow"),
                    run_id=RUN_ID,
                    idempotency_key="slow-1",
                ),
            )
        )
    return lines


class _ScriptedClient:
    """In-process ``MCPClient`` fake for responses a real server should never send."""

    def __init__(self, *, tools: Any, result: Any = None, error: Exception | None = None) -> None:
        self._tools = tools
        self._result = result
        self._error = error

    async def list_tools(self) -> Any:
        return self._tools

    async def call_tool(self, name: str, arguments: Mapping[str, JsonValue] | None = None) -> Any:
        if self._error is not None:
            raise self._error
        return self._result


async def protocol_section() -> list[str]:
    """Normalize malformed and failing responses through the same public adapter."""

    lines = ["== protocol section: malformed and failing responses normalized =="]
    try:
        await MCPToolExecutor.connect(
            _ScriptedClient(tools={"tools": "not-a-list"}), server_name="bad"
        )
    except ValueError as exc:
        lines.append(f"malformed-discovery: error=mcp_invalid_response detail={exc!r}")

    malformed = await MCPToolExecutor.connect(
        _ScriptedClient(
            tools=[{"name": "weather", "inputSchema": {"type": "object"}}],
            result={"content": "not-a-block-list", "structuredContent": 42},
        ),
        server_name="bad",
    )
    weird = await malformed.execute(
        ToolCall(name="bad__weather"), run_id=RUN_ID, idempotency_key="malformed-1"
    )
    lines.append(_line("malformed-call", weird, extra=f"data={weird.data!r}"))

    failing = await MCPToolExecutor.connect(
        _ScriptedClient(
            tools=[{"name": "weather", "inputSchema": {"type": "object"}}],
            error=ConnectionError("secret-internal-detail"),
        ),
        server_name="bad",
    )
    lines.append(
        _line(
            "transport-error",
            await failing.execute(
                ToolCall(name="bad__weather"), run_id=RUN_ID, idempotency_key="transport-1"
            ),
        )
    )

    server_error = await MCPToolExecutor.connect(
        _ScriptedClient(
            tools=[{"name": "weather", "inputSchema": {"type": "object"}}],
            result={"content": [{"type": "text", "text": "fixture says no"}], "isError": True},
        ),
        server_name="bad",
    )
    lines.append(
        _line(
            "tool-error",
            await server_error.execute(
                ToolCall(name="bad__weather"), run_id=RUN_ID, idempotency_key="tool-error-1"
            ),
        )
    )
    return lines


async def transcript() -> list[str]:
    return await stdio_section() + await protocol_section()


def main() -> int:
    if importlib.util.find_spec("mcp") is None:
        print("the mcp extra is not installed; run: uv sync --extra mcp", file=sys.stderr)
        return 2
    for line in asyncio.run(transcript()):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
