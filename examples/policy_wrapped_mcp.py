"""Connect one allowlisted stdio MCP tool through the harness safety policy.

The server command and arguments are explicit; no network or credentials are
required by this pattern. Replace them only with a reviewed local stdio server.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import UUID

from admires_harness.adapters.mcp import MCPToolExecutor, stdio_mcp_executor
from admires_harness.contracts import JsonValue, SideEffect, ToolCall, ToolResult
from admires_harness.safety import (
    ApprovalGate,
    SafetyPolicy,
    SecuredToolExecutor,
    ToolSafetySpec,
)


def secure_readonly_tool(executor: MCPToolExecutor, exposed_name: str) -> SecuredToolExecutor:
    """Wrap one namespaced MCP tool in an explicit read-only policy."""

    policy = SafetyPolicy([ToolSafetySpec(name=exposed_name, side_effect=SideEffect.READ)])
    return SecuredToolExecutor(executor, policy=policy, approvals=ApprovalGate())


async def call_allowlisted_stdio_tool(
    *,
    command: str,
    args: Sequence[str],
    remote_tool: str,
    arguments: Mapping[str, JsonValue],
    run_id: UUID,
) -> ToolResult:
    """Start a reviewed stdio server and call only its allowlisted tool."""

    exposed_name = f"local_demo__{remote_tool}"
    async with stdio_mcp_executor(
        server_name="local_demo",
        command=command,
        args=args,
        env={},
        allow_tools=frozenset({remote_tool}),
    ) as executor:
        secured = secure_readonly_tool(executor, exposed_name)
        return await secured.execute(
            ToolCall(name=exposed_name, arguments=dict(arguments)),
            run_id=run_id,
            idempotency_key="example-call",
        )
