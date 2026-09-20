"""Offline walkthrough: policy-wrapped MCP tools over a local fake stdio server.

Runs deterministically with no network, no credentials, and no real MCP server.
It uses the production MCPToolExecutor adapter, SafetyPolicy, ApprovalGate, and
SecuredToolExecutor exactly as a real deployment would wire them. See
docs/mcp-policy-demo.md for the full guide.

Usage: python examples/mcp_policy_demo/demo.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from uuid import UUID

from admires_harness.adapters.mcp import MCPToolExecutor
from admires_harness.contracts import ApprovalDecision, RiskLevel, SideEffect, ToolCall
from admires_harness.safety import (
    ApprovalGate,
    SafetyPolicy,
    SecuredToolExecutor,
    ToolSafetySpec,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stdio_client import StdioJsonRpcClient

RUN_ID = UUID("00000000-0000-0000-0000-000000000001")
ALLOWED_REMOTE_TOOLS = frozenset({"echo", "save_note", "slow", "broken"})
SERVER_PATH = Path(__file__).resolve().with_name("fake_server.py")


def _report(label: str, *, ok: bool, error_code: str | None, content: str) -> None:
    status = "ok" if ok else f"error:{error_code}"
    print(f"[{label}] {status} {content}".rstrip())


async def main() -> int:
    client = await StdioJsonRpcClient.start(sys.executable, [str(SERVER_PATH)])
    try:
        executor = await MCPToolExecutor.connect(
            client,
            server_name="demo",
            allow_tools=ALLOWED_REMOTE_TOOLS,
            timeout_seconds=0.5,
        )
        exposed = sorted(schema["function"]["name"] for schema in executor.schemas())
        print(f"[discover] namespaced tools: {exposed}")
        print("[allowlist] server tool hidden is not exposed")

        policy = SafetyPolicy(
            specs=(
                ToolSafetySpec(name="demo__echo", side_effect=SideEffect.READ),
                ToolSafetySpec(
                    name="demo__save_note", side_effect=SideEffect.WRITE, risk=RiskLevel.MEDIUM
                ),
                ToolSafetySpec(name="demo__slow", side_effect=SideEffect.READ),
                ToolSafetySpec(name="demo__broken", side_effect=SideEffect.READ),
                # Registered in policy but never exposed by the server allowlist,
                # so the adapter must normalize the unknown-tool failure.
                ToolSafetySpec(name="demo__ghost", side_effect=SideEffect.READ),
            )
        )
        approvals = ApprovalGate()
        secured = SecuredToolExecutor(executor, policy=policy, approvals=approvals)

        async def call(
            label: str,
            name: str,
            arguments: dict,
            *,
            key: str,
            approval_request_id=None,
        ) -> None:
            result = await secured.execute(
                ToolCall(name=name, arguments=arguments),
                run_id=RUN_ID,
                idempotency_key=key,
                approval_request_id=approval_request_id,
            )
            _report(label, ok=result.ok, error_code=result.error_code, content=result.content)

        await call("ok", "demo__echo", {"message": "hello offline mcp"}, key="echo-1")
        await call("policy-denied", "demo__hidden", {}, key="hidden-1")
        await call("approval-required", "demo__save_note", {"text": "remember this"}, key="s-1")

        note_arguments = {"text": "remember this"}
        request = await approvals.create(
            run_id=RUN_ID,
            checkpoint_revision=0,
            tool_name="demo__save_note",
            arguments=note_arguments,
            reason="demo write requires one-shot human approval",
            risk=RiskLevel.MEDIUM,
        )
        await approvals.resolve(
            ApprovalDecision(
                request_id=request.request_id, approved=True, decided_by="operator-demo"
            )
        )
        await call(
            "approved",
            "demo__save_note",
            note_arguments,
            key="s-1",
            approval_request_id=request.request_id,
        )

        rejected = await approvals.create(
            run_id=RUN_ID,
            checkpoint_revision=0,
            tool_name="demo__save_note",
            arguments={"text": "different note"},
            reason="reviewer rejects this write",
            risk=RiskLevel.MEDIUM,
        )
        await approvals.resolve(
            ApprovalDecision(
                request_id=rejected.request_id, approved=False, decided_by="operator-demo"
            )
        )
        await call(
            "approval-denied",
            "demo__save_note",
            {"text": "different note"},
            key="s-2",
            approval_request_id=rejected.request_id,
        )

        await call("unknown", "demo__ghost", {}, key="ghost-1")
        await call("timeout", "demo__slow", {}, key="slow-1")
        # Malformed transport frame; must run last because it breaks the stream.
        await call("malformed", "demo__broken", {}, key="broken-1")
    finally:
        await client.close()
    print("[done] offline MCP policy demo completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
