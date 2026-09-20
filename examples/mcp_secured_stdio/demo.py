"""Offline, deterministic demo: policy-wrapped MCP tools over stdio.

Runs entirely against a controlled local fixture server (fake_server.py). No network,
no credentials, no real MCP server. Every MCP executor is wrapped in the production
SecuredToolExecutor so policy and one-shot approvals stay on the adapter boundary.

Usage:  uv run python examples/mcp_secured_stdio/demo.py
Guide:  docs/mcp-secured-tools.md
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from stdio_client import StdioMCPClient

from admires_harness.adapters.mcp import MCPToolExecutor
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

SERVER_SCRIPT = Path(__file__).with_name("fake_server.py")
SERVER_NAME = "demo"
ALLOWED_REMOTE_TOOLS = frozenset({"echo", "write_note", "slow", "fail", "malformed"})
RUN_ID = uuid5(NAMESPACE_URL, "https://admires.example/mcp-secured-demo")


def exposed(remote: str) -> str:
    return f"{SERVER_NAME}__{remote}"


def build_policy() -> SafetyPolicy:
    """Register safety metadata for every namespaced tool the demo may call.

    Default-deny stays intact: writes/process/network/critical capabilities are not
    enabled here, so a network-capable spec is denied and writes need approval.
    """

    specs = [
        ToolSafetySpec(name=exposed("echo"), side_effect=SideEffect.READ),
        ToolSafetySpec(
            name=exposed("write_note"), side_effect=SideEffect.WRITE, risk=RiskLevel.MEDIUM
        ),
        ToolSafetySpec(name=exposed("slow"), side_effect=SideEffect.READ),
        ToolSafetySpec(name=exposed("fail"), side_effect=SideEffect.READ),
        ToolSafetySpec(name=exposed("malformed"), side_effect=SideEffect.READ),
        # Registered but never advertised by the server: exercises mcp_tool_not_found.
        ToolSafetySpec(name=exposed("ghost"), side_effect=SideEffect.READ),
        # Registered but capability stays disabled: exercises default-deny.
        ToolSafetySpec(name=exposed("fetch"), side_effect=SideEffect.NETWORK),
    ]
    return SafetyPolicy(specs)


@dataclass
class Scenario:
    name: str
    detail: str
    matched: bool


@dataclass
class DemoReport:
    scenarios: list[Scenario] = field(default_factory=list)

    def record(self, name: str, detail: str, matched: bool) -> None:
        self.scenarios.append(Scenario(name, detail, matched))

    @property
    def matched_count(self) -> int:
        return sum(1 for scenario in self.scenarios if scenario.matched)

    @property
    def ok(self) -> bool:
        return self.matched_count == len(self.scenarios)


def describe(result: ToolResult) -> str:
    error = result.error_code or "-"
    content = result.content.replace("\n", " ")[:60] or "-"
    return f"ok={result.ok} error={error} content={content}"


async def run_demo() -> DemoReport:
    report = DemoReport()
    client = await StdioMCPClient.start(sys.executable, str(SERVER_SCRIPT))
    try:
        executor = await MCPToolExecutor.connect(
            client,
            server_name=SERVER_NAME,
            timeout_seconds=2.0,
            allow_tools=ALLOWED_REMOTE_TOOLS,
        )
        approvals = ApprovalGate()
        secured = SecuredToolExecutor(executor, policy=build_policy(), approvals=approvals)

        async def call(
            name: str,
            arguments: Mapping[str, JsonValue] | None = None,
            *,
            key: str,
            approval_request_id: UUID | None = None,
        ) -> ToolResult:
            return await secured.execute(
                ToolCall(name=name, arguments=dict(arguments or {})),
                run_id=RUN_ID,
                idempotency_key=key,
                approval_request_id=approval_request_id,
            )

        # 1. Allowlist + namespaced discovery: admin_wipe is advertised by the
        # server but filtered out; exposed names carry the server__ prefix.
        names = sorted(str(schema["function"]["name"]) for schema in secured.schemas())
        expected_names = sorted(exposed(tool) for tool in ALLOWED_REMOTE_TOOLS)
        report.record(
            "allowlist",
            f"exposed={','.join(names)}",
            names == expected_names and exposed("admin_wipe") not in names,
        )

        # 2. Namespaced call of a read-only tool.
        result = await call(exposed("echo"), {"text": "hello"}, key="echo")
        report.record(
            "namespace_echo", describe(result), result.ok and result.content == "echo: hello"
        )

        # 3. Default-deny: a non-allowlisted server tool has no safety metadata.
        result = await call(exposed("admin_wipe"), key="admin_wipe")
        report.record(
            "policy_denied_unregistered",
            describe(result),
            result.error_code == "policy_denied",
        )

        # 4. Registered in policy but not exposed by the server: mcp_tool_not_found.
        result = await call(exposed("ghost"), key="ghost")
        report.record("unknown_tool", describe(result), result.error_code == "mcp_tool_not_found")

        # 5. Write tool without an approval: approval_required.
        result = await call(exposed("write_note"), {"text": "hi"}, key="write-1")
        report.record(
            "approval_required", describe(result), result.error_code == "approval_required"
        )

        # 6. One-shot approval bound to run, tool, and exact arguments.
        request = await approvals.create(
            run_id=RUN_ID,
            checkpoint_revision=0,
            tool_name=exposed("write_note"),
            arguments={"text": "hi"},
            reason="demo write",
            risk=RiskLevel.MEDIUM,
        )
        await approvals.resolve(
            ApprovalDecision(request_id=request.request_id, approved=True, decided_by="operator")
        )
        result = await call(
            exposed("write_note"),
            {"text": "hi"},
            key="write-2",
            approval_request_id=request.request_id,
        )
        report.record("approved_write", describe(result), result.ok)

        # 7. Approvals are one-shot: replaying the same request fails closed.
        result = await call(
            exposed("write_note"),
            {"text": "hi"},
            key="write-3",
            approval_request_id=request.request_id,
        )
        report.record(
            "approval_replay",
            describe(result),
            result.error_code == "approval_already_consumed",
        )

        # 8. Network capability stays disabled by default.
        result = await call(exposed("fetch"), {"url": "https://example.invalid"}, key="fetch")
        report.record("capability_denied", describe(result), result.error_code == "policy_denied")

        # 9. Slow server tool hits the adapter timeout: normalized, retryable.
        result = await call(exposed("slow"), key="slow")
        report.record(
            "timeout",
            describe(result),
            result.error_code == "mcp_timeout" and result.retryable,
        )

        # 10. Server-reported tool failure is normalized to mcp_tool_error.
        result = await call(exposed("fail"), key="fail")
        report.record("tool_error", describe(result), result.error_code == "mcp_tool_error")

        # 11. Malformed server result is normalized instead of crashing.
        result = await call(exposed("malformed"), key="malformed")
        report.record(
            "malformed",
            f"ok={result.ok} content={result.content!r} data={result.data}",
            result.ok
            and result.content == ""
            and result.data == {"result": ["not", "a", "mapping"]},
        )
    finally:
        await client.aclose()
    return report


def main() -> int:
    report = asyncio.run(run_demo())
    for scenario in report.scenarios:
        mark = "PASS" if scenario.matched else "FAIL"
        print(f"[{scenario.name}] {mark} {scenario.detail}")
    print(
        f"DEMO RESULT: {report.matched_count}/{len(report.scenarios)}"
        " scenarios matched expectations"
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
