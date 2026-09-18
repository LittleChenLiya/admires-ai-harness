#!/usr/bin/env python3
"""Offline walkthrough: a policy-wrapped MCP tool over a local stdio fixture.

Run it from the repository root::

    uv sync --extra mcp
    uv run python examples/mcp_policy/run_example.py

The script never reaches the network, never reads credentials, and never starts a real
external MCP server: it spawns ``fake_mcp_server.py`` from this directory and talks to it
over stdio with the *real* MCP client SDK through the production
``stdio_mcp_executor`` adapter. The adapter is then wrapped in the production
``SecuredToolExecutor`` with a ``SafetyPolicy`` and a one-shot ``ApprovalGate``.

Every scenario asserts the expected normalized outcome, so the process exits non-zero if
a security boundary regresses.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
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

SERVER_NAME = "warehouse"
FIXTURE_PATH = Path(__file__).with_name("fake_mcp_server.py")

# Only these remote tools are imported during discovery. ``internal_only`` stays on the
# wire but is deliberately absent here.
ALLOWED_TOOLS = frozenset({"echo", "record", "fetch_url", "always_fails", "slow", "broken"})

# Fixed identities make the output deterministic across runs.
RUN_ID = UUID("11111111-1111-4111-8111-111111111111")
OPERATOR = "operator@example.local"

# Environment variables a spawned stdio server is allowed to inherit. Credential-bearing
# variables (API keys, tokens, cloud metadata) are intentionally excluded.
_ENV_ALLOWLIST = (
    "PATH",
    "LANG",
    "LC_ALL",
    "TZ",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "TEMP",
    "TMP",
    "MCP_FAKE_DELAY_SECONDS",
)


@dataclass(frozen=True, slots=True)
class ScenarioOutcome:
    name: str
    expected_error: str | None
    result: ToolResult

    @property
    def passed(self) -> bool:
        if self.expected_error is None:
            return self.result.ok
        return (not self.result.ok) and self.result.error_code == self.expected_error


def scrubbed_env() -> dict[str, str]:
    """Return a minimal environment without credentials for the child process."""

    return {key: value for key, value in os.environ.items() if key.upper() in _ENV_ALLOWLIST}


def build_policy() -> SafetyPolicy:
    """Register safety metadata for every allowlisted, model-facing tool.

    The default policy denies unregistered tools, keeps network capability disabled, and
    requires one-shot human approval for writes. These flags are the deployment's
    decisions, not the MCP server's claims.
    """

    specs = (
        ToolSafetySpec(
            name=f"{SERVER_NAME}__echo",
            side_effect=SideEffect.READ,
            risk=RiskLevel.LOW,
            description="Echo text from the offline fixture.",
        ),
        ToolSafetySpec(
            name=f"{SERVER_NAME}__record",
            side_effect=SideEffect.WRITE,
            risk=RiskLevel.MEDIUM,
            description="Record a note; requires one-shot human approval.",
        ),
        ToolSafetySpec(
            name=f"{SERVER_NAME}__fetch_url",
            side_effect=SideEffect.NETWORK,
            risk=RiskLevel.HIGH,
            description="Fetch a URL; network capability stays disabled in this example.",
        ),
        ToolSafetySpec(
            name=f"{SERVER_NAME}__always_fails",
            side_effect=SideEffect.READ,
            risk=RiskLevel.LOW,
            description="Returns a well-formed remote tool error.",
        ),
        ToolSafetySpec(
            name=f"{SERVER_NAME}__slow",
            side_effect=SideEffect.READ,
            risk=RiskLevel.LOW,
            description="Sleeps past the client timeout.",
        ),
        ToolSafetySpec(
            name=f"{SERVER_NAME}__broken",
            side_effect=SideEffect.READ,
            risk=RiskLevel.LOW,
            description="Returns a malformed MCP content array.",
        ),
    )
    return SafetyPolicy(
        specs,
        allow_write=True,
        allow_process=False,
        allow_network=False,
        allow_critical=False,
    )


@asynccontextmanager
async def build_secured_executor(
    *, gate: ApprovalGate, timeout_seconds: float = 1.0
) -> AsyncIterator[tuple[SecuredToolExecutor, MCPToolExecutor, tuple[str, ...]]]:
    async with stdio_mcp_executor(
        server_name=SERVER_NAME,
        command=sys.executable,
        args=(str(FIXTURE_PATH),),
        env=scrubbed_env(),
        timeout_seconds=timeout_seconds,
        prefix_names=True,
        allow_tools=ALLOWED_TOOLS,
    ) as inner:
        exposed = tuple(schema["function"]["name"] for schema in inner.schemas())
        secured = SecuredToolExecutor(inner, policy=build_policy(), approvals=gate)
        yield secured, inner, exposed


async def _approve_once(
    gate: ApprovalGate, *, tool_name: str, arguments: Mapping[str, JsonValue]
) -> UUID:
    request = await gate.create(
        run_id=RUN_ID,
        checkpoint_revision=0,
        tool_name=tool_name,
        arguments=arguments,
        reason=f"Approve one offline demo write via {tool_name}",
        risk=RiskLevel.MEDIUM,
    )
    await gate.resolve(
        ApprovalDecision(request_id=request.request_id, approved=True, decided_by=OPERATOR)
    )
    return request.request_id


async def run_scenarios() -> tuple[bool, list[str], list[ScenarioOutcome]]:
    outcomes: list[ScenarioOutcome] = []
    gate = ApprovalGate()
    async with build_secured_executor(gate=gate) as (secured, _inner, exposed):
        # 1. Allowlisted discovery with server namespacing.
        discovery = list(exposed)

        await _check(
            outcomes,
            "discovery is allowlisted and namespaced",
            passed=(
                discovery
                == [
                    f"{SERVER_NAME}__echo",
                    f"{SERVER_NAME}__record",
                    f"{SERVER_NAME}__fetch_url",
                    f"{SERVER_NAME}__always_fails",
                    f"{SERVER_NAME}__slow",
                    f"{SERVER_NAME}__broken",
                ]
            ),
        )

        # 2. Allowed read needs no approval.
        outcomes.append(
            ScenarioOutcome(
                "read tool succeeds without approval",
                None,
                await secured.execute(
                    ToolCall(name=f"{SERVER_NAME}__echo", arguments={"text": "offline-ok"}),
                    run_id=RUN_ID,
                    idempotency_key="echo-1",
                ),
            )
        )

        # 3. Write is blocked until a bound, one-shot approval is consumed.
        write_arguments: dict[str, JsonValue] = {"note": "hello"}
        outcomes.append(
            ScenarioOutcome(
                "write requires approval before first call",
                "approval_required",
                await secured.execute(
                    ToolCall(name=f"{SERVER_NAME}__record", arguments=write_arguments),
                    run_id=RUN_ID,
                    idempotency_key="record-1",
                ),
            )
        )
        request_id = await _approve_once(
            gate,
            tool_name=f"{SERVER_NAME}__record",
            arguments=write_arguments,
        )
        approved_call = await secured.execute(
            ToolCall(name=f"{SERVER_NAME}__record", arguments=write_arguments),
            run_id=RUN_ID,
            idempotency_key="record-2",
            checkpoint_revision=0,
            approval_request_id=request_id,
        )
        outcomes.append(ScenarioOutcome("write succeeds after bound approval", None, approved_call))
        replay = await secured.execute(
            ToolCall(name=f"{SERVER_NAME}__record", arguments=write_arguments),
            run_id=RUN_ID,
            idempotency_key="record-3",
            checkpoint_revision=0,
            approval_request_id=request_id,
        )
        outcomes.append(
            ScenarioOutcome(
                "approval cannot be replayed",
                "approval_already_consumed",
                replay,
            )
        )

        # 4. A network-side-effect tool is denied by the default policy.
        outcomes.append(
            ScenarioOutcome(
                "network capability is denied by policy",
                "policy_denied",
                await secured.execute(
                    ToolCall(
                        name=f"{SERVER_NAME}__fetch_url",
                        arguments={"url": "https://example.invalid"},
                    ),
                    run_id=RUN_ID,
                    idempotency_key="fetch-1",
                ),
            )
        )

        # 5. Non-allowlisted and completely unknown tools fail closed at the policy.
        outcomes.append(
            ScenarioOutcome(
                "non-allowlisted server tool is denied",
                "policy_denied",
                await secured.execute(
                    ToolCall(name=f"{SERVER_NAME}__internal_only"),
                    run_id=RUN_ID,
                    idempotency_key="internal-1",
                ),
            )
        )
        outcomes.append(
            ScenarioOutcome(
                "unknown namespaced tool is denied",
                "policy_denied",
                await secured.execute(
                    ToolCall(name=f"{SERVER_NAME}__ghost"),
                    run_id=RUN_ID,
                    idempotency_key="ghost-1",
                ),
            )
        )

        # 6. Adapter-level defense in depth: even without policy, an unexposed name is
        # never forwarded to the untrusted server.
        outcomes.append(
            ScenarioOutcome(
                "adapter refuses an unexposed tool",
                "mcp_tool_not_found",
                await _inner.execute(
                    ToolCall(name=f"{SERVER_NAME}__ghost"),
                    run_id=RUN_ID,
                    idempotency_key="ghost-inner",
                ),
            )
        )

        # 7. Remote isError result is normalized.
        outcomes.append(
            ScenarioOutcome(
                "remote tool error is normalized",
                "mcp_tool_error",
                await secured.execute(
                    ToolCall(name=f"{SERVER_NAME}__always_fails"),
                    run_id=RUN_ID,
                    idempotency_key="fail-1",
                ),
            )
        )

        # 8. Timeout is normalized and marked retryable.
        outcomes.append(
            ScenarioOutcome(
                "timeout is normalized as retryable",
                "mcp_timeout",
                await secured.execute(
                    ToolCall(name=f"{SERVER_NAME}__slow"),
                    run_id=RUN_ID,
                    idempotency_key="slow-1",
                ),
            )
        )

        # 9. Malformed wire response is normalized, not shown as empty success.
        outcomes.append(
            ScenarioOutcome(
                "malformed response is normalized",
                "mcp_malformed_response",
                await secured.execute(
                    ToolCall(name=f"{SERVER_NAME}__broken"),
                    run_id=RUN_ID,
                    idempotency_key="broken-1",
                ),
            )
        )

    all_passed = all(outcome.passed for outcome in outcomes)
    return all_passed, discovery, outcomes


async def _check(outcomes: list[ScenarioOutcome], name: str, *, passed: bool) -> None:
    result = ToolResult(
        ok=passed,
        content="discovery matched the allowlist" if passed else "allowlist mismatch",
        error_code=None if passed else "example_assertion_failed",
    )
    outcomes.append(ScenarioOutcome(name, None if passed else "example_assertion_failed", result))


def render(discovery: list[str], outcomes: list[ScenarioOutcome]) -> str:
    lines = [
        "Policy-wrapped MCP example (offline stdio fixture)",
        f"server: {SERVER_NAME}  discovered tools: {', '.join(discovery)}",
        "",
    ]
    for outcome in outcomes:
        status = "PASS" if outcome.passed else "FAIL"
        code = outcome.result.error_code or "-"
        lines.append(
            f"[{status}] {outcome.name} -> error_code={code} retryable={outcome.result.retryable}"
        )
    return "\n".join(lines)


async def main() -> int:
    passed, discovery, outcomes = await run_scenarios()
    print(render(discovery, outcomes))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
