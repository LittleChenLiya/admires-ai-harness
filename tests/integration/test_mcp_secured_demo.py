"""Offline tests for the policy-wrapped MCP stdio example.

The demo runs against a controlled local fixture server (no network, no
credentials, no real MCP server), so these tests are deterministic.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "mcp_secured_stdio"

sys.path.insert(0, str(EXAMPLE_DIR))

import demo  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def report() -> demo.DemoReport:
    return asyncio.run(demo.run_demo())


def scenario(report: demo.DemoReport, name: str) -> demo.Scenario:
    matches = [entry for entry in report.scenarios if entry.name == name]
    assert len(matches) == 1, f"missing scenario: {name}"
    return matches[0]


def test_all_scenarios_match_expectations(report: demo.DemoReport) -> None:
    failed = [entry.name for entry in report.scenarios if not entry.matched]
    assert not failed, f"scenarios did not match expectations: {failed}"


def test_allowlist_hides_non_allowlisted_server_tools(report: demo.DemoReport) -> None:
    entry = scenario(report, "allowlist")
    assert entry.matched
    assert "demo__admin_wipe" not in entry.detail
    assert "demo__echo" in entry.detail


def test_tools_are_namespaced_and_callable(report: demo.DemoReport) -> None:
    entry = scenario(report, "namespace_echo")
    assert entry.matched
    assert "content=echo: hello" in entry.detail


def test_policy_denies_unregistered_tool(report: demo.DemoReport) -> None:
    entry = scenario(report, "policy_denied_unregistered")
    assert entry.matched
    assert "error=policy_denied" in entry.detail


def test_policy_denies_disabled_network_capability(report: demo.DemoReport) -> None:
    entry = scenario(report, "capability_denied")
    assert entry.matched
    assert "error=policy_denied" in entry.detail


def test_write_requires_one_shot_approval(report: demo.DemoReport) -> None:
    assert scenario(report, "approval_required").matched
    assert scenario(report, "approved_write").matched
    replay = scenario(report, "approval_replay")
    assert replay.matched
    assert "error=approval_already_consumed" in replay.detail


def test_unknown_tool_is_normalized(report: demo.DemoReport) -> None:
    entry = scenario(report, "unknown_tool")
    assert entry.matched
    assert "error=mcp_tool_not_found" in entry.detail


def test_timeout_is_normalized_and_retryable(report: demo.DemoReport) -> None:
    entry = scenario(report, "timeout")
    assert entry.matched
    assert "error=mcp_timeout" in entry.detail


def test_tool_error_is_normalized(report: demo.DemoReport) -> None:
    entry = scenario(report, "tool_error")
    assert entry.matched
    assert "error=mcp_tool_error" in entry.detail


def test_malformed_result_is_normalized(report: demo.DemoReport) -> None:
    entry = scenario(report, "malformed")
    assert entry.matched
    assert "ok=True" in entry.detail


def test_demo_script_smoke() -> None:
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE_DIR / "demo.py")],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "DEMO RESULT: 11/11 scenarios matched expectations" in completed.stdout
