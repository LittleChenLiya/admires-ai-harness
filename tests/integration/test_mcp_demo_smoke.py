"""Offline smoke test for the policy-wrapped MCP stdio example."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "examples" / "mcp_policy_demo" / "demo.py"

EXPECTED_MARKERS = (
    "[discover] namespaced tools:",
    "[allowlist] server tool hidden is not exposed",
    "[ok] ok",
    "error:policy_denied",
    "error:approval_required",
    "[approved] ok",
    "error:approval_denied",
    "error:mcp_tool_not_found",
    "error:mcp_timeout",
    "error:mcp_transport_error",
    "[done] offline MCP policy demo completed",
)


@pytest.mark.integration
def test_mcp_policy_demo_runs_offline_and_covers_all_scenarios() -> None:
    completed = subprocess.run(
        [sys.executable, str(DEMO)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    for marker in EXPECTED_MARKERS:
        assert marker in completed.stdout, marker
