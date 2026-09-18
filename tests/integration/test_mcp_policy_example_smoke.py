"""Offline smoke test: run the policy-wrapped MCP example end to end.

Spawns the same command a user runs. The test is skipped when the optional ``mcp``
extra is absent and never touches the network: the child server is a stdlib fixture.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples" / "mcp_policy" / "run_example.py"

pytest.importorskip("mcp")

_EXPECTED_CODES = (
    "error_code=approval_required",
    "error_code=approval_already_consumed",
    "error_code=policy_denied",
    "error_code=mcp_tool_not_found",
    "error_code=mcp_tool_error",
    "error_code=mcp_timeout",
    "error_code=mcp_malformed_response",
)


def _offline_env() -> Mapping[str, str]:
    # Force offline resolution and strip common credential variables so the test stays
    # honest about its no-secrets guarantee.
    forbidden = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "AWS_")
    env = dict(os.environ)
    env["NO_PROXY"] = "*"
    return {
        key: value for key, value in env.items() if not any(tag in key.upper() for tag in forbidden)
    }


@pytest.mark.integration
def test_mcp_policy_example_runs_offline_and_passes_every_scenario() -> None:
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=REPO_ROOT,
        env=dict(_offline_env()),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (
        "warehouse__internal_only"
        not in completed.stdout.split("discovered tools:")[1].splitlines()[0]
    )
    assert "warehouse__echo" in completed.stdout
    for expected in _EXPECTED_CODES:
        assert expected in completed.stdout, expected
    assert "[FAIL]" not in completed.stdout
