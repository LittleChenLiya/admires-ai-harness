from __future__ import annotations

import sys
from pathlib import Path

import pytest

from admires_harness.safety import (
    ProcessLimits,
    ProcessSandbox,
    SandboxViolation,
    WorkspaceSandbox,
)

pytestmark = pytest.mark.security


@pytest.mark.asyncio
async def test_process_requires_allowlisted_bare_executable(tmp_path: Path) -> None:
    workspace = WorkspaceSandbox(tmp_path / "workspace")
    process = ProcessSandbox(workspace)
    with pytest.raises(SandboxViolation) as error:
        await process.run([sys.executable, "-c", "print('unsafe')"])
    assert error.value.code == "command_denied"
    with pytest.raises(ValueError, match="bare executable"):
        ProcessSandbox(workspace, allowed_commands=[sys.executable])


@pytest.mark.asyncio
async def test_process_runs_without_shell_and_confines_cwd(tmp_path: Path) -> None:
    workspace = WorkspaceSandbox(tmp_path / "workspace")
    command = Path(sys.executable).name
    process = ProcessSandbox(workspace, allowed_commands=[command])

    result = await process.run([command, "-c", "import os; print(os.getcwd())"], cwd=".")

    assert result.returncode == 0
    assert result.stdout.strip() == str(workspace.root)
    assert not result.timed_out
    with pytest.raises(SandboxViolation):
        await process.run([command, "-c", "print(1)"], cwd="..")


@pytest.mark.asyncio
async def test_process_enforces_timeout_and_output_limit(tmp_path: Path) -> None:
    workspace = WorkspaceSandbox(tmp_path / "workspace")
    command = Path(sys.executable).name
    output_limited = ProcessSandbox(
        workspace, allowed_commands=[command], limits=ProcessLimits(max_output_bytes=3)
    )
    output = await output_limited.run([command, "-c", "print('abcdef', end='')"])
    assert output.stdout == "abc"
    assert output.output_truncated

    timed = ProcessSandbox(
        workspace,
        allowed_commands=[command],
        limits=ProcessLimits(timeout_seconds=0.05),
    )
    timeout = await timed.run([command, "-c", "import time; time.sleep(2)"])
    assert timeout.timed_out
    assert timeout.returncode != 0

    argument_limited = ProcessSandbox(
        workspace,
        allowed_commands=[command],
        limits=ProcessLimits(max_arguments=1, max_argument_characters=10),
    )
    with pytest.raises(SandboxViolation) as error:
        await argument_limited.run([command, "--version"])
    assert error.value.code == "argument_limit"
