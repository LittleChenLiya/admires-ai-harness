"""Deterministic ProcessSandbox lifecycle tests (no network, no shell)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from admires_harness.safety import (
    ProcessLimits,
    ProcessSandbox,
    WorkspaceSandbox,
)

pytestmark = pytest.mark.security

COMMAND = Path(sys.executable).name
MARKER = "grandchild.marker"

# The child spawns a grandchild that writes a marker after a delay, then sleeps.
# If tree termination works, the grandchild dies before the marker appears.
SPAWN_GRANDCHILD = (
    "import subprocess, sys, time;"
    "subprocess.Popen([sys.executable, '-c',"
    " 'import time, pathlib; time.sleep(2.0);"
    f' pathlib.Path("{MARKER}").write_text("x")\']);'
    "time.sleep(30)"
)


def make_sandbox(
    workspace: WorkspaceSandbox, *, limits: ProcessLimits | None = None
) -> ProcessSandbox:
    return ProcessSandbox(workspace, allowed_commands=[COMMAND], limits=limits)


def other_pending_tasks() -> set[asyncio.Task[object]]:
    current = asyncio.current_task()
    return {task for task in asyncio.all_tasks() if task is not current and not task.done()}


async def test_normal_and_nonzero_exit_are_distinguishable(tmp_path: Path) -> None:
    workspace = WorkspaceSandbox(tmp_path / "workspace")
    sandbox = make_sandbox(workspace)

    ok = await sandbox.run([COMMAND, "-c", "print('hello')"])
    assert ok.returncode == 0
    assert ok.stdout.strip() == "hello"
    assert not ok.timed_out
    assert not ok.output_truncated

    failed = await sandbox.run([COMMAND, "-c", "raise SystemExit(3)"])
    assert failed.returncode == 3
    assert not failed.timed_out
    assert not failed.output_truncated


async def test_timeout_kills_whole_process_tree(tmp_path: Path) -> None:
    workspace = WorkspaceSandbox(tmp_path / "workspace")
    sandbox = make_sandbox(workspace, limits=ProcessLimits(timeout_seconds=0.5))
    before = other_pending_tasks()

    result = await sandbox.run([COMMAND, "-c", SPAWN_GRANDCHILD])

    assert result.timed_out
    assert result.returncode != 0
    assert not result.output_truncated
    assert other_pending_tasks() <= before
    await asyncio.sleep(3.0)
    assert not (workspace.root / MARKER).exists()


async def test_caller_cancellation_kills_whole_process_tree(tmp_path: Path) -> None:
    workspace = WorkspaceSandbox(tmp_path / "workspace")
    sandbox = make_sandbox(workspace, limits=ProcessLimits(timeout_seconds=30.0))
    before = other_pending_tasks()

    run_task = asyncio.create_task(sandbox.run([COMMAND, "-c", SPAWN_GRANDCHILD]))
    await asyncio.sleep(1.0)
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    assert other_pending_tasks() <= before
    await asyncio.sleep(3.0)
    assert not (workspace.root / MARKER).exists()


async def test_detached_grandchild_does_not_block_run(tmp_path: Path) -> None:
    workspace = WorkspaceSandbox(tmp_path / "workspace")
    sandbox = make_sandbox(workspace, limits=ProcessLimits(timeout_seconds=30.0))
    script = (
        "import subprocess, sys;"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3)'])"
    )

    result = await sandbox.run([COMMAND, "-c", script])

    assert result.returncode == 0
    assert not result.timed_out


async def test_stdout_and_stderr_limits_are_strict(tmp_path: Path) -> None:
    workspace = WorkspaceSandbox(tmp_path / "workspace")
    sandbox = make_sandbox(workspace, limits=ProcessLimits(max_output_bytes=64))
    script = "import sys; sys.stdout.write('a' * 10000); sys.stderr.write('b' * 10000)"

    result = await sandbox.run([COMMAND, "-c", script])

    assert result.returncode == 0
    assert not result.timed_out
    assert result.output_truncated
    assert result.stdout == "a" * 64
    assert result.stderr == "b" * 64


async def test_utf8_truncation_replaces_split_character(tmp_path: Path) -> None:
    workspace = WorkspaceSandbox(tmp_path / "workspace")
    sandbox = make_sandbox(workspace, limits=ProcessLimits(max_output_bytes=5))

    result = await sandbox.run(
        [COMMAND, "-c", "import sys; sys.stdout.buffer.write('é'.encode() * 10)"]
    )

    assert result.returncode == 0
    assert result.output_truncated
    assert result.stdout == "\u00e9\u00e9\ufffd"


async def test_invalid_working_directory(tmp_path: Path) -> None:
    workspace = WorkspaceSandbox(tmp_path / "workspace")
    sandbox = make_sandbox(workspace)
    workspace.write_text("file.txt", "x")

    with pytest.raises(FileNotFoundError):
        await sandbox.run([COMMAND, "--version"], cwd="missing-directory")
    with pytest.raises(NotADirectoryError):
        await sandbox.run([COMMAND, "--version"], cwd="file.txt")


async def test_repeated_runs_converge_without_leaking_tasks(tmp_path: Path) -> None:
    workspace = WorkspaceSandbox(tmp_path / "workspace")
    sandbox = make_sandbox(workspace)
    before = other_pending_tasks()

    for _ in range(5):
        result = await sandbox.run([COMMAND, "-c", "print('again')"])
        assert result.returncode == 0
        assert result.stdout.strip() == "again"

    assert other_pending_tasks() <= before
