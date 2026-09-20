from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from admires_harness.safety import (
    ProcessLimits,
    ProcessSandbox,
    SandboxViolation,
    WorkspaceSandbox,
)

pytestmark = pytest.mark.security

PYTHON = Path(sys.executable).name

# The child records its own pid and the pid of a detached grandchild, then sleeps.
# The grandchild outlives any direct kill of the child unless the whole tree dies.
_GRANDCHILD_SCRIPT = (
    "import time\n"
    "from pathlib import Path\n"
    "Path('grandchild.pid').write_text(str(__import__('os').getpid()))\n"
    "time.sleep(30)\n"
)
_CHILD_SCRIPT = (
    "import os, subprocess, sys, time\n"
    "from pathlib import Path\n"
    "Path('child.pid').write_text(str(os.getpid()))\n"
    f"grandchild = subprocess.Popen([sys.executable, '-c', {_GRANDCHILD_SCRIPT!r}])\n"
    "Path('grandchild.pid').write_text(str(grandchild.pid))\n"
    "time.sleep(30)\n"
)


def _sandbox(tmp_path: Path, limits: ProcessLimits | None = None) -> ProcessSandbox:
    workspace = WorkspaceSandbox(tmp_path / "workspace")
    return ProcessSandbox(workspace, allowed_commands=[PYTHON], limits=limits)


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong(0)
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


async def _wait_pid_dead(pid: int, deadline_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + deadline_seconds
    while _pid_alive(pid):
        if time.monotonic() > deadline:
            raise AssertionError(f"process {pid} is still alive")
        await asyncio.sleep(0.05)


async def _wait_for_files(
    root: Path, names: tuple[str, ...], deadline_seconds: float = 10.0
) -> None:
    deadline = time.monotonic() + deadline_seconds
    while not all((root / name).exists() for name in names):
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {names}")
        await asyncio.sleep(0.05)


def _leaked_tasks() -> list[asyncio.Task[object]]:
    return [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and not task.done()
    ]


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
    process = ProcessSandbox(workspace, allowed_commands=[PYTHON])

    result = await process.run([PYTHON, "-c", "import os; print(os.getcwd())"], cwd=".")

    assert result.returncode == 0
    assert result.stdout.strip() == str(workspace.root)
    assert not result.timed_out
    assert not result.output_truncated
    with pytest.raises(SandboxViolation):
        await process.run([PYTHON, "-c", "print(1)"], cwd="..")


@pytest.mark.asyncio
async def test_normal_exit_has_unambiguous_result(tmp_path: Path) -> None:
    process = _sandbox(tmp_path)

    result = await process.run([PYTHON, "-c", "print('ok')"])

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"
    assert result.stderr == ""
    assert not result.timed_out
    assert not result.output_truncated
    assert not result.stdout_truncated
    assert not result.stderr_truncated
    assert _leaked_tasks() == []


@pytest.mark.asyncio
async def test_nonzero_exit_is_distinct_from_timeout(tmp_path: Path) -> None:
    process = _sandbox(tmp_path)

    result = await process.run([PYTHON, "-c", "import sys; sys.exit(3)"])

    assert result.returncode == 3
    assert not result.timed_out
    assert not result.output_truncated
    assert _leaked_tasks() == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal exit semantics")
@pytest.mark.asyncio
async def test_signal_termination_has_negative_returncode(tmp_path: Path) -> None:
    process = _sandbox(tmp_path)

    result = await process.run(
        [PYTHON, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"]
    )

    assert result.returncode == -signal.SIGTERM
    assert not result.timed_out
    assert not result.output_truncated
    assert _leaked_tasks() == []


@pytest.mark.asyncio
async def test_child_environment_is_minimal(tmp_path: Path) -> None:
    process = _sandbox(tmp_path)

    result = await process.run(
        [PYTHON, "-c", "import json, os, sys; json.dump(sorted(os.environ), sys.stdout)"]
    )

    assert result.returncode == 0
    assert set(json.loads(result.stdout)) <= {"PATH", "SYSTEMROOT"}


@pytest.mark.asyncio
async def test_timeout_kills_process_and_marks_result(tmp_path: Path) -> None:
    process = _sandbox(tmp_path, limits=ProcessLimits(timeout_seconds=0.5))

    result = await process.run([PYTHON, "-c", "import time; time.sleep(30)"])

    assert result.timed_out
    assert result.returncode != 0
    assert not result.output_truncated
    assert _leaked_tasks() == []


@pytest.mark.asyncio
async def test_timeout_kills_entire_process_tree(tmp_path: Path) -> None:
    process = _sandbox(tmp_path, limits=ProcessLimits(timeout_seconds=3.0))

    result = await process.run([PYTHON, "-c", _CHILD_SCRIPT])

    assert result.timed_out
    root = process.workspace.root
    child_pid = int((root / "child.pid").read_text())
    grandchild_pid = int((root / "grandchild.pid").read_text())
    await _wait_pid_dead(child_pid)
    await _wait_pid_dead(grandchild_pid)
    assert _leaked_tasks() == []


@pytest.mark.asyncio
async def test_cancellation_kills_tree_and_reaps_tasks(tmp_path: Path) -> None:
    process = _sandbox(tmp_path, limits=ProcessLimits(timeout_seconds=30.0))
    run_task = asyncio.create_task(process.run([PYTHON, "-c", _CHILD_SCRIPT]))
    root = process.workspace.root
    await _wait_for_files(root, ("child.pid", "grandchild.pid"))

    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    await _wait_pid_dead(int((root / "child.pid").read_text()))
    await _wait_pid_dead(int((root / "grandchild.pid").read_text()))
    assert _leaked_tasks() == []


@pytest.mark.asyncio
async def test_output_limit_is_strict(tmp_path: Path) -> None:
    process = _sandbox(tmp_path, limits=ProcessLimits(max_output_bytes=4))

    exact = await process.run([PYTHON, "-c", "import sys; sys.stdout.write('abcd')"])
    assert exact.stdout == "abcd"
    assert not exact.output_truncated

    over = await process.run([PYTHON, "-c", "import sys; sys.stdout.write('abcde')"])
    assert over.stdout == "abcd"
    assert over.output_truncated
    assert over.stdout_truncated
    assert not over.stderr_truncated
    assert not over.timed_out


@pytest.mark.asyncio
async def test_stdout_and_stderr_limits_apply_independently(tmp_path: Path) -> None:
    process = _sandbox(tmp_path, limits=ProcessLimits(max_output_bytes=10))
    script = "import sys;sys.stdout.write('x' * 100);sys.stderr.write('y' * 100)"

    result = await process.run([PYTHON, "-c", script])

    assert result.returncode == 0
    assert result.stdout == "x" * 10
    assert result.stderr == "y" * 10
    assert result.stdout_truncated
    assert result.stderr_truncated
    assert result.output_truncated
    assert not result.timed_out


@pytest.mark.asyncio
async def test_utf8_truncation_decodes_with_replacement(tmp_path: Path) -> None:
    process = _sandbox(tmp_path, limits=ProcessLimits(max_output_bytes=2))
    script = "import sys; sys.stdout.buffer.write('h\xe9llo'.encode('utf-8'))"

    result = await process.run([PYTHON, "-c", script])

    assert result.stdout == "h\ufffd"
    assert result.stdout_truncated
    assert result.output_truncated


@pytest.mark.asyncio
async def test_invalid_working_directory_is_rejected(tmp_path: Path) -> None:
    process = _sandbox(tmp_path)
    (process.workspace.root / "file.txt").write_text("not a directory")

    with pytest.raises(FileNotFoundError):
        await process.run([PYTHON, "--version"], cwd="missing")
    with pytest.raises(NotADirectoryError):
        await process.run([PYTHON, "--version"], cwd="file.txt")
    with pytest.raises(SandboxViolation):
        await process.run([PYTHON, "--version"], cwd="..")
    with pytest.raises(SandboxViolation):
        await process.run([PYTHON, "--version"], cwd=str(process.workspace.root))
    assert _leaked_tasks() == []


@pytest.mark.asyncio
async def test_repeated_runs_converge(tmp_path: Path) -> None:
    process = _sandbox(tmp_path)

    for _ in range(3):
        result = await process.run([PYTHON, "-c", "print('ok')"])
        assert result.returncode == 0
        assert result.stdout.strip() == "ok"
        assert not result.timed_out
        assert not result.output_truncated
    assert _leaked_tasks() == []


@pytest.mark.asyncio
async def test_argument_limits_are_enforced(tmp_path: Path) -> None:
    process = _sandbox(
        tmp_path,
        limits=ProcessLimits(max_arguments=1, max_argument_characters=10),
    )
    with pytest.raises(SandboxViolation) as error:
        await process.run([PYTHON, "--version"])
    assert error.value.code == "argument_limit"
