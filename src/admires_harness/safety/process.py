"""A deliberately narrow subprocess capability for opted-in coding agents."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from pydantic import Field

from admires_harness.contracts import StrictModel
from admires_harness.safety.workspace import SandboxViolation, WorkspaceSandbox

_POSIX = os.name != "nt"
_SIGKILL = getattr(signal, "SIGKILL", 9)  # Only used on POSIX; Windows has no SIGKILL.
_killpg = getattr(os, "killpg", None)  # POSIX only; absent on Windows.


class ProcessResult(StrictModel):
    """Outcome of one sandboxed process.

    returncode is the raw OS status: zero on success, positive for a non-zero exit,
    and negative when a POSIX signal terminated the child. timed_out is set only
    when the sandbox itself killed the process tree at the deadline, so a genuine
    timeout is never confused with a child that merely exited non-zero.
    """

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    output_truncated: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False


class ProcessLimits(StrictModel):
    timeout_seconds: float = Field(default=15.0, gt=0, le=300)
    max_output_bytes: int = Field(default=100_000, ge=1, le=10_000_000)
    max_arguments: int = Field(default=128, ge=1, le=1_024)
    max_argument_characters: int = Field(default=16_384, ge=1, le=1_000_000)


class ProcessSandbox:
    """Execute approved argument vectors without a shell inside a workspace.

    This is defense in depth, not an OS/container security boundary. Executable names
    must be explicitly allowlisted and are resolved once at construction. The child gets
    a minimal environment, bounded runtime/output, no stdin, and a workspace-confined
    current directory. Network isolation must still be supplied by the container/host.

    On timeout or caller cancellation the whole process tree is terminated (the process
    group on POSIX, taskkill /T on Windows) and the stdout/stderr reader tasks are
    always awaited, so no zombie process, pipe handle, or asyncio task leaks.
    """

    def __init__(
        self,
        workspace: WorkspaceSandbox,
        *,
        allowed_commands: Iterable[str] = (),
        limits: ProcessLimits | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.workspace = workspace
        self.limits = limits or ProcessLimits()
        self._commands: dict[str, str] = {}
        for name in allowed_commands:
            self._register_command(name)
        safe_environment = {"PATH": os.environ.get("PATH", "")}
        if os.name == "nt":
            safe_environment["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
        safe_environment.update(environment or {})
        self._environment = safe_environment

    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str | Path = ".",
    ) -> ProcessResult:
        if not argv or not argv[0].strip():
            raise SandboxViolation("invalid_command", "command argv must not be empty")
        if any("\x00" in argument for argument in argv):
            raise SandboxViolation("invalid_command", "command arguments cannot contain NUL")
        if len(argv) > self.limits.max_arguments:
            raise SandboxViolation("argument_limit", "command has too many arguments")
        if sum(len(argument) for argument in argv) > self.limits.max_argument_characters:
            raise SandboxViolation("argument_limit", "command arguments are too large")
        executable = self._commands.get(argv[0].casefold())
        if executable is None:
            raise SandboxViolation("command_denied", "executable is not allowlisted")
        working_directory = self.workspace.resolve(cwd, must_exist=True)
        if not working_directory.is_dir():
            raise NotADirectoryError(str(cwd))

        process = await asyncio.create_subprocess_exec(
            executable,
            *argv[1:],
            cwd=working_directory,
            env=self._environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # A dedicated session lets POSIX kill the whole process group;
            # Windows terminates the tree with taskkill instead.
            start_new_session=_POSIX,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_task = asyncio.create_task(self._collect_output(process.stdout))
        stderr_task = asyncio.create_task(self._collect_output(process.stderr))
        timed_out = False
        try:
            try:
                await asyncio.wait_for(process.wait(), timeout=self.limits.timeout_seconds)
            except TimeoutError:
                timed_out = True
                await self._terminate_process_tree(process)
            stdout, stdout_truncated = await stdout_task
            stderr, stderr_truncated = await stderr_task
        except BaseException:
            # Cancellation and unexpected errors must not leak the process tree,
            # the pipe transports, or the reader tasks.
            await self._terminate_process_tree(process)
            for task in (stdout_task, stderr_task):
                task.cancel()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            raise
        returncode = process.returncode
        if returncode is None:  # pragma: no cover - process.wait() guarantees a status
            raise RuntimeError("process finished without a return code")
        return ProcessResult(
            argv=tuple(argv),
            returncode=returncode,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            timed_out=timed_out,
            output_truncated=stdout_truncated or stderr_truncated,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

    def _register_command(self, name: str) -> None:
        has_separator = any(separator in name for separator in ("/", "\\"))
        if not name or Path(name).name != name or has_separator:
            raise ValueError("allowed commands must be bare executable names")
        resolved = shutil.which(name)
        if resolved is None:
            raise ValueError(f"allowlisted executable was not found: {name}")
        self._commands[name.casefold()] = str(Path(resolved).resolve(strict=True))

    async def _terminate_process_tree(self, process: asyncio.subprocess.Process) -> None:
        """Kill the child and every descendant, then reap the child."""
        if process.returncode is not None:
            return
        if _killpg is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                _killpg(process.pid, _SIGKILL)
        else:
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/F",
                "/T",
                "/PID",
                str(process.pid),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
        await process.wait()

    async def _collect_output(self, stream: asyncio.StreamReader) -> tuple[bytes, bool]:
        chunks: list[bytes] = []
        kept = 0
        truncated = False
        while chunk := await stream.read(65_536):
            remaining = self.limits.max_output_bytes - kept
            if remaining > 0:
                retained = chunk[:remaining]
                chunks.append(retained)
                kept += len(retained)
            if len(chunk) > remaining:
                truncated = True
        return b"".join(chunks), truncated
