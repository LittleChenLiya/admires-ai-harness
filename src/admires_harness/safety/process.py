"""A deliberately narrow subprocess capability for opted-in coding agents."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from pydantic import Field

from admires_harness.contracts import StrictModel
from admires_harness.safety.workspace import SandboxViolation, WorkspaceSandbox

# Grace period for stdout/stderr reader tasks after the child has exited. A
# surviving grandchild can hold the pipes open; readers are cancelled after
# this window so run() always converges.
_READER_GRACE_SECONDS = 1.0


class ProcessResult(StrictModel):
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    output_truncated: bool = False


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

    Timeouts and caller cancellation terminate the whole process tree (the process
    group on POSIX, taskkill /T on Windows) and always reap the child process and the
    output reader tasks, so no handles, zombies, or background asyncio tasks leak.
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
            start_new_session=sys.platform != "win32",
        )
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_collector = _OutputCollector(self.limits.max_output_bytes)
        stderr_collector = _OutputCollector(self.limits.max_output_bytes)
        readers = [
            asyncio.create_task(stdout_collector.drain(process.stdout)),
            asyncio.create_task(stderr_collector.drain(process.stderr)),
        ]
        timed_out = False
        try:
            try:
                await asyncio.wait_for(process.wait(), timeout=self.limits.timeout_seconds)
            except TimeoutError:
                timed_out = True
                await self._terminate_tree(process)
            except asyncio.CancelledError:
                await self._terminate_tree(process)
                raise
        finally:
            if process.returncode is None:
                await self._terminate_tree(process)
            await self._finish_readers(readers)
        return ProcessResult(
            argv=tuple(argv),
            returncode=process.returncode if process.returncode is not None else -1,
            stdout=stdout_collector.text(),
            stderr=stderr_collector.text(),
            timed_out=timed_out,
            output_truncated=stdout_collector.truncated or stderr_collector.truncated,
        )

    def _register_command(self, name: str) -> None:
        has_separator = any(separator in name for separator in ("/", "\\"))
        if not name or Path(name).name != name or has_separator:
            raise ValueError("allowed commands must be bare executable names")
        resolved = shutil.which(name)
        if resolved is None:
            raise ValueError(f"allowlisted executable was not found: {name}")
        self._commands[name.casefold()] = str(Path(resolved).resolve(strict=True))

    async def _terminate_tree(self, process: asyncio.subprocess.Process) -> None:
        """Kill the whole process tree rooted at the child and reap the child."""
        if process.returncode is not None:
            return
        if sys.platform == "win32":
            await self._taskkill_tree(process)
        else:
            # The child was started in its own session, so its descendants share
            # its process group and can be signalled together.
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGKILL)
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
        await process.wait()

    @staticmethod
    async def _taskkill_tree(process: asyncio.subprocess.Process) -> None:
        system_root = os.environ.get("SYSTEMROOT", r"C:\Windows")
        taskkill = str(Path(system_root) / "System32" / "taskkill.exe")
        killer = await asyncio.create_subprocess_exec(
            taskkill,
            "/PID",
            str(process.pid),
            "/T",
            "/F",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await killer.wait()

    @staticmethod
    async def _finish_readers(readers: list[asyncio.Task[None]]) -> None:
        """Wait for output readers, cancelling any stuck behind a live grandchild."""
        _, pending = await asyncio.wait(readers, timeout=_READER_GRACE_SECONDS)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


class _OutputCollector:
    """Drain one stream, retaining at most limit bytes and flagging overflow."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._chunks: list[bytes] = []
        self._kept = 0
        self.truncated = False

    async def drain(self, stream: asyncio.StreamReader) -> None:
        while chunk := await stream.read(65_536):
            remaining = self._limit - self._kept
            if remaining > 0:
                retained = chunk[:remaining]
                self._chunks.append(retained)
                self._kept += len(retained)
            if len(chunk) > remaining:
                self.truncated = True

    def text(self) -> str:
        return b"".join(self._chunks).decode("utf-8", errors="replace")
