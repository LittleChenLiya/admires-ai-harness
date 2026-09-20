"""Filesystem confinement for built-in workspace tools."""

from __future__ import annotations

import os
import stat
from contextlib import suppress
from pathlib import Path, PureWindowsPath

from pydantic import Field

from admires_harness.contracts import StrictModel

_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_BINARY = getattr(os, "O_BINARY", 0)


class WorkspaceLimits(StrictModel):
    max_read_bytes: int = Field(default=1_000_000, ge=1)
    max_write_bytes: int = Field(default=1_000_000, ge=1)
    max_list_entries: int = Field(default=1_000, ge=1)


class SandboxViolation(PermissionError):
    """Base workspace confinement failure carrying a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class SandboxNotFoundError(SandboxViolation, FileNotFoundError):
    """The requested workspace entry does not exist."""

    def __init__(self, message: str) -> None:
        super().__init__("not_found", message)


class SandboxExistsError(SandboxViolation, FileExistsError):
    """An exclusive create lost the race to an existing entry."""

    def __init__(self, message: str) -> None:
        super().__init__("already_exists", message)


class SandboxNotDirectoryError(SandboxViolation, NotADirectoryError):
    """A directory operation targeted a non-directory entry."""

    def __init__(self, message: str) -> None:
        super().__init__("not_directory", message)


class WorkspaceSandbox:
    """Resolve and operate on paths confined beneath one workspace root.

    Absolute or drive-rooted paths, parent traversal, Windows alternate
    streams/device names, and all symlinks/junctions are rejected. Reads and
    overwrite writes re-verify the opened file descriptor against the checked
    directory entry so a concurrent replacement cannot swap the target after
    validation. After every open, the descriptor identity is re-checked against
    a freshly re-walked link-free path so a parent directory swapped mid-call
    cannot redirect the operation outside the root. Exclusive creates use
    O_EXCL so concurrent writers fail atomically instead of truncating each
    other, and failed creates remove any partial file they produced.
    """

    def __init__(self, root: Path, *, limits: WorkspaceLimits | None = None) -> None:
        root.mkdir(parents=True, exist_ok=True)
        if self._is_link_or_junction(root):
            raise SandboxViolation("unsafe_root", "workspace root cannot be a link")
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("workspace root must be a directory")
        self.limits = limits or WorkspaceLimits()

    def resolve(self, relative_path: str | Path, *, must_exist: bool = False) -> Path:
        raw = str(relative_path)
        if not raw or "\x00" in raw:
            raise SandboxViolation("invalid_path", "path must be non-empty and contain no NUL")
        supplied = Path(relative_path)
        windows_path = PureWindowsPath(raw)
        if supplied.is_absolute() or windows_path.is_absolute() or windows_path.root:
            raise SandboxViolation("absolute_path", "absolute paths are not allowed")
        if windows_path.drive or any(":" in part for part in windows_path.parts):
            raise SandboxViolation(
                "alternate_stream", "drive paths and alternate streams are denied"
            )
        if any(PureWindowsPath(part).is_reserved() for part in windows_path.parts):
            raise SandboxViolation("reserved_path", "reserved device paths are denied")
        ambiguous_windows_part = any(
            part.endswith((" ", ".")) and part not in {".", ".."} for part in windows_path.parts
        )
        if ambiguous_windows_part:
            raise SandboxViolation("ambiguous_path", "path components cannot end in a space or dot")
        if any(part == ".." for part in supplied.parts):
            raise SandboxViolation("path_traversal", "parent path traversal is denied")

        candidate = self.root.joinpath(supplied)
        self._reject_links(candidate)
        resolved = candidate.resolve(strict=False)
        resolved = self._strip_extended_prefix(resolved)
        if not resolved.is_relative_to(self.root):
            raise SandboxViolation("outside_workspace", "path escapes the workspace")
        if must_exist and not os.path.lexists(resolved):
            raise SandboxNotFoundError(raw)
        return resolved

    def read_bytes(self, relative_path: str | Path) -> bytes:
        path = self.resolve(relative_path, must_exist=True)
        before = self._stat_for_read(path, relative_path)
        try:
            descriptor = os.open(path, os.O_RDONLY | _O_NOFOLLOW | _O_BINARY)
        except FileNotFoundError as error:
            raise SandboxNotFoundError(str(relative_path)) from error
        except PermissionError as error:
            raise SandboxViolation("permission_denied", "read permission denied") from error
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            self._verify_same_entry(before, opened)
            self._require_regular_stat(opened)
            self._verify_open_within_root(path, opened)
            if opened.st_size > self.limits.max_read_bytes:
                raise SandboxViolation("read_limit", "file exceeds the workspace read limit")
            content = stream.read(self.limits.max_read_bytes + 1)
        if len(content) > self.limits.max_read_bytes:
            raise SandboxViolation("read_limit", "file changed beyond the read limit")
        return content

    def read_text(self, relative_path: str | Path, *, encoding: str = "utf-8") -> str:
        return self.read_bytes(relative_path).decode(encoding)

    def write_bytes(
        self, relative_path: str | Path, content: bytes, *, overwrite: bool = False
    ) -> Path:
        if len(content) > self.limits.max_write_bytes:
            raise SandboxViolation("write_limit", "content exceeds the workspace write limit")
        path = self.resolve(relative_path)
        self._make_safe_parents(path.parent)
        # Recheck after directory creation to narrow the link-swap window.
        path = self.resolve(relative_path)
        if overwrite and os.path.lexists(path):
            self._overwrite_existing(path, content)
        else:
            self._create_exclusive(path, content)
        return path

    def write_text(
        self,
        relative_path: str | Path,
        content: str,
        *,
        encoding: str = "utf-8",
        overwrite: bool = False,
    ) -> Path:
        return self.write_bytes(relative_path, content.encode(encoding), overwrite=overwrite)

    def list_files(self, relative_path: str | Path = ".") -> tuple[str, ...]:
        directory = self.resolve(relative_path, must_exist=True)
        if self._is_link_or_junction(directory):
            raise SandboxViolation("link_denied", "links and junctions are denied")
        if not directory.is_dir():
            raise SandboxNotDirectoryError(str(relative_path))
        files: list[str] = []
        for current_root, directories, names in os.walk(directory, followlinks=False):
            current = Path(current_root)
            safe_directories: list[str] = []
            for name in sorted(directories):
                child = current / name
                if not self._is_link_or_junction(child):
                    safe_directories.append(name)
            directories[:] = safe_directories
            for name in sorted(names):
                child = current / name
                if self._is_link_or_junction(child) or not child.is_file():
                    continue
                files.append(child.relative_to(self.root).as_posix())
                if len(files) > self.limits.max_list_entries:
                    raise SandboxViolation("list_limit", "workspace listing exceeds its limit")
        return tuple(files)

    def _create_exclusive(self, path: Path, content: bytes) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_BINARY
        try:
            descriptor = os.open(path, flags, 0o666)
        except FileExistsError as error:
            raise SandboxExistsError(f"workspace entry already exists: {path.name}") from error
        except FileNotFoundError as error:
            raise SandboxViolation("unsafe_parent", "parent is not a real directory") from error
        except PermissionError as error:
            raise SandboxViolation("permission_denied", "write permission denied") from error
        try:
            with os.fdopen(descriptor, "wb") as stream:
                opened = os.fstat(stream.fileno())
                self._require_regular_stat(opened)
                self._verify_open_within_root(path, opened)
                stream.write(content)
        except Exception:
            self._remove_partial(path)
            raise

    def _overwrite_existing(self, path: Path, content: bytes) -> None:
        before = self._stat_for_read(path, path.name)
        try:
            descriptor = os.open(path, os.O_WRONLY | _O_NOFOLLOW | _O_BINARY)
        except FileNotFoundError as error:
            raise SandboxNotFoundError(path.name) from error
        except PermissionError as error:
            raise SandboxViolation("permission_denied", "write permission denied") from error
        with os.fdopen(descriptor, "wb") as stream:
            opened = os.fstat(stream.fileno())
            self._verify_same_entry(before, opened)
            self._require_regular_stat(opened)
            self._verify_open_within_root(path, opened)
            stream.truncate(0)
            stream.write(content)

    def _stat_for_read(self, path: Path, relative_path: str | Path) -> os.stat_result:
        try:
            return self._require_regular_file(path)
        except SandboxViolation:
            raise
        except FileNotFoundError as error:
            raise SandboxNotFoundError(str(relative_path)) from error
        except PermissionError as error:
            raise SandboxViolation("permission_denied", "stat permission denied") from error

    def _verify_open_within_root(self, path: Path, opened: os.stat_result) -> None:
        """Prove an open descriptor still names a regular file beneath the root.

        The path components are re-walked for links and the final entry is
        re-stat'ed; if its identity matches the open descriptor, the descriptor
        cannot refer to anything outside the workspace, regardless of any
        concurrent swap that happens after this check.
        """
        self._reject_links(path)
        try:
            final = self._require_regular_file(path)
        except SandboxViolation:
            raise
        except FileNotFoundError as error:
            raise SandboxViolation(
                "replaced_file", "workspace entry was replaced during the operation"
            ) from error
        except PermissionError as error:
            raise SandboxViolation("permission_denied", "stat permission denied") from error
        self._verify_same_entry(final, opened)

    @staticmethod
    def _strip_extended_prefix(path: Path) -> Path:
        # Windows resolve() may surface a ? extended-length prefix when the
        # final component is concurrently created; strip it before comparing.
        raw = str(path)
        if raw.startswith("\\\\?\\"):
            return Path(raw[4:])
        return path

    def _make_safe_parents(self, parent: Path) -> None:
        relative = parent.relative_to(self.root)
        current = self.root
        for part in relative.parts:
            current = current / part
            try:
                current.mkdir()
                continue
            except FileExistsError:
                pass
            except FileNotFoundError as error:
                raise SandboxViolation(
                    "unsafe_parent", "parent chain is not a directory"
                ) from error
            except PermissionError as error:
                raise SandboxViolation("permission_denied", "mkdir permission denied") from error
            if self._is_link_or_junction(current) or not current.is_dir():
                raise SandboxViolation("unsafe_parent", "parent is not a real directory")

    def _reject_links(self, candidate: Path) -> None:
        try:
            relative = candidate.relative_to(self.root)
        except ValueError as error:
            raise SandboxViolation("outside_workspace", "path escapes the workspace") from error
        current = self.root
        for part in relative.parts:
            current = current / part
            if os.path.lexists(current) and self._is_link_or_junction(current):
                raise SandboxViolation("link_denied", "links and junctions are denied")

    @staticmethod
    def _is_link_or_junction(path: Path) -> bool:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction is not None and is_junction())

    @staticmethod
    def _require_regular_file(path: Path) -> os.stat_result:
        file_status = path.stat(follow_symlinks=False)
        WorkspaceSandbox._require_regular_stat(file_status)
        return file_status

    @staticmethod
    def _require_regular_stat(file_status: os.stat_result) -> None:
        if not stat.S_ISREG(file_status.st_mode):
            raise SandboxViolation("not_regular_file", "only regular files are allowed")
        if file_status.st_nlink != 1:
            raise SandboxViolation("hard_link_denied", "files with multiple hard links are denied")

    @staticmethod
    def _verify_same_entry(before: os.stat_result, opened: os.stat_result) -> None:
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise SandboxViolation(
                "replaced_file", "workspace entry was replaced during the operation"
            )

    @staticmethod
    def _remove_partial(path: Path) -> None:
        with suppress(OSError):
            path.unlink()
