"""Filesystem confinement for built-in workspace tools.

Error semantics are deliberately split so callers can react without parsing
messages:

- SandboxViolation (a PermissionError subclass) carries a stable "code" for
  every policy denial, including OS permission failures ("permission_denied")
  and paths replaced mid-operation ("path_replaced").
- FileNotFoundError means the requested file does not exist.
- FileExistsError means an exclusive create lost to an existing target.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path, PureWindowsPath

from pydantic import Field

from admires_harness.contracts import StrictModel


class WorkspaceLimits(StrictModel):
    max_read_bytes: int = Field(default=1_000_000, ge=1)
    max_write_bytes: int = Field(default=1_000_000, ge=1)
    max_list_entries: int = Field(default=1_000, ge=1)


class SandboxViolation(PermissionError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


_OPEN_FLAGS = getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _normalize_resolved(path: Path) -> Path:
    # Windows can intermittently return an extended-length prefix from resolve().
    raw = str(path)
    if raw.startswith("\\\\?\\"):
        return Path(raw[4:])
    return path


class WorkspaceSandbox:
    """Resolve and operate on paths confined beneath one workspace root.

    Absolute paths, parent traversal, Windows alternate streams/device names, and all
    symlinks/junctions are rejected. Every component is re-validated after directory
    creation, and opened files are re-identified through their descriptor, so a
    concurrently swapped parent, link, or hard link cannot redirect an operation
    outside the root.
    """

    def __init__(self, root: Path, *, limits: WorkspaceLimits | None = None) -> None:
        root.mkdir(parents=True, exist_ok=True)
        if self._is_link_or_junction(root):
            raise SandboxViolation("unsafe_root", "workspace root cannot be a link")
        self.root = _normalize_resolved(root.resolve(strict=True))
        if not self.root.is_dir():
            raise ValueError("workspace root must be a directory")
        self.limits = limits or WorkspaceLimits()

    def resolve(self, relative_path: str | Path, *, must_exist: bool = False) -> Path:
        raw = str(relative_path)
        if not raw or "\x00" in raw:
            raise SandboxViolation("invalid_path", "path must be non-empty and contain no NUL")
        supplied = Path(relative_path)
        if supplied.is_absolute() or PureWindowsPath(raw).is_absolute():
            raise SandboxViolation("absolute_path", "absolute paths are not allowed")
        windows_path = PureWindowsPath(raw)
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
        resolved = _normalize_resolved(candidate.resolve(strict=False))
        if not resolved.is_relative_to(self.root):
            raise SandboxViolation("outside_workspace", "path escapes the workspace")
        if must_exist and not resolved.exists():
            raise FileNotFoundError(raw)
        return resolved

    def read_bytes(self, relative_path: str | Path) -> bytes:
        path = self.resolve(relative_path, must_exist=True)
        self._require_regular_file(path)
        try:
            descriptor = os.open(path, os.O_RDONLY | _OPEN_FLAGS)
        except PermissionError as error:
            raise SandboxViolation("permission_denied", f"cannot read {path.name!r}") from error
        try:
            opened = self._verify_open_file(descriptor, path)
            if opened.st_size > self.limits.max_read_bytes:
                raise SandboxViolation("read_limit", "file exceeds the workspace read limit")
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = -1  # the stream owns the descriptor now
                content = stream.read(self.limits.max_read_bytes + 1)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
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
        if os.path.lexists(path):
            if not overwrite:
                raise FileExistsError(str(relative_path))
            self._require_regular_file(path)
        # O_TRUNC is applied via truncate only after the descriptor is verified, so a
        # swapped-in link cannot cause an outside file to be truncated on platforms
        # without O_NOFOLLOW.
        flags = os.O_WRONLY | os.O_CREAT | _OPEN_FLAGS
        if not overwrite:
            flags |= os.O_EXCL
        try:
            descriptor = os.open(path, flags, 0o666)
        except FileExistsError:
            raise FileExistsError(str(relative_path)) from None
        except PermissionError as error:
            raise SandboxViolation("permission_denied", f"cannot write {path.name!r}") from error
        try:
            self._verify_open_file(descriptor, path)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1  # the stream owns the descriptor now
                if overwrite:
                    stream.truncate(0)
                stream.write(content)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
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
        if not directory.is_dir():
            raise NotADirectoryError(str(relative_path))
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
                if self._is_link_or_junction(child):
                    continue
                try:
                    child_status = child.stat(follow_symlinks=False)
                except OSError:
                    continue
                if not stat.S_ISREG(child_status.st_mode) or child_status.st_nlink != 1:
                    continue
                files.append(child.relative_to(self.root).as_posix())
                if len(files) > self.limits.max_list_entries:
                    raise SandboxViolation("list_limit", "workspace listing exceeds its limit")
        return tuple(files)

    def _make_safe_parents(self, parent: Path) -> None:
        relative = parent.relative_to(self.root)
        current = self.root
        for part in relative.parts:
            current = current / part
            if os.path.lexists(current):
                self._require_safe_directory(current)
                continue
            try:
                current.mkdir()
            except FileExistsError:
                # A concurrent creator won; verify what actually landed.
                self._require_safe_directory(current)
            except PermissionError as error:
                raise SandboxViolation(
                    "permission_denied", f"cannot create parent {part!r}"
                ) from error

    def _require_safe_directory(self, path: Path) -> None:
        if self._is_link_or_junction(path):
            raise SandboxViolation("link_denied", "links and junctions are denied")
        if not path.is_dir():
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

    def _verify_open_file(self, descriptor: int, path: Path) -> os.stat_result:
        """Confirm the opened descriptor still identifies the in-root path."""
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise SandboxViolation("not_regular_file", "only regular files are allowed")
        if opened.st_nlink != 1:
            raise SandboxViolation("hard_link_denied", "files with multiple hard links are denied")
        latest = path.stat(follow_symlinks=False)
        if (latest.st_dev, latest.st_ino) != (opened.st_dev, opened.st_ino):
            raise SandboxViolation("path_replaced", "path was replaced during the operation")
        return opened

    @staticmethod
    def _is_link_or_junction(path: Path) -> bool:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction is not None and is_junction())

    @staticmethod
    def _require_regular_file(path: Path) -> None:
        file_status = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(file_status.st_mode):
            raise SandboxViolation("not_regular_file", "only regular files are allowed")
        if file_status.st_nlink != 1:
            raise SandboxViolation("hard_link_denied", "files with multiple hard links are denied")
