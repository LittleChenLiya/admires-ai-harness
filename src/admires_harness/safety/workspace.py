"""Filesystem confinement for built-in workspace tools."""

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


class WorkspaceSandbox:
    """Resolve and operate on paths confined beneath one workspace root.

    Absolute paths, parent traversal, Windows alternate streams/device names, and all
    symlinks/junctions are rejected. Rejecting links also prevents a workspace link from
    turning an apparently safe relative path into host filesystem access.
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
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(self.root):
            raise SandboxViolation("outside_workspace", "path escapes the workspace")
        if must_exist and not resolved.exists():
            raise FileNotFoundError(raw)
        return resolved

    def read_bytes(self, relative_path: str | Path) -> bytes:
        path = self.resolve(relative_path, must_exist=True)
        self._require_regular_file(path)
        size = path.stat().st_size
        if size > self.limits.max_read_bytes:
            raise SandboxViolation("read_limit", "file exceeds the workspace read limit")
        with path.open("rb") as stream:
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
        if path.exists():
            self._require_regular_file(path)
        mode = "wb" if overwrite else "xb"
        with path.open(mode) as stream:
            stream.write(content)
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
                if self._is_link_or_junction(child) or not child.is_file():
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
            if current.exists():
                if self._is_link_or_junction(current) or not current.is_dir():
                    raise SandboxViolation("unsafe_parent", "parent is not a real directory")
            else:
                current.mkdir()

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
    def _require_regular_file(path: Path) -> None:
        file_status = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(file_status.st_mode):
            raise SandboxViolation("not_regular_file", "only regular files are allowed")
        if file_status.st_nlink != 1:
            raise SandboxViolation("hard_link_denied", "files with multiple hard links are denied")
