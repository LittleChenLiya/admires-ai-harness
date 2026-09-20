from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

import pytest

from admires_harness.safety import (
    SandboxExistsError,
    SandboxNotDirectoryError,
    SandboxNotFoundError,
    SandboxViolation,
    WorkspaceLimits,
    WorkspaceSandbox,
)

pytestmark = pytest.mark.security


def test_workspace_allows_bounded_relative_file_operations(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(tmp_path / "workspace")

    sandbox.write_text("reports/result.md", "complete")

    assert sandbox.read_text("reports/result.md") == "complete"
    assert sandbox.list_files() == ("reports/result.md",)
    with pytest.raises(FileExistsError):
        sandbox.write_text("reports/result.md", "replace")
    sandbox.write_text("reports/result.md", "updated", overwrite=True)
    assert sandbox.read_text("reports/result.md") == "updated"


@pytest.mark.parametrize(
    "path",
    [
        "../secret.txt",
        "reports/../../secret.txt",
        "C:\\Windows\\system.ini",
        "C:drive-relative.txt",
        "/etc/passwd",
        "\\\\server\\share\\file.txt",
        "file.txt:secret",
        "dir:stream\\file.txt",
        "CON",
        "COM1",
        "LPT9.txt",
        "nul.txt",
        "ambiguous.",
        "ambiguous ",
        "\x00bad",
        "",
    ],
)
def test_workspace_rejects_unsafe_paths(tmp_path: Path, path: str) -> None:
    sandbox = WorkspaceSandbox(tmp_path / "workspace")

    with pytest.raises(SandboxViolation):
        sandbox.resolve(path)


def test_workspace_enforces_read_write_and_listing_limits(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(
        tmp_path / "workspace",
        limits=WorkspaceLimits(max_read_bytes=3, max_write_bytes=3, max_list_entries=1),
    )
    sandbox.write_bytes("one", b"123")
    with pytest.raises(SandboxViolation) as write_error:
        sandbox.write_bytes("too-big", b"1234")
    assert write_error.value.code == "write_limit"
    (sandbox.root / "external").write_bytes(b"1234")
    with pytest.raises(SandboxViolation) as read_error:
        sandbox.read_bytes("external")
    assert read_error.value.code == "read_limit"
    sandbox.write_bytes("two", b"2")
    with pytest.raises(SandboxViolation) as list_error:
        sandbox.list_files()
    assert list_error.value.code == "list_limit"


def test_workspace_rejects_symlink_escape_when_supported(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    sandbox = WorkspaceSandbox(tmp_path / "workspace")
    link = sandbox.root / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("creating symlinks is not permitted on this host")

    with pytest.raises(SandboxViolation) as error:
        sandbox.read_text("escape/secret.txt")
    assert error.value.code == "link_denied"
    assert sandbox.list_files() == ()


def test_workspace_rejects_non_regular_files(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(tmp_path / "workspace")
    directory = sandbox.root / "directory"
    directory.mkdir()

    with pytest.raises(SandboxViolation) as error:
        sandbox.read_bytes("directory")
    assert error.value.code == "not_regular_file"


def test_workspace_error_semantics_distinguish_missing_existing_and_type(
    tmp_path: Path,
) -> None:
    sandbox = WorkspaceSandbox(tmp_path / "workspace")

    with pytest.raises(SandboxNotFoundError) as not_found:
        sandbox.read_text("missing.txt")
    assert not_found.value.code == "not_found"
    with pytest.raises(SandboxNotFoundError):
        sandbox.list_files("missing-dir")

    sandbox.write_text("entry.txt", "content")
    with pytest.raises(SandboxExistsError) as exists:
        sandbox.write_text("entry.txt", "other")
    assert exists.value.code == "already_exists"
    with pytest.raises(SandboxNotDirectoryError) as not_directory:
        sandbox.list_files("entry.txt")
    assert not_directory.value.code == "not_directory"
    assert isinstance(not_found.value, SandboxViolation)
    assert isinstance(exists.value, SandboxViolation)


def test_workspace_successful_paths_stay_under_root(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(tmp_path / "workspace")

    written = sandbox.write_text("nested/deep/file.txt", "payload")

    assert written.is_relative_to(sandbox.root)
    assert sandbox.resolve("nested/deep/file.txt").is_relative_to(sandbox.root)
    for entry in sandbox.list_files():
        assert not entry.startswith(("/", "../"))
        assert ".." not in Path(entry).parts
        assert (sandbox.root / entry).is_relative_to(sandbox.root)


def test_workspace_rejects_hard_link_escape_when_supported(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("host secret", encoding="utf-8")
    sandbox = WorkspaceSandbox(tmp_path / "workspace")
    link = sandbox.root / "hard-link.txt"
    try:
        link.hardlink_to(outside)
    except OSError:
        pytest.skip("creating hard links is not permitted on this host")

    with pytest.raises(SandboxViolation) as read_error:
        sandbox.read_text("hard-link.txt")
    assert read_error.value.code == "hard_link_denied"
    with pytest.raises(SandboxViolation) as write_error:
        sandbox.write_text("hard-link.txt", "overwrite", overwrite=True)
    assert write_error.value.code == "hard_link_denied"
    assert outside.read_text(encoding="utf-8") == "host secret"


@pytest.mark.skipif(os.name != "nt", reason="directory junctions are Windows-only")
def test_workspace_rejects_junction_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    sandbox = WorkspaceSandbox(tmp_path / "workspace")
    junction = sandbox.root / "junction"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("creating junctions is not permitted on this host")

    with pytest.raises(SandboxViolation) as read_error:
        sandbox.read_text("junction/secret.txt")
    assert read_error.value.code == "link_denied"
    with pytest.raises(SandboxViolation):
        sandbox.write_text("junction/new.txt", "payload")
    assert sandbox.list_files() == ()
    assert not (outside / "new.txt").exists()


def test_workspace_write_survives_parent_directory_swap_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    sandbox = WorkspaceSandbox(tmp_path / "workspace")
    (sandbox.root / "dir").mkdir()
    real_open = os.open
    swapped = False

    def racing_open(path: str, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if not swapped and str(path).endswith("target.txt") and flags & os.O_CREAT:
            swapped = True
            victim = sandbox.root / "dir"
            victim.rmdir()
            try:
                victim.symlink_to(outside, target_is_directory=True)
            except OSError:
                victim.mkdir()
                raise
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", racing_open)
    try:
        with pytest.raises(SandboxViolation) as error:
            sandbox.write_text("dir/target.txt", "payload")
    except OSError:
        pytest.skip("creating symlinks is not permitted on this host")
    assert error.value.code == "link_denied"
    assert not (outside / "target.txt").exists()


def test_workspace_read_detects_file_replacement_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("host secret", encoding="utf-8")
    sandbox = WorkspaceSandbox(tmp_path / "workspace")
    sandbox.write_text("target.txt", "safe")
    real_open = os.open
    swapped = False

    def racing_open(path: str, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if not swapped and str(path).endswith("target.txt"):
            swapped = True
            os.unlink(path)
            os.link(outside, path)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", racing_open)
    with pytest.raises(SandboxViolation) as error:
        sandbox.read_text("target.txt")
    assert error.value.code == "replaced_file"
    assert outside.read_text(encoding="utf-8") == "host secret"


def test_workspace_overwrite_detects_file_replacement_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("host secret", encoding="utf-8")
    sandbox = WorkspaceSandbox(tmp_path / "workspace")
    sandbox.write_text("target.txt", "safe")
    real_open = os.open
    swapped = False

    def racing_open(path: str, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if not swapped and str(path).endswith("target.txt"):
            swapped = True
            os.unlink(path)
            os.link(outside, path)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", racing_open)
    with pytest.raises(SandboxViolation) as error:
        sandbox.write_text("target.txt", "overwrite", overwrite=True)
    assert error.value.code == "replaced_file"
    assert outside.read_text(encoding="utf-8") == "host secret"


def test_workspace_concurrent_parent_creation_is_atomic(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(tmp_path / "workspace")
    worker_count = 8
    barrier = threading.Barrier(worker_count)
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            barrier.wait(timeout=30)
            sandbox.write_text(f"shared/file-{index}.txt", f"payload-{index}")
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors
    assert len(sandbox.list_files()) == worker_count


def test_workspace_exclusive_write_allows_a_single_winner(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(tmp_path / "workspace")
    barrier = threading.Barrier(2)
    outcomes: list[tuple[str, str]] = []
    lock = threading.Lock()

    def worker(tag: str) -> None:
        barrier.wait(timeout=30)
        try:
            sandbox.write_text("contended.txt", tag)
            outcome = ("ok", tag)
        except SandboxExistsError:
            outcome = ("exists", tag)
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=worker, args=(tag,)) for tag in ("alpha", "beta")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert sorted(outcome for outcome, _ in outcomes) == ["exists", "ok"]
    winner = next(tag for outcome, tag in outcomes if outcome == "ok")
    assert sandbox.read_text("contended.txt") == winner


def test_workspace_overwrite_failures_preserve_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = WorkspaceSandbox(tmp_path / "workspace", limits=WorkspaceLimits(max_write_bytes=16))
    sandbox.write_text("data.txt", "original")

    with pytest.raises(SandboxViolation) as limit_error:
        sandbox.write_text("data.txt", "x" * 17, overwrite=True)
    assert limit_error.value.code == "write_limit"
    assert sandbox.read_text("data.txt") == "original"

    (sandbox.root / "directory").mkdir()
    with pytest.raises(SandboxViolation) as type_error:
        sandbox.write_text("directory", "payload", overwrite=True)
    assert type_error.value.code == "not_regular_file"

    real_open = os.open

    def denied_open(path: str, flags: int, *args: object, **kwargs: object) -> int:
        if str(path).endswith("data.txt"):
            raise PermissionError("denied by test")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", denied_open)
    with pytest.raises(SandboxViolation) as denied_error:
        sandbox.write_text("data.txt", "updated", overwrite=True)
    assert denied_error.value.code == "permission_denied"
    monkeypatch.undo()
    assert sandbox.read_text("data.txt") == "original"


def test_workspace_failed_create_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = WorkspaceSandbox(tmp_path / "workspace")

    def swapped_check(self: WorkspaceSandbox, path: Path, opened: os.stat_result) -> None:
        raise SandboxViolation("replaced_file", "workspace entry was replaced")

    monkeypatch.setattr(WorkspaceSandbox, "_verify_open_within_root", swapped_check)
    with pytest.raises(SandboxViolation) as error:
        sandbox.write_text("partial.txt", "data")
    assert error.value.code == "replaced_file"
    assert not os.path.lexists(sandbox.root / "partial.txt")
