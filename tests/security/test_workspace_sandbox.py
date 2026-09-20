from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

import pytest

from admires_harness.safety import SandboxViolation, WorkspaceLimits, WorkspaceSandbox

pytestmark = pytest.mark.security


def _make_directory_link(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError:
        pass
    if os.name == "nt":
        # Junctions do not require elevated privileges on Windows.
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return
    pytest.skip("creating directory links is not permitted on this host")


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
        "/etc/passwd",
        "file.txt:secret",
        "CON",
        "nul.txt",
        "com1",
        "LPT9.log",
        "aux",
        "C:file.txt",
        "\\\\server\\share\\file.txt",
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


def test_workspace_rejects_junction_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    sandbox = WorkspaceSandbox(tmp_path / "workspace")
    _make_directory_link(sandbox.root / "junction", outside)

    with pytest.raises(SandboxViolation) as read_error:
        sandbox.read_text("junction/secret.txt")
    assert read_error.value.code == "link_denied"
    with pytest.raises(SandboxViolation) as write_error:
        sandbox.write_text("junction/new.txt", "escaped")
    assert write_error.value.code == "link_denied"
    assert sandbox.list_files() == ()
    assert not (outside / "new.txt").exists()


def test_workspace_rejects_parent_swapped_for_link_during_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    sandbox = WorkspaceSandbox(tmp_path / "workspace")
    real_mkdir = Path.mkdir

    def fake_mkdir(self: Path, *args: object, **kwargs: object) -> None:
        if self == sandbox.root / "race":
            _make_directory_link(self, outside)
            raise FileExistsError(str(self))
        real_mkdir(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "mkdir", fake_mkdir)

    with pytest.raises(SandboxViolation) as error:
        sandbox.write_text("race/secret.txt", "pwned")
    assert error.value.code == "link_denied"
    assert (outside / "secret.txt").read_text(encoding="utf-8") == "secret"


def test_workspace_rejects_file_replaced_by_hard_link_mid_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("host secret", encoding="utf-8")
    sandbox = WorkspaceSandbox(tmp_path / "workspace")
    sandbox.write_text("target.txt", "safe")
    real_open = os.open

    def fake_open(path: object, flags: int, *args: object) -> int:
        candidate = Path(path)  # type: ignore[arg-type]
        if candidate == sandbox.root / "target.txt":
            candidate.unlink()
            candidate.hardlink_to(outside)
        return real_open(path, flags, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", fake_open)

    with pytest.raises(SandboxViolation) as error:
        sandbox.read_text("target.txt")
    assert error.value.code == "hard_link_denied"
    assert outside.read_text(encoding="utf-8") == "host secret"


def test_workspace_detects_descriptor_path_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = WorkspaceSandbox(tmp_path / "workspace")
    sandbox.write_text("target.txt", "safe")
    sandbox.write_text("other.txt", "other")
    real_fstat = os.fstat

    def fake_fstat(descriptor: int) -> os.stat_result:
        stat_result = real_fstat(descriptor)
        if stat_result.st_size == len("safe"):
            return os.stat(sandbox.root / "other.txt")
        return stat_result

    monkeypatch.setattr(os, "fstat", fake_fstat)

    with pytest.raises(SandboxViolation) as error:
        sandbox.read_text("target.txt")
    assert error.value.code == "path_replaced"


def test_workspace_rejects_file_as_parent(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(tmp_path / "workspace")
    sandbox.write_text("plain.txt", "content")

    with pytest.raises(SandboxViolation) as error:
        sandbox.write_text("plain.txt/child.txt", "nested")
    assert error.value.code == "unsafe_parent"
    assert sandbox.read_text("plain.txt") == "content"
    assert sandbox.list_files() == ("plain.txt",)


def test_concurrent_parent_creation_is_atomic(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(tmp_path / "workspace")
    worker_count = 8
    barrier = threading.Barrier(worker_count)
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        barrier.wait()
        try:
            sandbox.write_text(f"shared/nested/file-{index}.txt", f"content-{index}")
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(sandbox.list_files()) == worker_count
    for index in range(worker_count):
        assert sandbox.read_text(f"shared/nested/file-{index}.txt") == f"content-{index}"


def test_concurrent_exclusive_write_has_single_winner(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(tmp_path / "workspace")
    payloads = (b"aaaa" * 100, b"bbbb" * 100)
    barrier = threading.Barrier(len(payloads))
    outcomes: list[str] = []

    def worker(payload: bytes) -> None:
        barrier.wait()
        try:
            sandbox.write_bytes("contended.bin", payload)
            outcomes.append("created")
        except FileExistsError:
            outcomes.append("exists")

    threads = [threading.Thread(target=worker, args=(payload,)) for payload in payloads]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["created", "exists"]
    assert sandbox.read_bytes("contended.bin") in payloads


def test_overwrite_failure_leaves_original_intact(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(tmp_path / "workspace")
    path = sandbox.write_text("locked.txt", "original")
    os.chmod(path, 0o444)
    try:
        with pytest.raises(SandboxViolation) as error:
            sandbox.write_text("locked.txt", "replacement", overwrite=True)
        assert error.value.code == "permission_denied"
        assert sandbox.read_text("locked.txt") == "original"
    finally:
        os.chmod(path, 0o666)


def test_rejected_writes_leave_no_partial_files(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(tmp_path / "workspace")
    sandbox.write_text("kept.txt", "kept")

    for attempt in ("../escape.txt", "CON", "a/b/../..", "stream.txt:ads", "trailing."):
        with pytest.raises(SandboxViolation):
            sandbox.write_text(attempt, "x")
    with pytest.raises(FileExistsError):
        sandbox.write_text("kept.txt", "other")
    with pytest.raises(FileNotFoundError):
        sandbox.read_text("missing.txt")

    assert sandbox.list_files() == ("kept.txt",)
    assert sandbox.read_text("kept.txt") == "kept"
    assert not (tmp_path / "escape.txt").exists()


def test_successful_operations_stay_inside_root(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(tmp_path / "workspace")

    written = sandbox.write_text("sub/dir/file.txt", "payload")

    assert written.is_relative_to(sandbox.root)
    assert sandbox.resolve("sub/dir/file.txt").is_relative_to(sandbox.root)
    assert sandbox.list_files() == ("sub/dir/file.txt",)
    assert sandbox.read_text("sub/dir/file.txt") == "payload"
