from __future__ import annotations

from pathlib import Path

import pytest

from admires_harness.safety import SandboxViolation, WorkspaceLimits, WorkspaceSandbox

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
        "/etc/passwd",
        "file.txt:secret",
        "CON",
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
