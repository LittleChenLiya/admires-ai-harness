"""Deterministic coverage for JSONL dataset loading and error reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from admires_harness.evaluation.dataset import (
    MAX_ID_LENGTH,
    MAX_INPUT_LENGTH,
    MAX_LINE_LENGTH,
    DatasetError,
    EvalCase,
    load_jsonl,
    write_jsonl,
)

SECRET = "sk-live-secret-payload-9f3a"


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_jsonl_skips_bom_blank_lines_comments_and_trailing_whitespace(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_bytes(
        b"\xef\xbb\xbf"  # UTF-8 BOM
        b"\n"
        b"   \t\n"
        b"# a comment\n"
        b'{"id":"one","input":"hello"}   \n'
        b'{"id":"two","input":"world"}\t\n'
    )

    cases = load_jsonl(path)

    assert [case.id for case in cases] == ["one", "two"]


def test_load_jsonl_preserves_order_and_stays_serializable(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "dataset.jsonl",
        "\n".join(
            json.dumps({"id": f"case-{index}", "input": f"prompt {index}", "tags": ["t"]})
            for index in range(5)
        )
        + "\n",
    )

    cases = load_jsonl(path)

    assert [case.id for case in cases] == [f"case-{index}" for index in range(5)]
    assert isinstance(cases, tuple)
    json.dumps([case.model_dump(mode="json") for case in cases])
    for case in cases:
        json.loads(case.model_dump_json())


def test_load_jsonl_accepts_unicode_content(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "unicode.jsonl",
        '{"id":"用例-🚀","input":"输入带有 emoji 🎉 与中文","expected":"结果","tags":["标签"]}\n',
    )

    (case,) = load_jsonl(path)

    assert case.id == "用例-🚀"
    assert case.input == "输入带有 emoji 🎉 与中文"
    assert case.tags == frozenset({"标签"})

    round_tripped = tmp_path / "round-trip.jsonl"
    write_jsonl((case,), round_tripped)
    assert load_jsonl(round_tripped) == (case,)


def test_load_jsonl_rejects_invalid_json_with_location_and_no_payload(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path / "broken.jsonl",
        f'{{"id":"ok","input":"fine"}}\n{{"id":"bad","input":"{SECRET}",}}\n',
    )

    with pytest.raises(DatasetError) as excinfo:
        load_jsonl(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert ":2:" in message
    assert "invalid JSON" in message
    assert SECRET not in message


@pytest.mark.parametrize(
    ("record", "kind"),
    [
        ("[1, 2, 3]", "array"),
        ('"just a string"', "string"),
        ("42", "number"),
        ("null", "null"),
        ("true", "boolean"),
    ],
)
def test_load_jsonl_rejects_non_object_records(tmp_path: Path, record: str, kind: str) -> None:
    path = _write(tmp_path / "non-object.jsonl", record + "\n")

    with pytest.raises(DatasetError, match=rf"expected a JSON object, got {kind}") as excinfo:
        load_jsonl(path)

    assert f"{path}:1:" in str(excinfo.value)


@pytest.mark.parametrize(
    ("record", "field"),
    [
        ('{"input":"x"}', "id"),
        ('{"id":"a"}', "input"),
    ],
)
def test_load_jsonl_rejects_missing_required_fields(
    tmp_path: Path, record: str, field: str
) -> None:
    path = _write(tmp_path / "missing.jsonl", record + "\n")

    with pytest.raises(DatasetError, match=rf"field '{field}'.*Field required") as excinfo:
        load_jsonl(path)

    assert f"{path}:1:" in str(excinfo.value)


@pytest.mark.parametrize(
    ("record", "field"),
    [
        ('{"id":1,"input":"x"}', "id"),
        ('{"id":"a","input":2}', "input"),
        ('{"id":"a","input":"x","tags":"not-a-list"}', "tags"),
        ('{"id":"a","input":"x","metadata":[1]}', "metadata"),
    ],
)
def test_load_jsonl_rejects_wrong_field_types(tmp_path: Path, record: str, field: str) -> None:
    path = _write(tmp_path / "types.jsonl", record + "\n")

    with pytest.raises(DatasetError, match=rf"invalid evaluation case: field '{field}'"):
        load_jsonl(path)


def test_load_jsonl_rejects_unknown_fields_without_leaking_values(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "extra.jsonl",
        f'{{"id":"a","input":"x","payload":"{SECRET}"}}\n',
    )

    with pytest.raises(DatasetError, match="field 'payload'") as excinfo:
        load_jsonl(path)

    assert SECRET not in str(excinfo.value)


def test_load_jsonl_rejects_duplicate_ids_with_both_line_numbers(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "duplicates.jsonl",
        "# comment\n"
        '{"id":"dup","input":"alpha"}\n'
        "\n"
        '{"id":"other","input":"ok"}\n'
        '{"id":"dup","input":"omega"}\n',
    )

    with pytest.raises(DatasetError) as excinfo:
        load_jsonl(path)

    message = str(excinfo.value)
    assert f"{path}:5:" in message
    assert "duplicate case id 'dup'" in message
    assert "first defined on line 2" in message
    assert "alpha" not in message
    assert "omega" not in message


def test_load_jsonl_rejects_oversized_id_and_input(tmp_path: Path) -> None:
    long_id = _write(
        tmp_path / "long-id.jsonl",
        json.dumps({"id": "x" * (MAX_ID_LENGTH + 1), "input": "ok"}) + "\n",
    )
    with pytest.raises(DatasetError, match=r"long-id.jsonl:1: field 'id' exceeds maximum"):
        load_jsonl(long_id)

    long_input = _write(
        tmp_path / "long-input.jsonl",
        json.dumps({"id": "a", "input": "y" * (MAX_INPUT_LENGTH + 1)}) + "\n",
    )
    with pytest.raises(DatasetError, match=r"field 'input' exceeds maximum"):
        load_jsonl(long_input)

    at_limit = _write(
        tmp_path / "at-limit.jsonl",
        json.dumps({"id": "x" * MAX_ID_LENGTH, "input": "y" * MAX_INPUT_LENGTH}) + "\n",
    )
    (case,) = load_jsonl(at_limit)
    assert len(case.id) == MAX_ID_LENGTH
    assert len(case.input) == MAX_INPUT_LENGTH


def test_load_jsonl_rejects_oversized_raw_line(tmp_path: Path) -> None:
    path = _write(tmp_path / "huge.jsonl", " " * (MAX_LINE_LENGTH + 1) + "\n")

    with pytest.raises(DatasetError, match=r"huge.jsonl:1: line exceeds maximum length"):
        load_jsonl(path)


def test_load_jsonl_reports_unreadable_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(DatasetError) as excinfo:
        load_jsonl(missing)
    message = str(excinfo.value)
    assert str(missing) in message
    assert "cannot read dataset" in message

    with pytest.raises(DatasetError, match="cannot read dataset"):
        load_jsonl(tmp_path)


def test_load_jsonl_wraps_read_errors_during_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path / "flaky.jsonl", '{"id":"a","input":"x"}\n')
    real_open = Path.open

    class FailingHandle:
        def __init__(self, handle: object) -> None:
            self._handle = handle

        def __enter__(self) -> FailingHandle:
            return self

        def __exit__(self, *exc: object) -> None:
            self._handle.close()  # type: ignore[attr-defined]

        def __iter__(self) -> FailingHandle:
            return self

        def __next__(self) -> str:
            raise OSError("simulated disk failure")

    def fake_open(self: Path, *args: object, **kwargs: object) -> FailingHandle:
        return FailingHandle(real_open(self, *args, **kwargs))  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", fake_open)

    with pytest.raises(DatasetError, match=r"flaky.jsonl:0: failed to read dataset"):
        load_jsonl(path)


def test_eval_case_remains_frozen_and_hashable() -> None:
    case = EvalCase(id="a", input="x", tags={"one", "two"})
    with pytest.raises(Exception, match="frozen"):
        case.id = "b"  # type: ignore[misc]
    assert case.tags == frozenset({"one", "two"})
