"""Deterministic coverage for JSONL dataset loading and error reporting."""

from __future__ import annotations

import json
import re
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

SECRET = "s3cr3t-payload-marker"


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_skips_bom_blank_lines_comments_and_trailing_whitespace(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "bom.jsonl",
        '\ufeff{"id":"a","input":"x"}   \n\n   \n# a comment\n\t{"id":"b","input":"y"}\t\n',
    )

    cases = load_jsonl(path)

    assert [case.id for case in cases] == ["a", "b"]


def test_load_preserves_order_and_stays_serializable(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "ordered.jsonl",
        "".join(
            json.dumps({"id": f"case-{index}", "input": f"prompt {index}", "tags": ["t"]}) + "\n"
            for index in range(5)
        ),
    )

    first = load_jsonl(path)
    second = load_jsonl(path)

    assert isinstance(first, tuple)
    assert [case.id for case in first] == [f"case-{index}" for index in range(5)]
    assert first == second
    serialized = json.dumps([case.model_dump(mode="json") for case in first])
    assert json.loads(serialized)[0]["id"] == "case-0"
    assert all(json.loads(case.model_dump_json())["id"] for case in first)


def test_load_rejects_invalid_json_with_location_and_without_payload(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "broken.jsonl",
        '{"id":"a","input":"x"}\n' + f'{{"id":"b","input":"{SECRET}",}}\n',
    )

    with pytest.raises(DatasetError) as excinfo:
        load_jsonl(path)

    message = str(excinfo.value)
    assert "broken.jsonl:2" in message
    assert "invalid JSON" in message
    assert SECRET not in message


def test_load_rejects_deeply_nested_json(tmp_path: Path) -> None:
    path = _write(tmp_path / "nested.jsonl", "[" * 50_000 + "]" * 50_000 + "\n")

    with pytest.raises(DatasetError, match=r"nested.jsonl:1: invalid JSON: nesting too deep"):
        load_jsonl(path)


@pytest.mark.parametrize("record", ["[1, 2, 3]", '"just a string"', "42", "null", "true"])
def test_load_rejects_non_object_records(tmp_path: Path, record: str) -> None:
    path = _write(tmp_path / "array.jsonl", record + "\n")

    with pytest.raises(DatasetError, match=r"array.jsonl:1: expected a JSON object"):
        load_jsonl(path)


@pytest.mark.parametrize(
    ("record", "field"),
    [
        ({"input": "x"}, "id"),
        ({"id": "a"}, "input"),
        ({"id": "", "input": "x"}, "id"),
        ({"id": 7, "input": "x"}, "id"),
        ({"id": "a", "input": 3.5}, "input"),
        ({"id": "a", "input": "x", "tags": "oops"}, "tags"),
        ({"id": "a", "input": "x", "metadata": ["not", "a", "dict"]}, "metadata"),
    ],
)
def test_load_rejects_missing_and_mistyped_fields_without_values(
    tmp_path: Path, record: dict[str, object], field: str
) -> None:
    path = _write(tmp_path / "fields.jsonl", json.dumps(record) + "\n")

    with pytest.raises(DatasetError) as excinfo:
        load_jsonl(path)

    message = str(excinfo.value)
    assert "fields.jsonl:1" in message
    assert field in message
    assert "oops" not in message


def test_load_rejects_duplicate_case_ids_with_second_location(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "dupes.jsonl",
        '{"id":"same","input":"a"}\n{"id":"other","input":"b"}\n{"id":"same","input":"c"}\n',
    )

    with pytest.raises(DatasetError, match=r"dupes.jsonl:3: duplicate case id"):
        load_jsonl(path)


def test_load_rejects_oversized_id_input_and_line(tmp_path: Path) -> None:
    long_id = _write(
        tmp_path / "long_id.jsonl",
        json.dumps({"id": "i" * (MAX_ID_LENGTH + 1), "input": "x"}) + "\n",
    )
    with pytest.raises(DatasetError, match=r"long_id.jsonl:1: field 'id' exceeds maximum length"):
        load_jsonl(long_id)

    long_input = _write(
        tmp_path / "long_input.jsonl",
        json.dumps({"id": "a", "input": "x" * (MAX_INPUT_LENGTH + 1)}) + "\n",
    )
    with pytest.raises(
        DatasetError, match=r"long_input.jsonl:1: field 'input' exceeds maximum length"
    ):
        load_jsonl(long_input)

    long_line = _write(tmp_path / "long_line.jsonl", " " * (MAX_LINE_LENGTH + 1) + "\n")
    with pytest.raises(DatasetError, match=r"long_line.jsonl:1: line exceeds maximum length"):
        load_jsonl(long_line)


def test_load_round_trips_unicode_content(tmp_path: Path) -> None:
    cases = (
        EvalCase(
            id="用例-🚀",
            input="翻译：你好，世界 🌍",  # noqa: RUF001
            expected={"reply": "مرحبا", "emoji": "✅"},
            metadata={"note": "Grüße — テスト"},
            tags={"中文", "emoji"},
        ),
        EvalCase(id="plain", input="ascii only"),
    )
    path = tmp_path / "unicode.jsonl"
    write_jsonl(cases, path)

    loaded = load_jsonl(path)

    assert loaded == cases
    assert loaded[0].input == "翻译：你好，世界 🌍"  # noqa: RUF001
    json.dumps([case.model_dump(mode="json") for case in loaded], ensure_ascii=False)


def test_load_reports_unreadable_file_without_line_number(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"

    with pytest.raises(DatasetError) as excinfo:
        load_jsonl(missing)

    message = str(excinfo.value)
    assert "missing.jsonl" in message
    assert "cannot read dataset" in message
    assert not re.search(r"missing\.jsonl:\d", message)


class _FailingHandle:
    """File-like object whose second read raises, simulating mid-stream I/O failure."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self._reads = 0

    def readline(self, size: int = -1) -> str:
        self._reads += 1
        if self._reads == 1:
            return '{"id":"a","input":"x"}\n'
        raise self._error

    def __enter__(self) -> _FailingHandle:
        return self

    def __exit__(self, *args: object) -> None:
        return None


@pytest.mark.parametrize(
    "error",
    [
        OSError("disk on fire"),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
    ],
)
def test_load_converts_mid_stream_iterator_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    handle = _FailingHandle(error)
    monkeypatch.setattr(Path, "open", lambda self, **kwargs: handle)

    with pytest.raises(DatasetError) as excinfo:
        load_jsonl(tmp_path / "flaky.jsonl")

    message = str(excinfo.value)
    assert "flaky.jsonl:2" in message
    assert "cannot read line" in message
    assert "disk on fire" not in message


def test_load_does_not_leak_payload_in_non_object_error(tmp_path: Path) -> None:
    path = _write(tmp_path / "secret.jsonl", f'["{SECRET}"]\n')

    with pytest.raises(DatasetError) as excinfo:
        load_jsonl(path)

    assert SECRET not in str(excinfo.value)


def test_write_then_load_is_stable_round_trip(tmp_path: Path) -> None:
    cases = tuple(
        EvalCase(id=f"rt-{index}", input=f"in {index}", expected={"v": index}) for index in range(3)
    )
    path = tmp_path / "nested" / "round_trip.jsonl"

    write_jsonl(cases, path)
    write_jsonl(load_jsonl(path), path)

    assert load_jsonl(path) == cases
