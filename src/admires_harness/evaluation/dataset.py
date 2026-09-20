"""Versionable JSONL evaluation datasets."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from admires_harness.contracts import JsonValue

MAX_ID_LENGTH = 256
"""Maximum accepted length of an evaluation case ID, in characters."""

MAX_INPUT_LENGTH = 100_000
"""Maximum accepted length of an evaluation case input, in characters."""

MAX_LINE_LENGTH = 1_000_000
"""Maximum accepted length of a single JSONL line, in characters."""


class EvalCase(BaseModel):
    """One immutable benchmark case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    input: str
    expected: JsonValue = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    tags: frozenset[str] = frozenset()


class DatasetError(ValueError):
    """An invalid evaluation dataset, including its source line."""


def _describe_validation_error(exc: ValidationError) -> str:
    """Summarize field errors without echoing any submitted values."""

    parts = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        parts.append(f"{location}: {error['type']}")
    return "; ".join(parts)


def _read_line(handle: TextIO, source: Path, line_number: int) -> str:
    """Read one line, converting I/O and decoding failures into DatasetError."""

    try:
        return handle.readline(MAX_LINE_LENGTH + 2)
    except (OSError, UnicodeError) as exc:
        raise DatasetError(
            f"{source}:{line_number + 1}: cannot read line: {type(exc).__name__}"
        ) from exc


def _parse_case(raw: Any, source: Path, line_number: int) -> EvalCase:
    if not isinstance(raw, dict):
        raise DatasetError(
            f"{source}:{line_number}: expected a JSON object, got {type(raw).__name__}"
        )
    raw_id = raw.get("id")
    if isinstance(raw_id, str) and len(raw_id) > MAX_ID_LENGTH:
        raise DatasetError(
            f"{source}:{line_number}: field 'id' exceeds maximum length of {MAX_ID_LENGTH}"
        )
    raw_input = raw.get("input")
    if isinstance(raw_input, str) and len(raw_input) > MAX_INPUT_LENGTH:
        raise DatasetError(
            f"{source}:{line_number}: field 'input' exceeds maximum length of {MAX_INPUT_LENGTH}"
        )
    try:
        return EvalCase.model_validate(raw)
    except ValidationError as exc:
        raise DatasetError(
            f"{source}:{line_number}: invalid evaluation case: {_describe_validation_error(exc)}"
        ) from exc


def load_jsonl(path: str | Path) -> tuple[EvalCase, ...]:
    """Load strict JSONL, rejecting blank IDs, invalid rows, and duplicates.

    The file is decoded as UTF-8 (an optional BOM is tolerated) and processed
    line by line so arbitrarily large datasets stream with constant memory.
    Blank lines and '#' comments are skipped. Every failure raises
    DatasetError carrying the source path and 1-based line number without
    echoing the offending row content.
    """

    source = Path(path)
    cases: list[EvalCase] = []
    seen: set[str] = set()
    try:
        handle = source.open(encoding="utf-8-sig")
    except OSError as exc:
        raise DatasetError(f"{source}: cannot read dataset: {type(exc).__name__}") from exc
    with handle:
        line_number = 0
        while True:
            line = _read_line(handle, source, line_number)
            if line == "":
                break
            line_number += 1
            if len(line) > MAX_LINE_LENGTH + 1 or (
                len(line) == MAX_LINE_LENGTH + 1 and not line.endswith("\n")
            ):
                raise DatasetError(
                    f"{source}:{line_number}: line exceeds maximum length of {MAX_LINE_LENGTH}"
                )
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                raw: Any = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetError(
                    f"{source}:{line_number}: invalid JSON: {exc.msg} (column {exc.colno})"
                ) from exc
            except RecursionError as exc:
                raise DatasetError(
                    f"{source}:{line_number}: invalid JSON: nesting too deep"
                ) from exc
            case = _parse_case(raw, source, line_number)
            if case.id in seen:
                raise DatasetError(f"{source}:{line_number}: duplicate case id {case.id!r}")
            seen.add(case.id)
            cases.append(case)
    return tuple(cases)


def write_jsonl(cases: Iterable[EvalCase], path: str | Path) -> None:
    """Write a stable, diff-friendly JSONL representation."""

    source = Path(path)
    source.parent.mkdir(parents=True, exist_ok=True)
    with source.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(case.model_dump_json(exclude_none=True) + "\n")
