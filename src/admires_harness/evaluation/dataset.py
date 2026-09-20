"""Versionable JSONL evaluation datasets."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from admires_harness.contracts import JsonValue

MAX_ID_LENGTH = 256
"""Maximum accepted length of an EvalCase id."""

MAX_INPUT_LENGTH = 100_000
"""Maximum accepted length of an EvalCase input."""

MAX_LINE_LENGTH = 1_000_000
"""Maximum accepted length of a single raw JSONL line."""


class EvalCase(BaseModel):
    """One immutable benchmark case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    input: str
    expected: JsonValue = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    tags: frozenset[str] = frozenset()


class DatasetError(ValueError):
    """An invalid evaluation dataset, including its source line.

    Messages always carry the dataset path and, when available, the 1-based
    source line number. They never echo record payloads, so datasets may
    contain sensitive inputs without leaking them into logs or CI output.
    """


def _json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def _format_validation_error(exc: ValidationError) -> str:
    """Summarize pydantic errors without embedding rejected input values."""

    parts = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<case>"
        parts.append(f"field {location!r}: {error['msg']}")
    return "; ".join(parts)


def load_jsonl(path: str | Path) -> tuple[EvalCase, ...]:
    """Load strict JSONL, rejecting blank IDs, invalid rows, and duplicates.

    The file is decoded as UTF-8 (an optional leading BOM is stripped) and
    processed line by line so arbitrarily large datasets stream. Blank or
    whitespace-only lines and '#' comments are skipped; surrounding
    whitespace on record lines is ignored. Every failure raises
    :class:`DatasetError` pointing at the path and source line.
    """

    source = Path(path)
    cases: list[EvalCase] = []
    seen: dict[str, int] = {}
    line_number = 0
    try:
        handle = source.open(encoding="utf-8-sig")
    except OSError as exc:
        reason = exc.strerror or str(exc)
        raise DatasetError(f"{source}: cannot read dataset: {reason}") from exc
    with handle:
        try:
            for line_number, line in enumerate(handle, start=1):
                if len(line) > MAX_LINE_LENGTH:
                    raise DatasetError(
                        f"{source}:{line_number}: line exceeds maximum length of "
                        f"{MAX_LINE_LENGTH} characters"
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
                if not isinstance(raw, dict):
                    raise DatasetError(
                        f"{source}:{line_number}: expected a JSON object, "
                        f"got {_json_type_name(raw)}"
                    )
                try:
                    case = EvalCase.model_validate(raw)
                except ValidationError as exc:
                    raise DatasetError(
                        f"{source}:{line_number}: invalid evaluation case: "
                        f"{_format_validation_error(exc)}"
                    ) from exc
                if len(case.id) > MAX_ID_LENGTH:
                    raise DatasetError(
                        f"{source}:{line_number}: field 'id' exceeds maximum length of "
                        f"{MAX_ID_LENGTH} characters (got {len(case.id)})"
                    )
                if len(case.input) > MAX_INPUT_LENGTH:
                    raise DatasetError(
                        f"{source}:{line_number}: field 'input' exceeds maximum length of "
                        f"{MAX_INPUT_LENGTH} characters (got {len(case.input)})"
                    )
                if case.id in seen:
                    raise DatasetError(
                        f"{source}:{line_number}: duplicate case id {case.id!r} "
                        f"(first defined on line {seen[case.id]})"
                    )
                seen[case.id] = line_number
                cases.append(case)
        except OSError as exc:
            raise DatasetError(f"{source}:{line_number}: failed to read dataset: {exc}") from exc
    return tuple(cases)


def write_jsonl(cases: Iterable[EvalCase], path: str | Path) -> None:
    """Write a stable, diff-friendly JSONL representation."""

    source = Path(path)
    source.parent.mkdir(parents=True, exist_ok=True)
    with source.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(case.model_dump_json(exclude_none=True) + "\n")
