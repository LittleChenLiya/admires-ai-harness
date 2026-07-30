"""Versionable JSONL evaluation datasets."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from admires_harness.contracts import JsonValue


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


def load_jsonl(path: str | Path) -> tuple[EvalCase, ...]:
    """Load strict JSONL, rejecting blank IDs, invalid rows, and duplicates."""

    source = Path(path)
    cases: list[EvalCase] = []
    seen: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                raw: Any = json.loads(stripped)
                case = EvalCase.model_validate(raw)
            except (json.JSONDecodeError, ValidationError) as exc:
                raise DatasetError(
                    f"{source}:{line_number}: invalid evaluation case: {exc}"
                ) from exc
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
