"""Deterministic built-in scorers for quality, tools, and service-level objectives."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from admires_harness.evaluation.dataset import EvalCase


class Score(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scorer: str
    value: float = Field(ge=0, le=1)
    passed: bool
    reason: str = ""


class ScorableOutput(Protocol):
    text: str
    tool_calls: tuple[str, ...]
    latency_ms: float


class Scorer(Protocol):
    name: str
    weight: float

    def score(self, case: EvalCase, output: ScorableOutput) -> Score: ...


def _expected_text(case: EvalCase) -> str | None:
    if isinstance(case.expected, str):
        return case.expected
    if isinstance(case.expected, dict):
        value = case.expected.get("text")
        return value if isinstance(value, str) else None
    return None


class ExactMatchScorer:
    name = "exact_match"

    def __init__(self, *, case_sensitive: bool = False, weight: float = 1.0) -> None:
        self.case_sensitive = case_sensitive
        self.weight = weight

    def score(self, case: EvalCase, output: ScorableOutput) -> Score:
        expected = _expected_text(case)
        if expected is None:
            return Score(scorer=self.name, value=0, passed=False, reason="expected text is missing")
        actual = output.text.strip()
        wanted = expected.strip()
        if not self.case_sensitive:
            actual, wanted = actual.casefold(), wanted.casefold()
        passed = actual == wanted
        return Score(
            scorer=self.name,
            value=float(passed),
            passed=passed,
            reason="matched" if passed else "output did not exactly match expected text",
        )


class ContainsScorer:
    name = "contains"

    def __init__(self, *, case_sensitive: bool = False, weight: float = 1.0) -> None:
        self.case_sensitive = case_sensitive
        self.weight = weight

    def score(self, case: EvalCase, output: ScorableOutput) -> Score:
        expected = _expected_text(case)
        if expected is None:
            return Score(scorer=self.name, value=0, passed=False, reason="expected text is missing")
        actual = output.text
        wanted = expected
        if not self.case_sensitive:
            actual, wanted = actual.casefold(), wanted.casefold()
        passed = wanted in actual
        return Score(
            scorer=self.name,
            value=float(passed),
            passed=passed,
            reason="contained expected text" if passed else "expected text was absent",
        )


class JsonValidityScorer:
    name = "json_validity"

    def __init__(self, *, weight: float = 1.0) -> None:
        self.weight = weight

    def score(self, case: EvalCase, output: ScorableOutput) -> Score:
        del case
        try:
            json.loads(output.text)
        except json.JSONDecodeError as exc:
            return Score(scorer=self.name, value=0, passed=False, reason=str(exc))
        return Score(scorer=self.name, value=1, passed=True, reason="valid JSON")


class ToolSequenceScorer:
    name = "tool_sequence"

    def __init__(self, *, exact: bool = True, weight: float = 1.0) -> None:
        self.exact = exact
        self.weight = weight

    def score(self, case: EvalCase, output: ScorableOutput) -> Score:
        raw = case.expected.get("tools") if isinstance(case.expected, dict) else None
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            return Score(
                scorer=self.name, value=0, passed=False, reason="expected tools are missing"
            )
        expected = tuple(raw)
        actual = output.tool_calls
        if self.exact:
            passed = actual == expected
        else:
            cursor = iter(actual)
            passed = all(any(candidate == wanted for candidate in cursor) for wanted in expected)
        prefix = 0
        for wanted, candidate in zip(expected, actual, strict=False):
            if wanted != candidate:
                break
            prefix += 1
        value = 1.0 if passed else prefix / max(len(expected), 1)
        return Score(
            scorer=self.name,
            value=value,
            passed=passed,
            reason="tool sequence matched" if passed else f"expected {expected}, got {actual}",
        )


class LatencyScorer:
    name = "latency"

    def __init__(self, threshold_ms: float, *, weight: float = 1.0) -> None:
        if threshold_ms <= 0:
            raise ValueError("threshold_ms must be positive")
        self.threshold_ms = threshold_ms
        self.weight = weight

    def score(self, case: EvalCase, output: ScorableOutput) -> Score:
        del case
        passed = output.latency_ms <= self.threshold_ms
        value = min(1.0, self.threshold_ms / max(output.latency_ms, self.threshold_ms))
        return Score(
            scorer=self.name,
            value=value,
            passed=passed,
            reason=f"{output.latency_ms:.2f} ms (threshold {self.threshold_ms:.2f} ms)",
        )


def weighted_mean(scores: Sequence[Score], scorers: Sequence[Scorer]) -> float:
    weights = [max(0.0, scorer.weight) for scorer in scorers]
    denominator = sum(weights)
    if not scores or denominator == 0:
        return 0.0
    weighted_total = sum(
        score.value * weight for score, weight in zip(scores, weights, strict=True)
    )
    return weighted_total / denominator
