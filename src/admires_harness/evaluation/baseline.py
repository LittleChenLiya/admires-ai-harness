"""Compare benchmark reports and turn regressions into a CI gate."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from admires_harness.evaluation.runner import BenchmarkReport, CaseResult


class CaseChange(BaseModel):
    """A material per-case change between a baseline and a candidate report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    kind: Literal["missing", "new_failure", "score_drop", "improvement", "new_case"]
    baseline_score: float | None = Field(default=None, ge=0, le=1)
    candidate_score: float | None = Field(default=None, ge=0, le=1)
    reason: str


class ReportComparison(BaseModel):
    """Machine-readable comparison suitable for CI artifacts and human review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_name: str
    candidate_name: str
    passed: bool
    pass_rate_delta: float = Field(ge=-1, le=1)
    mean_score_delta: float = Field(ge=-1, le=1)
    regressions: tuple[CaseChange, ...] = ()
    improvements: tuple[CaseChange, ...] = ()
    unchanged_cases: int = Field(default=0, ge=0)

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    def raise_for_regressions(self) -> None:
        """Raise a concise error so a CI command exits unsuccessfully."""

        if self.passed:
            return
        details = "; ".join(f"{change.case_id}: {change.kind}" for change in self.regressions)
        raise EvaluationRegressionError(
            f"benchmark regressed against {self.baseline_name!r}: {details}"
        )


class EvaluationRegressionError(AssertionError):
    """Raised when a candidate violates its configured baseline tolerances."""


def load_report(path: str | Path) -> BenchmarkReport:
    """Load and validate a report previously emitted by ``BenchmarkReport``."""

    return BenchmarkReport.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _score_drop(
    baseline: CaseResult, candidate: CaseResult, *, tolerance: float
) -> CaseChange | None:
    drop = baseline.score - candidate.score
    if drop < tolerance or math.isclose(drop, tolerance, abs_tol=1e-12):
        return None
    return CaseChange(
        case_id=baseline.case_id,
        kind="score_drop",
        baseline_score=baseline.score,
        candidate_score=candidate.score,
        reason=f"case score dropped by {drop:.4f} (allowed {tolerance:.4f})",
    )


def compare_reports(
    candidate: BenchmarkReport,
    baseline: BenchmarkReport,
    *,
    case_score_tolerance: float = 0.0,
    mean_score_tolerance: float = 0.0,
    pass_rate_tolerance: float = 0.0,
    allow_missing_cases: bool = False,
) -> ReportComparison:
    """Compare reports without hiding failures behind an aggregate average.

    A newly failing case is always a regression. Missing baseline cases fail closed unless
    explicitly allowed. Score and aggregate tolerances are absolute fractions in ``[0, 1]``.
    """

    tolerances = {
        "case_score_tolerance": case_score_tolerance,
        "mean_score_tolerance": mean_score_tolerance,
        "pass_rate_tolerance": pass_rate_tolerance,
    }
    for name, value in tolerances.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

    baseline_cases = {result.case_id: result for result in baseline.results}
    candidate_cases = {result.case_id: result for result in candidate.results}
    regressions: list[CaseChange] = []
    improvements: list[CaseChange] = []
    unchanged = 0

    for case_id, expected in baseline_cases.items():
        actual = candidate_cases.get(case_id)
        if actual is None:
            if not allow_missing_cases:
                regressions.append(
                    CaseChange(
                        case_id=case_id,
                        kind="missing",
                        baseline_score=expected.score,
                        reason="candidate report omitted a baseline case",
                    )
                )
            continue
        if expected.passed and not actual.passed:
            regressions.append(
                CaseChange(
                    case_id=case_id,
                    kind="new_failure",
                    baseline_score=expected.score,
                    candidate_score=actual.score,
                    reason=actual.error or "a previously passing case now fails",
                )
            )
            continue
        if not expected.passed and actual.passed:
            improvements.append(
                CaseChange(
                    case_id=case_id,
                    kind="improvement",
                    baseline_score=expected.score,
                    candidate_score=actual.score,
                    reason="a previously failing case now passes",
                )
            )
            continue
        regression = _score_drop(expected, actual, tolerance=case_score_tolerance)
        if regression is not None:
            regressions.append(regression)
        else:
            unchanged += 1

    for case_id, result in candidate_cases.items():
        if case_id not in baseline_cases:
            improvements.append(
                CaseChange(
                    case_id=case_id,
                    kind="new_case",
                    candidate_score=result.score,
                    reason="candidate report adds evaluation coverage",
                )
            )

    pass_rate_delta = candidate.pass_rate - baseline.pass_rate
    mean_score_delta = candidate.mean_score - baseline.mean_score
    pass_rate_drop = -pass_rate_delta
    if pass_rate_drop > pass_rate_tolerance and not math.isclose(
        pass_rate_drop, pass_rate_tolerance, abs_tol=1e-12
    ):
        regressions.append(
            CaseChange(
                case_id="__aggregate_pass_rate__",
                kind="score_drop",
                baseline_score=baseline.pass_rate,
                candidate_score=candidate.pass_rate,
                reason=(
                    f"pass rate dropped by {-pass_rate_delta:.4f} "
                    f"(allowed {pass_rate_tolerance:.4f})"
                ),
            )
        )
    mean_score_drop = -mean_score_delta
    if mean_score_drop > mean_score_tolerance and not math.isclose(
        mean_score_drop, mean_score_tolerance, abs_tol=1e-12
    ):
        regressions.append(
            CaseChange(
                case_id="__aggregate_mean_score__",
                kind="score_drop",
                baseline_score=baseline.mean_score,
                candidate_score=candidate.mean_score,
                reason=(
                    f"mean score dropped by {-mean_score_delta:.4f} "
                    f"(allowed {mean_score_tolerance:.4f})"
                ),
            )
        )

    return ReportComparison(
        baseline_name=baseline.name,
        candidate_name=candidate.name,
        passed=not regressions,
        pass_rate_delta=pass_rate_delta,
        mean_score_delta=mean_score_delta,
        regressions=tuple(regressions),
        improvements=tuple(improvements),
        unchanged_cases=unchanged,
    )
