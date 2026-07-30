from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from admires_harness.evaluation import (
    BenchmarkReport,
    CaseResult,
    EvaluationRegressionError,
    compare_reports,
    load_report,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def report(
    name: str,
    results: list[CaseResult],
    *,
    pass_rate: float | None = None,
    mean_score: float | None = None,
) -> BenchmarkReport:
    return BenchmarkReport(
        name=name,
        started_at=NOW,
        ended_at=NOW,
        threshold=0.8,
        passed=all(result.passed for result in results),
        pass_rate=(
            sum(result.passed for result in results) / len(results)
            if pass_rate is None
            else pass_rate
        ),
        mean_score=(
            sum(result.score for result in results) / len(results)
            if mean_score is None
            else mean_score
        ),
        p50_latency_ms=1,
        p95_latency_ms=2,
        total_tokens=0,
        total_cost_usd=0,
        results=tuple(results),
    )


def case(case_id: str, score: float, *, passed: bool = True) -> CaseResult:
    return CaseResult(case_id=case_id, passed=passed, score=score)


def test_compare_reports_passes_unchanged_and_reports_improvements() -> None:
    baseline = report("baseline", [case("stable", 1), case("fixed", 0, passed=False)])
    candidate = report("candidate", [case("stable", 1), case("fixed", 1), case("new", 1)])

    comparison = compare_reports(candidate, baseline)

    assert comparison.passed
    assert comparison.unchanged_cases == 1
    assert [(change.case_id, change.kind) for change in comparison.improvements] == [
        ("fixed", "improvement"),
        ("new", "new_case"),
    ]
    comparison.raise_for_regressions()


def test_compare_reports_fails_closed_for_new_failure_missing_and_score_drop() -> None:
    baseline = report("baseline", [case("failed", 1), case("missing", 1), case("lower", 0.9)])
    candidate = report("candidate", [case("failed", 0, passed=False), case("lower", 0.7)])

    comparison = compare_reports(candidate, baseline, case_score_tolerance=0.1)

    assert not comparison.passed
    kinds = {(change.case_id, change.kind) for change in comparison.regressions}
    assert ("failed", "new_failure") in kinds
    assert ("missing", "missing") in kinds
    assert ("lower", "score_drop") in kinds
    assert ("__aggregate_pass_rate__", "score_drop") in kinds
    assert ("__aggregate_mean_score__", "score_drop") in kinds
    with pytest.raises(EvaluationRegressionError, match="failed: new_failure"):
        comparison.raise_for_regressions()


def test_compare_reports_honors_explicit_tolerances_and_missing_case_policy() -> None:
    baseline = report("baseline", [case("one", 1), case("optional", 1)])
    candidate = report("candidate", [case("one", 0.95)], pass_rate=0.95, mean_score=0.95)

    comparison = compare_reports(
        candidate,
        baseline,
        case_score_tolerance=0.05,
        pass_rate_tolerance=0.05,
        mean_score_tolerance=0.05,
        allow_missing_cases=True,
    )

    assert comparison.passed
    with pytest.raises(ValueError, match="between 0 and 1"):
        compare_reports(candidate, baseline, case_score_tolerance=-0.1)


def test_report_round_trip_and_duplicate_case_rejection(tmp_path: Path) -> None:
    original = report("candidate", [case("one", 1)])
    path = tmp_path / "report.json"
    original.write_json(path)

    assert load_report(path) == original

    with pytest.raises(ValidationError, match="case IDs must be unique"):
        report("bad", [case("same", 1), case("same", 1)])


def test_comparison_report_can_be_written(tmp_path: Path) -> None:
    baseline = report("baseline", [case("one", 1)])
    candidate = report("candidate", [case("one", 1)])
    comparison = compare_reports(candidate, baseline)
    path = tmp_path / "comparison.json"

    comparison.write_json(path)

    assert '"passed": true' in path.read_text(encoding="utf-8")
