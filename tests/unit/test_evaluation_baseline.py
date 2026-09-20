from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from admires_harness.contracts import ModelUsage
from admires_harness.evaluation import (
    BenchmarkReport,
    CaseResult,
    EvalOutput,
    EvaluationRegressionError,
    ReportComparison,
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


def test_compare_reports_rejects_duplicate_case_ids() -> None:
    baseline = report("baseline", [case("one", 1)]).model_copy(
        update={"results": (case("dup", 1), case("dup", 0.5))}
    )
    candidate = report("candidate", [case("one", 1)])

    with pytest.raises(ValueError, match="duplicate case ID 'dup'"):
        compare_reports(candidate, baseline)

    candidate = candidate.model_copy(update={"results": (case("dup", 1), case("dup", 1))})
    with pytest.raises(ValueError, match="report 'candidate' contains duplicate case ID"):
        compare_reports(candidate, report("baseline", [case("one", 1)]))


def test_compare_reports_handles_empty_reports() -> None:
    empty_baseline = report("baseline", [], pass_rate=0.0, mean_score=0.0)
    empty_candidate = report("candidate", [], pass_rate=0.0, mean_score=0.0)

    comparison = compare_reports(empty_candidate, empty_baseline)

    assert comparison.passed
    assert comparison.pass_rate_delta == 0
    assert comparison.mean_score_delta == 0
    assert comparison.regressions == ()
    assert comparison.improvements == ()
    assert comparison.unchanged_cases == 0
    comparison.raise_for_regressions()

    grown = compare_reports(report("candidate", [case("new", 1)]), empty_baseline)
    assert grown.passed
    assert [(change.case_id, change.kind) for change in grown.improvements] == [("new", "new_case")]

    shrunk = compare_reports(empty_candidate, report("baseline", [case("gone", 1)]))
    assert not shrunk.passed
    kinds = {(change.case_id, change.kind) for change in shrunk.regressions}
    assert ("gone", "missing") in kinds


def test_compare_reports_marks_recovered_case_as_improvement() -> None:
    baseline = report("baseline", [case("flaky", 0.2, passed=False)])
    candidate = report("candidate", [case("flaky", 1)])

    comparison = compare_reports(candidate, baseline)

    assert comparison.passed
    assert comparison.regressions == ()
    (improvement,) = comparison.improvements
    assert improvement.case_id == "flaky"
    assert improvement.kind == "improvement"
    assert improvement.baseline_score == 0.2
    assert improvement.candidate_score == 1
    assert "now passes" in improvement.reason


def test_score_drop_tolerance_boundary_is_inclusive() -> None:
    baseline = report("baseline", [case("one", 0.8)])
    at_boundary = report("candidate", [case("one", 0.7)])
    just_over = report("candidate", [case("one", 0.69)])

    assert compare_reports(
        at_boundary, baseline, case_score_tolerance=0.1, mean_score_tolerance=1.0
    ).passed

    comparison = compare_reports(
        just_over, baseline, case_score_tolerance=0.1, mean_score_tolerance=1.0
    )
    assert not comparison.passed
    (regression,) = comparison.regressions
    assert regression.case_id == "one"
    assert regression.kind == "score_drop"
    assert regression.baseline_score == 0.8
    assert regression.candidate_score == 0.69
    assert "allowed 0.1000" in regression.reason


def test_aggregate_tolerances_boundary_is_inclusive() -> None:
    baseline = report("baseline", [case("one", 1)], pass_rate=1.0, mean_score=1.0)
    candidate = report("candidate", [case("one", 0.9)], pass_rate=0.9, mean_score=0.9)

    comparison = compare_reports(
        candidate,
        baseline,
        case_score_tolerance=0.1,
        pass_rate_tolerance=0.1,
        mean_score_tolerance=0.1,
    )

    assert comparison.passed
    assert comparison.pass_rate_delta == pytest.approx(-0.1)
    assert comparison.mean_score_delta == pytest.approx(-0.1)

    failing = compare_reports(candidate, baseline, case_score_tolerance=0.1)
    assert not failing.passed
    kinds = {(change.case_id, change.kind) for change in failing.regressions}
    assert ("__aggregate_pass_rate__", "score_drop") in kinds
    assert ("__aggregate_mean_score__", "score_drop") in kinds


def test_single_case_regression_is_not_hidden_by_aggregate_gain() -> None:
    baseline = report(
        "baseline",
        [case("weak", 0.4), case("strong", 0.4)],
        pass_rate=1.0,
        mean_score=0.4,
    )
    candidate = report(
        "candidate",
        [case("weak", 0.9), case("strong", 0.1)],
        pass_rate=1.0,
        mean_score=0.5,
    )

    comparison = compare_reports(candidate, baseline)

    assert comparison.mean_score_delta > 0
    assert not comparison.passed
    assert [(change.case_id, change.kind) for change in comparison.regressions] == [
        ("strong", "score_drop")
    ]
    with pytest.raises(EvaluationRegressionError, match="strong: score_drop"):
        comparison.raise_for_regressions()


def test_nan_and_infinity_are_rejected_everywhere() -> None:
    baseline = report("baseline", [case("one", 1)])
    candidate = report("candidate", [case("one", 1)])

    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite value between 0 and 1"):
            compare_reports(candidate, baseline, case_score_tolerance=bad)
        with pytest.raises(ValueError, match="finite value between 0 and 1"):
            compare_reports(candidate, baseline, mean_score_tolerance=bad)
        with pytest.raises(ValueError, match="finite value between 0 and 1"):
            compare_reports(candidate, baseline, pass_rate_tolerance=bad)

    with pytest.raises(ValidationError):
        case("nan", math.nan)
    with pytest.raises(ValidationError):
        report("inf", [case("one", 1)], pass_rate=math.inf)
    with pytest.raises(ValidationError):
        EvalOutput(latency_ms=math.inf)
    with pytest.raises(ValidationError):
        BenchmarkReport.model_validate_json(
            report("x", [case("one", 1)])
            .model_dump_json()
            .replace('"pass_rate":1.0', '"pass_rate":NaN')
        )


def test_json_round_trip_preserves_errors_latency_tokens_and_cost(tmp_path: Path) -> None:
    original = report(
        "candidate",
        [
            CaseResult(
                case_id="failed",
                passed=False,
                score=0,
                error="RuntimeError: boom",
                output=EvalOutput(
                    text="partial",
                    latency_ms=12.5,
                    usage=ModelUsage(input_tokens=3, output_tokens=4, cost_usd=0.25),
                ),
            ),
            CaseResult(
                case_id="ok",
                passed=True,
                score=1,
                output=EvalOutput(
                    text="done",
                    latency_ms=7.5,
                    usage=ModelUsage(input_tokens=1, output_tokens=2, cost_usd=0.75),
                ),
            ),
        ],
    ).model_copy(update={"total_tokens": 10, "total_cost_usd": 1.0})
    path = tmp_path / "report.json"
    original.write_json(path)

    loaded = load_report(path)

    assert loaded == original
    failed = loaded.results[0]
    assert failed.error == "RuntimeError: boom"
    assert failed.output is not None
    assert failed.output.latency_ms == 12.5
    assert failed.output.usage.total_tokens == 7
    assert failed.output.usage.cost_usd == 0.25
    assert loaded.total_tokens == 10
    assert loaded.total_cost_usd == 1.0


def test_comparison_round_trip_and_error_message(tmp_path: Path) -> None:
    baseline = report("baseline", [case("failed", 1), case("missing", 1)])
    candidate = report("candidate", [case("failed", 0, passed=False)])
    comparison = compare_reports(candidate, baseline)
    path = tmp_path / "comparison.json"
    comparison.write_json(path)

    loaded = ReportComparison.model_validate_json(path.read_text(encoding="utf-8"))
    assert loaded == comparison
    assert [(change.case_id, change.kind) for change in loaded.regressions] == [
        ("failed", "new_failure"),
        ("missing", "missing"),
        ("__aggregate_pass_rate__", "score_drop"),
        ("__aggregate_mean_score__", "score_drop"),
    ]

    with pytest.raises(EvaluationRegressionError) as excinfo:
        loaded.raise_for_regressions()
    message = str(excinfo.value)
    assert "benchmark regressed against 'baseline'" in message
    assert "failed: new_failure" in message
    assert "missing: missing" in message
