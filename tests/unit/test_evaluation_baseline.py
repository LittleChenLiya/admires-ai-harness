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
    if pass_rate is None:
        pass_rate = sum(result.passed for result in results) / len(results) if results else 0.0
    if mean_score is None:
        mean_score = sum(result.score for result in results) / len(results) if results else 0.0
    return BenchmarkReport(
        name=name,
        started_at=NOW,
        ended_at=NOW,
        threshold=0.8,
        passed=all(result.passed for result in results),
        pass_rate=pass_rate,
        mean_score=mean_score,
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


def test_compare_reports_rejects_duplicate_case_ids_defensively() -> None:
    duplicated = report("baseline", [case("one", 1)]).model_copy(
        update={"results": (case("dup", 1), case("dup", 0.5))}
    )
    candidate = report("candidate", [case("dup", 1)])

    with pytest.raises(ValueError, match="duplicate case IDs: dup"):
        compare_reports(candidate, duplicated)
    with pytest.raises(ValueError, match="duplicate case IDs: dup"):
        compare_reports(duplicated, candidate)


def test_compare_reports_handles_empty_reports() -> None:
    comparison = compare_reports(report("empty-candidate", []), report("empty-baseline", []))

    assert comparison.passed
    assert comparison.pass_rate_delta == 0
    assert comparison.mean_score_delta == 0
    assert comparison.regressions == ()
    assert comparison.improvements == ()
    assert comparison.unchanged_cases == 0
    comparison.raise_for_regressions()

    only_baseline = compare_reports(report("empty", []), report("baseline", [case("gone", 1)]))
    kinds = {(change.case_id, change.kind) for change in only_baseline.regressions}
    assert ("gone", "missing") in kinds
    assert ("__aggregate_pass_rate__", "score_drop") in kinds
    assert ("__aggregate_mean_score__", "score_drop") in kinds
    only_candidate = compare_reports(report("candidate", [case("added", 1)]), report("empty", []))
    assert [(change.case_id, change.kind) for change in only_candidate.improvements] == [
        ("added", "new_case")
    ]


def test_compare_reports_distinguishes_change_kinds_and_reasons() -> None:
    baseline = report(
        "baseline",
        [
            case("now-failing", 1),
            case("dropped", 1),
            case("recovered", 0.2, passed=False),
            case("missing", 1),
            case("stable", 0.9),
        ],
    )
    candidate = report(
        "candidate",
        [
            case("now-failing", 0.1, passed=False),
            case("dropped", 0.5),
            case("recovered", 1),
            case("stable", 0.9),
            case("brand-new", 1),
        ],
    )

    comparison = compare_reports(candidate, baseline, allow_missing_cases=False)

    regressions = {change.case_id: change for change in comparison.regressions}
    assert regressions["now-failing"].kind == "new_failure"
    assert regressions["now-failing"].reason == "a previously passing case now fails"
    assert regressions["now-failing"].baseline_score == 1
    assert regressions["now-failing"].candidate_score == 0.1
    assert regressions["dropped"].kind == "score_drop"
    assert "dropped by 0.5000" in regressions["dropped"].reason
    assert regressions["missing"].kind == "missing"
    assert regressions["missing"].candidate_score is None
    assert "omitted a baseline case" in regressions["missing"].reason

    improvements = {change.case_id: change for change in comparison.improvements}
    assert improvements["recovered"].kind == "improvement"
    assert improvements["recovered"].reason == "a previously failing case now passes"
    assert improvements["brand-new"].kind == "new_case"
    assert improvements["brand-new"].baseline_score is None
    assert improvements["brand-new"].candidate_score == 1
    assert "adds evaluation coverage" in improvements["brand-new"].reason
    assert comparison.unchanged_cases == 1


def test_new_failure_reason_prefers_the_recorded_error() -> None:
    baseline = report("baseline", [case("flaky", 1)])
    failing = CaseResult(case_id="flaky", passed=False, score=0, error="TimeoutError: slow")
    candidate = report("candidate", [failing])

    comparison = compare_reports(candidate, baseline, pass_rate_tolerance=1)

    regression = next(change for change in comparison.regressions if change.case_id == "flaky")
    assert regression.kind == "new_failure"
    assert regression.reason == "TimeoutError: slow"


def test_case_score_tolerance_boundary_is_inclusive() -> None:
    baseline = report("baseline", [case("one", 1)])

    at_boundary = compare_reports(
        report("candidate", [case("one", 0.5)]),
        baseline,
        case_score_tolerance=0.5,
        mean_score_tolerance=0.5,
    )
    assert at_boundary.passed
    assert at_boundary.unchanged_cases == 1

    beyond = compare_reports(
        report("candidate", [case("one", 0.25)]),
        baseline,
        case_score_tolerance=0.5,
        mean_score_tolerance=1,
    )
    assert [(change.case_id, change.kind) for change in beyond.regressions] == [
        ("one", "score_drop")
    ]


def test_aggregate_tolerances_never_hide_per_case_regressions() -> None:
    baseline = report("baseline", [case("one", 1), case("two", 1)])
    candidate = report("candidate", [case("one", 1), case("two", 0.4, passed=False)])

    comparison = compare_reports(
        candidate,
        baseline,
        case_score_tolerance=1,
        mean_score_tolerance=1,
        pass_rate_tolerance=1,
    )

    assert not comparison.passed
    assert [(change.case_id, change.kind) for change in comparison.regressions] == [
        ("two", "new_failure")
    ]


def test_aggregate_pass_rate_and_mean_score_boundaries() -> None:
    baseline = report("baseline", [case("one", 1), case("two", 1)])
    candidate = report("candidate", [case("one", 1), case("two", 0.5, passed=False)])

    at_boundary = compare_reports(
        candidate,
        baseline,
        case_score_tolerance=1,
        pass_rate_tolerance=0.5,
        mean_score_tolerance=0.25,
    )
    assert [change.case_id for change in at_boundary.regressions] == ["two"]

    too_tight = compare_reports(
        candidate,
        baseline,
        case_score_tolerance=1,
        pass_rate_tolerance=0.4,
        mean_score_tolerance=0.2,
    )
    aggregate = {change.case_id for change in too_tight.regressions}
    assert "__aggregate_pass_rate__" in aggregate
    assert "__aggregate_mean_score__" in aggregate


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -0.1, 1.1])
def test_tolerances_reject_non_finite_and_out_of_range_values(bad: float) -> None:
    with pytest.raises(ValueError, match="case_score_tolerance must be between 0 and 1"):
        compare_reports(report("a", []), report("b", []), case_score_tolerance=bad)
    with pytest.raises(ValueError, match="mean_score_tolerance must be between 0 and 1"):
        compare_reports(report("a", []), report("b", []), mean_score_tolerance=bad)
    with pytest.raises(ValueError, match="pass_rate_tolerance must be between 0 and 1"):
        compare_reports(report("a", []), report("b", []), pass_rate_tolerance=bad)


def test_reports_reject_nan_and_infinity() -> None:
    with pytest.raises(ValidationError):
        CaseResult(case_id="bad", passed=True, score=math.nan)
    with pytest.raises(ValidationError):
        report("bad", [], pass_rate=math.inf)
    with pytest.raises(ValidationError):
        report("bad", [], mean_score=math.nan)
    with pytest.raises(ValidationError):
        EvalOutput(latency_ms=math.inf)
    with pytest.raises(ValidationError):
        ModelUsage(cost_usd=math.nan)


def test_raise_for_regressions_message_lists_every_regression() -> None:
    baseline = report("baseline-v1", [case("failed", 1), case("lower", 1)])
    candidate = report("candidate", [case("failed", 0, passed=False), case("lower", 0.5)])

    comparison = compare_reports(candidate, baseline)

    with pytest.raises(EvaluationRegressionError) as excinfo:
        comparison.raise_for_regressions()
    message = str(excinfo.value)
    assert "baseline-v1" in message
    assert "failed: new_failure" in message
    assert "lower: score_drop" in message


def test_report_round_trip_preserves_error_latency_tokens_and_cost(tmp_path: Path) -> None:
    output = EvalOutput(
        text="partial",
        tool_calls=("search", "read"),
        usage=ModelUsage(input_tokens=11, output_tokens=7, cost_usd=0.0025),
        latency_ms=123.5,
        metadata={"run_id": "run-1"},
    )
    results = [
        CaseResult(case_id="ok", passed=True, score=1, output=output),
        CaseResult(case_id="boom", passed=False, score=0, error="RuntimeError: exploded"),
    ]
    original = BenchmarkReport(
        name="candidate",
        started_at=NOW,
        ended_at=NOW,
        threshold=0.8,
        passed=False,
        pass_rate=0.5,
        mean_score=0.5,
        p50_latency_ms=123.5,
        p95_latency_ms=123.5,
        total_tokens=18,
        total_cost_usd=0.0025,
        results=tuple(results),
        environment={"python": "3.12"},
    )
    path = tmp_path / "report.json"

    original.write_json(path)
    loaded = load_report(path)

    assert loaded == original
    ok, boom = loaded.results
    assert ok.output is not None
    assert ok.output.latency_ms == 123.5
    assert ok.output.usage.total_tokens == 18
    assert ok.output.usage.cost_usd == 0.0025
    assert boom.error == "RuntimeError: exploded"
    assert loaded.total_tokens == 18
    assert loaded.total_cost_usd == 0.0025
