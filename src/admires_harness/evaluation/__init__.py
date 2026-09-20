"""Offline-first evaluation and benchmark primitives."""

from admires_harness.evaluation.baseline import (
    CaseChange,
    EvaluationRegressionError,
    ReportComparison,
    compare_reports,
    load_report,
)
from admires_harness.evaluation.dataset import (
    DatasetError,
    EvalCase,
    load_jsonl,
    write_jsonl,
)
from admires_harness.evaluation.runner import (
    BenchmarkReport,
    CaseResult,
    EvalOutput,
    EvalRunner,
)
from admires_harness.evaluation.scorers import (
    ContainsScorer,
    ExactMatchScorer,
    JsonValidityScorer,
    LatencyScorer,
    Score,
    ToolSequenceScorer,
)

__all__ = [
    "BenchmarkReport",
    "CaseChange",
    "CaseResult",
    "ContainsScorer",
    "DatasetError",
    "EvalCase",
    "EvalOutput",
    "EvalRunner",
    "EvaluationRegressionError",
    "ExactMatchScorer",
    "JsonValidityScorer",
    "LatencyScorer",
    "ReportComparison",
    "Score",
    "ToolSequenceScorer",
    "compare_reports",
    "load_jsonl",
    "load_report",
    "write_jsonl",
]
