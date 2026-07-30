from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from admires_harness.contracts import ModelUsage
from admires_harness.evaluation.dataset import (
    DatasetError,
    EvalCase,
    load_jsonl,
    write_jsonl,
)
from admires_harness.evaluation.runner import EvalOutput, EvalRunner
from admires_harness.evaluation.scorers import (
    ContainsScorer,
    ExactMatchScorer,
    JsonValidityScorer,
    LatencyScorer,
    ToolSequenceScorer,
)


def test_jsonl_dataset_round_trip_and_duplicate_detection(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    cases = (
        EvalCase(id="one", input="say hello", expected="hello", tags={"smoke"}),
        EvalCase(id="two", input="use a tool", expected={"tools": ["search"]}),
    )
    write_jsonl(cases, path)

    assert load_jsonl(path) == cases

    path.write_text(
        '{"id":"same","input":"a"}\n{"id":"same","input":"b"}\n',
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="duplicate case id"):
        load_jsonl(path)


def test_jsonl_dataset_reports_source_line(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text("# comment\nnot-json\n", encoding="utf-8")

    with pytest.raises(DatasetError, match=r"invalid.jsonl:2"):
        load_jsonl(path)


def test_built_in_scorers_cover_text_json_tools_and_latency() -> None:
    text_case = EvalCase(id="text", input="x", expected="Hello")
    text_output = EvalOutput(text="hello", latency_ms=5)
    assert ExactMatchScorer().score(text_case, text_output).passed
    assert ContainsScorer().score(text_case, EvalOutput(text="Say HELLO!")).passed
    assert JsonValidityScorer().score(text_case, EvalOutput(text='{"ok": true}')).passed
    assert not JsonValidityScorer().score(text_case, EvalOutput(text="nope")).passed

    tool_case = EvalCase(id="tool", input="x", expected={"tools": ["search", "read"]})
    assert ToolSequenceScorer().score(tool_case, EvalOutput(tool_calls=("search", "read"))).passed
    assert (
        ToolSequenceScorer(exact=False)
        .score(tool_case, EvalOutput(tool_calls=("search", "log", "read")))
        .passed
    )
    assert not LatencyScorer(10).score(text_case, EvalOutput(latency_ms=11)).passed


@pytest.mark.asyncio
async def test_eval_runner_aggregates_quality_cost_and_latency(tmp_path: Path) -> None:
    active = 0
    max_active = 0

    async def target(case: EvalCase) -> EvalOutput:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.005)
        active -= 1
        return EvalOutput(
            text=str(case.expected),
            usage=ModelUsage(input_tokens=2, output_tokens=1, cost_usd=0.001),
        )

    runner = EvalRunner(target, [ExactMatchScorer()], concurrency=2)
    report = await runner.run(
        [EvalCase(id=str(index), input="echo", expected=f"value-{index}") for index in range(4)],
        name="offline-smoke",
        environment={"model": "mock"},
    )

    assert report.passed
    assert report.pass_rate == 1
    assert report.mean_score == 1
    assert report.total_tokens == 12
    assert report.total_cost_usd == pytest.approx(0.004)
    assert max_active == 2
    assert report.environment["model"] == "mock"

    report_path = tmp_path / "report.json"
    report.write_json(report_path)
    assert json.loads(report_path.read_text(encoding="utf-8"))["name"] == "offline-smoke"


@pytest.mark.asyncio
async def test_eval_runner_records_errors_timeouts_and_fail_fast() -> None:
    async def target(case: EvalCase) -> str:
        if case.id == "error":
            raise RuntimeError("boom")
        await asyncio.sleep(0.02)
        return "late"

    cases = [
        EvalCase(id="error", input="x", expected="x"),
        EvalCase(id="timeout", input="x", expected="x"),
    ]
    report = await EvalRunner(target, [ExactMatchScorer()], case_timeout_seconds=0.001).run(cases)
    assert report.results[0].error == "RuntimeError: boom"
    assert report.results[1].error == "evaluation case timed out"
    assert not report.passed

    fail_fast = await EvalRunner(target, [ExactMatchScorer()], fail_fast=True).run(cases)
    assert len(fail_fast.results) == 1


@pytest.mark.asyncio
async def test_eval_runner_rejects_duplicate_case_ids() -> None:
    async def target(case: EvalCase) -> str:
        return case.input

    runner = EvalRunner(target, [ExactMatchScorer()])
    duplicates = [
        EvalCase(id="same", input="a", expected="a"),
        EvalCase(id="same", input="b", expected="b"),
    ]

    with pytest.raises(ValueError, match="case IDs must be unique"):
        await runner.run(duplicates)
