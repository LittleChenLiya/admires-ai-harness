"""Concurrent, deterministic benchmark runner with machine-readable reports."""

from __future__ import annotations

import asyncio
import platform
import statistics
import time
from collections.abc import Awaitable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from admires_harness.contracts import JsonValue, ModelUsage, utc_now
from admires_harness.evaluation.dataset import EvalCase
from admires_harness.evaluation.scorers import Score, Scorer, weighted_mean


class EvalOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = ""
    tool_calls: tuple[str, ...] = ()
    usage: ModelUsage = Field(default_factory=ModelUsage)
    latency_ms: float = Field(default=0, ge=0, allow_inf_nan=False)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class CaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    passed: bool
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    output: EvalOutput | None = None
    scores: tuple[Score, ...] = ()
    error: str | None = None


class BenchmarkReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    started_at: datetime
    ended_at: datetime
    threshold: float
    passed: bool
    pass_rate: float = Field(ge=0, le=1, allow_inf_nan=False)
    mean_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    p50_latency_ms: float = Field(ge=0, allow_inf_nan=False)
    p95_latency_ms: float = Field(ge=0, allow_inf_nan=False)
    total_tokens: int = Field(ge=0)
    total_cost_usd: float = Field(ge=0, allow_inf_nan=False)
    results: tuple[CaseResult, ...]
    environment: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def case_ids_are_unique(self) -> BenchmarkReport:
        case_ids = [result.case_id for result in self.results]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("benchmark result case IDs must be unique")
        return self

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.model_dump_json(indent=2), encoding="utf-8")


class EvalTarget(Protocol):
    def __call__(self, case: EvalCase) -> Awaitable[EvalOutput | str]: ...


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.5)))
    return ordered[index]


class EvalRunner:
    """Run cases with bounded concurrency and score every completed output."""

    def __init__(
        self,
        target: EvalTarget,
        scorers: Sequence[Scorer],
        *,
        concurrency: int = 4,
        case_timeout_seconds: float = 120.0,
        pass_threshold: float = 0.8,
        fail_fast: bool = False,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        if case_timeout_seconds <= 0:
            raise ValueError("case_timeout_seconds must be positive")
        if not 0 <= pass_threshold <= 1:
            raise ValueError("pass_threshold must be between 0 and 1")
        if not scorers:
            raise ValueError("at least one scorer is required")
        self.target = target
        self.scorers = tuple(scorers)
        self.concurrency = concurrency
        self.case_timeout_seconds = case_timeout_seconds
        self.pass_threshold = pass_threshold
        self.fail_fast = fail_fast

    async def _run_case(self, case: EvalCase, semaphore: asyncio.Semaphore) -> CaseResult:
        async with semaphore:
            started = time.perf_counter()
            try:
                async with asyncio.timeout(self.case_timeout_seconds):
                    raw_output = await self.target(case)
                latency = (time.perf_counter() - started) * 1000
                output = (
                    EvalOutput(text=raw_output, latency_ms=latency)
                    if isinstance(raw_output, str)
                    else raw_output.model_copy(
                        update={"latency_ms": raw_output.latency_ms or latency}
                    )
                )
                scores = tuple(scorer.score(case, output) for scorer in self.scorers)
                value = weighted_mean(scores, self.scorers)
                return CaseResult(
                    case_id=case.id,
                    passed=value >= self.pass_threshold and all(score.passed for score in scores),
                    score=value,
                    output=output,
                    scores=scores,
                )
            except TimeoutError:
                return CaseResult(
                    case_id=case.id, passed=False, score=0, error="evaluation case timed out"
                )
            except Exception as exc:
                return CaseResult(
                    case_id=case.id,
                    passed=False,
                    score=0,
                    error=f"{type(exc).__name__}: {exc}",
                )

    async def run(
        self,
        cases: Sequence[EvalCase],
        *,
        name: str = "benchmark",
        environment: Mapping[str, str] | None = None,
    ) -> BenchmarkReport:
        case_ids = [case.id for case in cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("evaluation case IDs must be unique")
        started_at = utc_now()
        semaphore = asyncio.Semaphore(self.concurrency)
        results: list[CaseResult] = []
        if self.fail_fast:
            for case in cases:
                result = await self._run_case(case, semaphore)
                results.append(result)
                if not result.passed:
                    break
        else:
            results = list(
                await asyncio.gather(*(self._run_case(case, semaphore) for case in cases))
            )
        outputs = [result.output for result in results if result.output is not None]
        latencies = [output.latency_ms for output in outputs]
        pass_rate = sum(result.passed for result in results) / len(results) if results else 0.0
        mean_score = statistics.fmean(result.score for result in results) if results else 0.0
        default_environment = {
            "python": platform.python_version(),
            "platform": platform.system(),
        }
        return BenchmarkReport(
            name=name,
            started_at=started_at,
            ended_at=utc_now(),
            threshold=self.pass_threshold,
            passed=bool(results) and all(result.passed for result in results),
            pass_rate=pass_rate,
            mean_score=mean_score,
            p50_latency_ms=_percentile(latencies, 0.50),
            p95_latency_ms=_percentile(latencies, 0.95),
            total_tokens=sum(output.usage.total_tokens for output in outputs),
            total_cost_usd=sum(output.usage.cost_usd for output in outputs),
            results=tuple(results),
            environment={**default_environment, **dict(environment or {})},
        )
