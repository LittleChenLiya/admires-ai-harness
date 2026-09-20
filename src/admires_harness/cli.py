"""Command-line interface for local runs, server startup, and evaluations.

Output contract shared by every command:

* Machine-readable JSON goes to stdout only, one document per invocation.
* Human-readable panels, tables, and diagnostics go to stderr in ``--json``
  mode so long output never corrupts the JSON document.
* Exit codes: ``0`` success, ``1`` business failure (failed run, failed
  benchmark, baseline regression, unwritable report), ``2`` usage errors
  (unknown workflow, invalid paths or option values).
* Sensitive settings (API keys, tokens, secrets) are never printed.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Annotated, NoReturn, cast

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from admires_harness.agents import list_demos
from admires_harness.api import create_app
from admires_harness.config import Settings
from admires_harness.contracts import EventType, JsonValue, ModelUsage
from admires_harness.evaluation import (
    ContainsScorer,
    EvalCase,
    EvalOutput,
    EvalRunner,
    ToolSequenceScorer,
    load_jsonl,
)
from admires_harness.evaluation.baseline import compare_reports, load_report
from admires_harness.evaluation.dataset import DatasetError
from admires_harness.services.runs import RunManager, as_camel

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2

app = typer.Typer(
    name="admire",
    help="Run, observe, and evaluate AI agents with safe offline defaults.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)

_SENSITIVE_KEY = re.compile(
    r"api[_-]?key|secret|passw(?:or)?d|credential|authorization|"
    r"access[_-]?token|refresh[_-]?token|private[_-]?key|bearer",
    re.IGNORECASE,
)
REDACTED = "***redacted***"


def redact_sensitive(value: JsonValue) -> JsonValue:
    """Recursively redact secret-looking mapping keys without touching metrics."""

    if isinstance(value, dict):
        return {
            str(key): REDACTED if _SENSITIVE_KEY.search(str(key)) else redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_sensitive(item) for item in value]
    return value


def _emit_json(payload: JsonValue) -> None:
    """Write exactly one unwrapped UTF-8 JSON document to stdout."""

    sys.stdout.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def _fail(message: str, *, code: int = EXIT_FAILED) -> NoReturn:
    err_console.print(f"[red]error:[/red] {message}", soft_wrap=True)
    raise typer.Exit(code=code)


def _usage_error(message: str) -> NoReturn:
    _fail(message, code=EXIT_USAGE)


def _settings(data_dir: Path | None) -> Settings:
    settings = Settings()
    if data_dir is not None:
        if data_dir.exists() and not data_dir.is_dir():
            _usage_error(f"--data-dir must be a directory: {data_dir}")
        settings = settings.model_copy(
            update={
                "database_path": data_dir / "harness.db",
                "workspace_root": data_dir / "workspaces",
            }
        )
    return settings


def _report_environment(settings: Settings) -> dict[str, str]:
    """Whitelisted, non-sensitive environment metadata for benchmark reports."""

    return {
        "environment": settings.environment,
        "model_provider": settings.model_provider,
        "model_name": settings.model_name,
    }


@app.command("list")
def list_workflows(
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit a machine-readable workflow list")
    ] = False,
) -> None:
    """List the built-in reference workflows."""

    demos = list(list_demos())
    if json_output:
        _emit_json(
            {
                "workflows": [
                    {
                        "id": demo.id,
                        "title": demo.title,
                        "agents": list(demo.agents),
                        "capabilities": list(demo.capabilities),
                    }
                    for demo in demos
                ]
            }
        )
        return
    table = Table(title="Reference workflows")
    table.add_column("ID", style="cyan")
    table.add_column("Agents")
    table.add_column("Capabilities")
    for demo in demos:
        table.add_row(demo.id, ", ".join(demo.agents), ", ".join(demo.capabilities))
    console.print(table)


@app.command()
def demo(
    workflow: Annotated[
        str, typer.Argument(help="research, coding, collaboration, or all")
    ] = "all",
    task: Annotated[str | None, typer.Option("--task", help="Override task input")] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit a machine-readable report")
    ] = False,
    data_dir: Annotated[Path | None, typer.Option(help="Override the local data directory")] = None,
) -> None:
    """Run one or all deterministic offline demos."""

    asyncio.run(_demo(workflow, task, json_output, data_dir))


async def _demo(workflow: str, task: str | None, json_output: bool, data_dir: Path | None) -> None:
    settings = _settings(data_dir)
    manager = RunManager(settings)
    definitions = list(list_demos())
    if workflow != "all":
        definitions = [demo for demo in definitions if demo.id == workflow]
        if not definitions:
            raise typer.BadParameter("choose research, coding, collaboration, or all")
    reports = []
    for definition in definitions:
        view = await manager.run_sync(
            definition.id, task or definition.default_input, auto_approve=True
        )
        report = as_camel(view)
        reports.append(report)
        if not json_output:
            console.print(
                Panel.fit(
                    str(view.output or view.error),
                    title=f"{definition.title} 路 {view.status.value}",
                    subtitle=(
                        f"{view.metrics.steps} steps 路 "
                        f"{view.metrics.input_tokens + view.metrics.output_tokens} tokens 路 "
                        f"${view.metrics.cost_usd:.4f}"
                    ),
                    border_style="green" if view.status.value == "succeeded" else "red",
                )
            )
    if json_output:
        _emit_json(redact_sensitive({"runs": cast("list[JsonValue]", reports)}))
    if any(report["status"] != "succeeded" for report in reports):
        failed = ", ".join(
            f"{report['demo']}={report['status']}"
            for report in reports
            if report["status"] != "succeeded"
        )
        _fail(f"demo run did not succeed: {failed}")


@app.command()
def evaluate(
    dataset: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)] = Path(
        "evaluation/reference.jsonl"
    ),
    report: Annotated[Path | None, typer.Option(help="Write the JSON report to this path")] = None,
    concurrency: Annotated[int, typer.Option(min=1, max=32)] = 3,
    data_dir: Annotated[Path | None, typer.Option(help="Override the local data directory")] = None,
    baseline: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, readable=True)
    ] = None,
    comparison: Annotated[
        Path | None, typer.Option(help="Write the baseline comparison JSON to this path")
    ] = None,
) -> None:
    """Run deterministic reference-workflow evaluations from JSONL."""

    asyncio.run(_evaluate(dataset, report, concurrency, data_dir, baseline, comparison))


async def _evaluate(
    dataset: Path,
    report_path: Path | None,
    concurrency: int,
    data_dir: Path | None,
    baseline_path: Path | None,
    comparison_path: Path | None,
) -> None:
    settings = _settings(data_dir)
    manager = RunManager(settings)

    async def target(case: EvalCase) -> EvalOutput:
        raw_demo = case.metadata.get("demo")
        if not isinstance(raw_demo, str):
            raise ValueError(f"case {case.id!r} requires metadata.demo")
        view = await manager.run_sync(raw_demo, case.input, auto_approve=True)
        events = await manager.list_events(view.run_id)
        tool_calls = tuple(
            str(event.payload["tool"])
            for event in events
            if event.type is EventType.TOOL_REQUESTED and isinstance(event.payload.get("tool"), str)
        )
        return EvalOutput(
            text=str(view.output or view.error or ""),
            tool_calls=tool_calls,
            usage=ModelUsage(
                input_tokens=view.metrics.input_tokens,
                output_tokens=view.metrics.output_tokens,
                cost_usd=view.metrics.cost_usd,
            ),
            metadata={
                "run_id": str(view.run_id),
                "status": view.status.value,
            },
        )

    try:
        cases = load_jsonl(dataset)
    except DatasetError as exc:
        _usage_error(str(exc))
    scorers = (ContainsScorer(weight=2), ToolSequenceScorer(exact=True, weight=1))
    benchmark = await EvalRunner(target, scorers, concurrency=concurrency, pass_threshold=1.0).run(
        cases, name=dataset.stem, environment=_report_environment(settings)
    )
    _emit_json(json.loads(benchmark.model_dump_json()))
    if report_path is not None:
        try:
            benchmark.write_json(report_path)
        except OSError as exc:
            _fail(f"could not write report to {report_path}: {exc}")
    if baseline_path is not None:
        try:
            baseline_report = load_report(baseline_path)
        except Exception as exc:
            _usage_error(f"invalid baseline report {baseline_path}: {exc}")
        comparison_result = compare_reports(benchmark, baseline_report)
        if comparison_path is not None:
            try:
                comparison_result.write_json(comparison_path)
            except OSError as exc:
                _fail(f"could not write comparison to {comparison_path}: {exc}")
        if not comparison_result.passed:
            details = "; ".join(
                f"{change.case_id}: {change.kind} ({change.reason})"
                for change in comparison_result.regressions
            )
            _fail(f"benchmark regressed against baseline {baseline_report.name!r}: {details}")
    if not benchmark.passed:
        failed = ", ".join(result.case_id for result in benchmark.results if not result.passed)
        _fail(f"benchmark {benchmark.name!r} failed; failing cases: {failed}")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8000,
    reload: Annotated[bool, typer.Option(help="Reload Python files during development")] = False,
) -> None:
    """Serve the API and the built Web console."""

    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, reload=reload)


_DEV_COMMANDS = ("uv run admire serve --reload", "pnpm.cmd --dir web dev")


@app.command()
def dev(
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit a machine-readable summary")
    ] = False,
) -> None:
    """Explain the two-process local development command."""

    if json_output:
        _emit_json(
            {
                "commands": list(_DEV_COMMANDS),
                "urls": {
                    "web": "http://localhost:5173",
                    "apiDocs": "http://localhost:8000/docs",
                },
            }
        )
        return
    console.print(
        "Run [cyan]uv run admire serve --reload[/cyan] and "
        "[cyan]pnpm.cmd --dir web dev[/cyan] in separate terminals."
    )


if __name__ == "__main__":
    app()
