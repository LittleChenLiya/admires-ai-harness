"""Command-line interface for local runs, server startup, and evaluations."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from admires_harness.agents import list_demos
from admires_harness.api import create_app
from admires_harness.config import Settings
from admires_harness.contracts import EventType, ModelUsage
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

app = typer.Typer(
    name="admire",
    help="Run, observe, and evaluate AI agents with safe offline defaults.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)

_SENSITIVE_SEGMENTS = frozenset(
    {"secret", "token", "password", "authorization", "credential", "credentials"}
)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_REDACTED = "***"


def _is_sensitive_key(key: object) -> bool:
    """Match whole key segments so metrics like input_tokens stay visible."""

    normalized = _CAMEL_BOUNDARY.sub("_", str(key)).lower().replace("-", "_").replace(" ", "_")
    if "api_key" in normalized or "apikey" in normalized:
        return True
    return any(segment in _SENSITIVE_SEGMENTS for segment in normalized.split("_"))


def _configure_utf8() -> None:
    """Keep JSON and error output readable on non-UTF-8 Windows consoles."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            continue


def _redact(value: Any) -> Any:
    """Mask sensitive mapping values so secrets never reach stdout or stderr."""

    if isinstance(value, dict):
        return {
            key: _REDACTED if _is_sensitive_key(key) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def _echo_json(payload: Any) -> None:
    """Write one machine-readable JSON document to stdout, and nothing else.

    A plain write keeps long outputs parseable: Rich may soft-wrap long lines
    to the console width, which would corrupt JSON string values.
    """

    sys.stdout.write(json.dumps(_redact(payload), ensure_ascii=False, default=str) + "\n")


def _fail(message: str) -> None:
    """Report a business failure on stderr and exit with code 1."""

    err_console.print(f"[bold red]error:[/bold red] {escape(message)}")
    raise typer.Exit(code=1)


@app.command("list")
def list_workflows(
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit a machine-readable workflow list")
    ] = False,
) -> None:
    """List the built-in reference workflows."""

    _configure_utf8()
    demos = list(list_demos())
    if json_output:
        _echo_json(
            {
                "workflows": [
                    {
                        "id": demo.id,
                        "title": demo.title,
                        "description": demo.description,
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

    _configure_utf8()
    asyncio.run(_demo(workflow, task, json_output, data_dir))


async def _demo(workflow: str, task: str | None, json_output: bool, data_dir: Path | None) -> None:
    settings = Settings()
    if data_dir is not None:
        settings = settings.model_copy(
            update={
                "database_path": data_dir / "harness.db",
                "workspace_root": data_dir / "workspaces",
            }
        )
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
                    title=f"{definition.title} · {view.status.value}",
                    subtitle=(
                        f"{view.metrics.steps} steps · "
                        f"{view.metrics.input_tokens + view.metrics.output_tokens} tokens · "
                        f"${view.metrics.cost_usd:.4f}"
                    ),
                    border_style="green" if view.status.value == "succeeded" else "red",
                )
            )
    if json_output:
        _echo_json({"runs": reports})
    if any(report["status"] != "succeeded" for report in reports):
        failed = [str(report["demo"]) for report in reports if report["status"] != "succeeded"]
        _fail(f"demo workflow(s) failed: {', '.join(failed)}")


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
) -> None:
    """Run deterministic reference-workflow evaluations from JSONL."""

    _configure_utf8()
    asyncio.run(_evaluate(dataset, report, concurrency, data_dir, baseline))


async def _evaluate(
    dataset: Path,
    report_path: Path | None,
    concurrency: int,
    data_dir: Path | None,
    baseline_path: Path | None,
) -> None:
    settings = Settings()
    if data_dir is not None:
        settings = settings.model_copy(
            update={
                "database_path": data_dir / "harness.db",
                "workspace_root": data_dir / "workspaces",
            }
        )
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
    except (DatasetError, OSError) as exc:
        raise typer.BadParameter(f"invalid dataset {dataset}: {exc}") from exc
    scorers = (ContainsScorer(weight=2), ToolSequenceScorer(exact=True, weight=1))
    benchmark = await EvalRunner(target, scorers, concurrency=concurrency, pass_threshold=1.0).run(
        cases, name=dataset.stem, environment={"model_provider": settings.model_provider}
    )
    write_error: str | None = None
    if report_path is not None:
        try:
            benchmark.write_json(report_path)
        except OSError as exc:
            write_error = f"could not write report to {report_path}: {exc}"
    comparison = None
    if baseline_path is not None:
        try:
            baseline_report = load_report(baseline_path)
        except (OSError, ValueError) as exc:
            raise typer.BadParameter(f"invalid baseline report {baseline_path}: {exc}") from exc
        comparison = compare_reports(benchmark, baseline_report)
    payload: dict[str, Any] = json.loads(benchmark.model_dump_json())
    if comparison is not None:
        payload["baseline"] = json.loads(comparison.model_dump_json())
    _echo_json(payload)
    if write_error is not None:
        _fail(write_error)
    if comparison is not None and not comparison.passed:
        details = "; ".join(f"{change.case_id}: {change.kind}" for change in comparison.regressions)
        _fail(f"benchmark regressed against baseline {comparison.baseline_name!r}: {details}")
    if not benchmark.passed:
        failed = sum(not result.passed for result in benchmark.results)
        _fail(
            f"benchmark {benchmark.name!r} failed: "
            f"{failed} of {len(benchmark.results)} cases failed"
        )


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8000,
    reload: Annotated[bool, typer.Option(help="Reload Python files during development")] = False,
) -> None:
    """Serve the API and the built Web console."""

    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, reload=reload)


@app.command()
def dev(
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit a machine-readable command list")
    ] = False,
) -> None:
    """Explain the two-process local development command."""

    _configure_utf8()
    if json_output:
        _echo_json(
            {
                "commands": [
                    "uv run admire serve --reload",
                    "pnpm.cmd --dir web dev",
                ],
                "hint": "run the commands in separate terminals",
            }
        )
        return
    console.print(
        "Run [cyan]uv run admire serve --reload[/cyan] and "
        "[cyan]pnpm.cmd --dir web dev[/cyan] in separate terminals."
    )


if __name__ == "__main__":
    app()
