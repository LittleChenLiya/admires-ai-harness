"""Command-line interface for local runs, server startup, and evaluations."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
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
from admires_harness.evaluation.baseline import (
    EvaluationRegressionError,
    compare_reports,
    load_report,
)
from admires_harness.services.runs import RunManager, as_camel

app = typer.Typer(
    name="admire",
    help="Run, observe, and evaluate AI agents with safe offline defaults.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


@app.command("list")
def list_workflows() -> None:
    """List the built-in reference workflows."""

    table = Table(title="Reference workflows")
    table.add_column("ID", style="cyan")
    table.add_column("Agents")
    table.add_column("Capabilities")
    for demo in list_demos():
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
        console.print_json(json.dumps({"runs": reports}, default=str))
    if any(report["status"] != "succeeded" for report in reports):
        raise typer.Exit(code=1)


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

    cases = load_jsonl(dataset)
    scorers = (ContainsScorer(weight=2), ToolSequenceScorer(exact=True, weight=1))
    benchmark = await EvalRunner(target, scorers, concurrency=concurrency, pass_threshold=1.0).run(
        cases, name=dataset.stem, environment={"model_provider": settings.model_provider}
    )
    if report_path is not None:
        benchmark.write_json(report_path)
    console.print_json(benchmark.model_dump_json())
    if baseline_path is not None:
        comparison = compare_reports(benchmark, load_report(baseline_path))
        if not comparison.passed:
            console.print_json(comparison.model_dump_json())
            try:
                comparison.raise_for_regressions()
            except EvaluationRegressionError as exc:
                err_console.print(f"[red]{exc}[/red]")
                raise typer.Exit(code=1) from exc
    if not benchmark.passed:
        raise typer.Exit(code=1)


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
def dev() -> None:
    """Explain the two-process local development command."""

    console.print(
        "Run [cyan]uv run admire serve --reload[/cyan] and "
        "[cyan]pnpm.cmd --dir web dev[/cyan] in separate terminals."
    )


if __name__ == "__main__":
    app()
