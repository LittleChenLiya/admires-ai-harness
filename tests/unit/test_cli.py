from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from admires_harness.cli import app

runner = CliRunner()


def test_list_and_dev_expose_reference_workflows_and_local_commands() -> None:
    listed = runner.invoke(app, ["list"])

    assert listed.exit_code == 0
    assert all(name in listed.stdout for name in ("research", "coding", "collaboration"))

    dev = runner.invoke(app, ["dev"])

    assert dev.exit_code == 0
    assert "admire serve --reload" in dev.stdout
    assert "pnpm.cmd --dir web dev" in dev.stdout


def test_demo_command_runs_offline_and_rejects_unknown_workflow(tmp_path: Path) -> None:
    completed = runner.invoke(
        app,
        [
            "demo",
            "research",
            "--task",
            "Explain replayable event streams.",
            "--json",
            "--data-dir",
            str(tmp_path / "demo"),
        ],
    )

    assert completed.exit_code == 0
    payload = json.loads(completed.stdout)
    assert payload["runs"][0]["demo"] == "research"
    assert payload["runs"][0]["status"] == "succeeded"
    assert "Sources" in payload["runs"][0]["output"]

    invalid = runner.invoke(app, ["demo", "unknown", "--data-dir", str(tmp_path / "bad")])
    assert invalid.exit_code == 2
    assert "choose research, coding, collaboration, or all" in invalid.output


def test_evaluate_writes_report_and_accepts_it_as_a_baseline(tmp_path: Path) -> None:
    dataset = tmp_path / "reference.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "research-smoke",
                "input": "Why are durable events useful?",
                "expected": {"text": "Sources", "tools": []},
                "metadata": {"demo": "research"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "baseline.json"

    first = runner.invoke(
        app,
        [
            "evaluate",
            str(dataset),
            "--report",
            str(report),
            "--data-dir",
            str(tmp_path / "first"),
        ],
    )

    assert first.exit_code == 0
    assert report.is_file()
    assert json.loads(report.read_text(encoding="utf-8"))["passed"] is True

    compared = runner.invoke(
        app,
        [
            "evaluate",
            str(dataset),
            "--baseline",
            str(report),
            "--data-dir",
            str(tmp_path / "candidate"),
        ],
    )

    assert compared.exit_code == 0
    assert json.loads(compared.stdout)["passed"] is True


def test_evaluate_exits_nonzero_when_the_candidate_regresses(tmp_path: Path) -> None:
    dataset = tmp_path / "reference.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "research-smoke",
                "input": "Why are durable events useful?",
                "expected": {"text": "Sources", "tools": []},
                "metadata": {"demo": "research"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "baseline.json"
    first = runner.invoke(
        app,
        [
            "evaluate",
            str(dataset),
            "--report",
            str(report),
            "--data-dir",
            str(tmp_path / "first"),
        ],
    )
    assert first.exit_code == 0

    baseline_payload = json.loads(report.read_text(encoding="utf-8"))
    baseline_payload["results"].append(
        {
            "case_id": "ghost-case",
            "passed": True,
            "score": 1.0,
            "output": None,
            "scores": [],
            "error": None,
        }
    )
    baseline_payload["pass_rate"] = 1.0
    baseline_payload["mean_score"] = 1.0
    report.write_text(json.dumps(baseline_payload), encoding="utf-8")

    regressed = runner.invoke(
        app,
        [
            "evaluate",
            str(dataset),
            "--baseline",
            str(report),
            "--data-dir",
            str(tmp_path / "candidate"),
        ],
    )

    assert regressed.exit_code == 1
    assert "ghost-case: missing" in regressed.output
