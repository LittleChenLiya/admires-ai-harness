"""Contract tests for the CLI output, exit-code, and redaction guarantees."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from admires_harness.cli import REDACTED, app, redact_sensitive

runner = CliRunner()


def _write_dataset(path: Path, case_id: str, expected_text: str, task: str = "Why events?") -> None:
    path.write_text(
        json.dumps(
            {
                "id": case_id,
                "input": task,
                "expected": {"text": expected_text, "tools": []},
                "metadata": {"demo": "research"},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_list_and_dev_support_machine_readable_json() -> None:
    listed = runner.invoke(app, ["list", "--json"])

    assert listed.exit_code == 0
    assert listed.stderr == ""
    payload = json.loads(listed.stdout)
    ids = {workflow["id"] for workflow in payload["workflows"]}
    assert {"research", "coding", "collaboration"} <= ids
    assert all(
        set(workflow) == {"id", "title", "agents", "capabilities"}
        for workflow in payload["workflows"]
    )

    again = runner.invoke(app, ["list", "--json"])
    assert again.stdout == listed.stdout

    dev = runner.invoke(app, ["dev", "--json"])

    assert dev.exit_code == 0
    assert dev.stderr == ""
    summary = json.loads(dev.stdout)
    assert summary["commands"] == [
        "uv run admire serve --reload",
        "pnpm.cmd --dir web dev",
    ]
    assert summary["urls"]["web"] == "http://localhost:5173"


def test_demo_json_is_stable_parseable_and_keeps_stderr_clean(tmp_path: Path) -> None:
    args = [
        "demo",
        "research",
        "--task",
        "Explain replayable event streams.",
        "--json",
        "--data-dir",
        str(tmp_path / "demo"),
    ]

    first = runner.invoke(app, args)
    second = runner.invoke(app, args)

    for completed in (first, second):
        assert completed.exit_code == 0
        assert completed.stderr == ""
        payload = json.loads(completed.stdout)
        assert list(payload) == ["runs"]
        assert payload["runs"][0]["status"] == "succeeded"
    assert set(json.loads(first.stdout)["runs"][0]) == set(json.loads(second.stdout)["runs"][0])


def test_demo_json_handles_long_and_unicode_input(tmp_path: Path) -> None:
    task = "总结事件溯源的好处。" + "长输出" * 2000
    completed = runner.invoke(
        app,
        ["demo", "research", "--task", task, "--json", "--data-dir", str(tmp_path / "u")],
    )

    assert completed.exit_code == 0
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["runs"][0]["input"] == task
    assert "总结事件溯源的好处。" in completed.stdout


def test_demo_rejects_unknown_workflow_and_file_data_dir(tmp_path: Path) -> None:
    unknown = runner.invoke(app, ["demo", "unknown", "--data-dir", str(tmp_path / "bad")])

    assert unknown.exit_code == 2
    assert "choose research, coding, collaboration, or all" in unknown.stderr

    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("occupied", encoding="utf-8")
    invalid_dir = runner.invoke(app, ["demo", "research", "--data-dir", str(not_a_dir)])

    assert invalid_dir.exit_code == 2
    assert "--data-dir must be a directory" in invalid_dir.stderr


def test_evaluate_rejects_invalid_dataset_and_baseline(tmp_path: Path) -> None:
    missing = runner.invoke(app, ["evaluate", str(tmp_path / "missing.jsonl")])

    assert missing.exit_code == 2

    broken = tmp_path / "broken.jsonl"
    broken.write_text("{not json}\n", encoding="utf-8")
    invalid_dataset = runner.invoke(
        app, ["evaluate", str(broken), "--data-dir", str(tmp_path / "data")]
    )

    assert invalid_dataset.exit_code == 2
    assert "invalid evaluation case" in invalid_dataset.stderr

    dataset = tmp_path / "reference.jsonl"
    _write_dataset(dataset, "case-a", "Sources")
    bad_baseline = tmp_path / "baseline.json"
    bad_baseline.write_text("{not a report}", encoding="utf-8")
    invalid_baseline = runner.invoke(
        app,
        [
            "evaluate",
            str(dataset),
            "--baseline",
            str(bad_baseline),
            "--data-dir",
            str(tmp_path / "data2"),
        ],
    )

    assert invalid_baseline.exit_code == 2
    assert "invalid baseline report" in invalid_baseline.stderr


def test_evaluate_report_write_failure_is_a_clean_error(tmp_path: Path) -> None:
    dataset = tmp_path / "reference.jsonl"
    _write_dataset(dataset, "case-a", "Sources")
    report_dir = tmp_path / "occupied"
    report_dir.mkdir()

    completed = runner.invoke(
        app,
        [
            "evaluate",
            str(dataset),
            "--report",
            str(report_dir),
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert completed.exit_code == 1
    assert "could not write report" in completed.stderr
    assert json.loads(completed.stdout)["passed"] is True


def test_evaluate_baseline_regression_exits_one_with_details(tmp_path: Path) -> None:
    baseline_dataset = tmp_path / "baseline.jsonl"
    _write_dataset(baseline_dataset, "case-a", "Sources")
    baseline_report = tmp_path / "baseline.json"
    recorded = runner.invoke(
        app,
        [
            "evaluate",
            str(baseline_dataset),
            "--report",
            str(baseline_report),
            "--data-dir",
            str(tmp_path / "first"),
        ],
    )
    assert recorded.exit_code == 0

    candidate_dataset = tmp_path / "candidate.jsonl"
    _write_dataset(candidate_dataset, "case-a", "definitely-not-present-xyz")
    comparison_path = tmp_path / "comparison.json"
    regressed = runner.invoke(
        app,
        [
            "evaluate",
            str(candidate_dataset),
            "--baseline",
            str(baseline_report),
            "--comparison",
            str(comparison_path),
            "--data-dir",
            str(tmp_path / "second"),
        ],
    )

    assert regressed.exit_code == 1
    assert "regressed against baseline" in regressed.stderr
    assert "case-a" in regressed.stderr
    assert json.loads(regressed.stdout)["passed"] is False
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert comparison["passed"] is False
    assert comparison["regressions"][0]["case_id"] == "case-a"
    assert comparison["regressions"][0]["kind"] == "new_failure"


def test_evaluate_success_keeps_stderr_clean_and_output_stable(tmp_path: Path) -> None:
    dataset = tmp_path / "reference.jsonl"
    _write_dataset(dataset, "case-a", "Sources", task="为什么持久事件有用")
    args = ["evaluate", str(dataset), "--data-dir", str(tmp_path / "data")]

    first = runner.invoke(app, args)
    second = runner.invoke(app, args)

    for completed in (first, second):
        assert completed.exit_code == 0
        assert completed.stderr == ""
        payload = json.loads(completed.stdout)
        assert payload["passed"] is True
        assert payload["results"][0]["case_id"] == "case-a"
    assert set(json.loads(first.stdout)) == set(json.loads(second.stdout))


def test_sensitive_settings_never_reach_stdout_or_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADMIRE_MODEL_API_KEY", "sk-test-secret-value")
    dataset = tmp_path / "reference.jsonl"
    _write_dataset(dataset, "case-a", "Sources")

    evaluated = runner.invoke(
        app,
        [
            "evaluate",
            str(dataset),
            "--report",
            str(tmp_path / "report.json"),
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )
    assert evaluated.exit_code == 0
    combined = evaluated.stdout + evaluated.stderr
    assert "sk-test-secret-value" not in combined
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert "sk-test-secret-value" not in json.dumps(report)
    assert set(report["environment"]) <= {
        "python",
        "platform",
        "environment",
        "model_provider",
        "model_name",
    }

    demoed = runner.invoke(
        app, ["demo", "research", "--json", "--data-dir", str(tmp_path / "demo")]
    )
    assert demoed.exit_code == 0
    assert "sk-test-secret-value" not in demoed.stdout + demoed.stderr


def test_redact_sensitive_scrubs_keys_but_keeps_metrics() -> None:
    payload = {
        "model_api_key": "sk-secret",
        "nested": {"access_token": "tok", "password": "pw"},
        "metrics": {"inputTokens": 12, "outputTokens": 3},
        "items": [{"authorization": "Bearer x"}],
    }

    redacted = redact_sensitive(payload)

    assert redacted == {
        "model_api_key": REDACTED,
        "nested": {"access_token": REDACTED, "password": REDACTED},
        "metrics": {"inputTokens": 12, "outputTokens": 3},
        "items": [{"authorization": REDACTED}],
    }
