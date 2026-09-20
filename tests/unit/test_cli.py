from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from admires_harness.cli import _redact, app

runner = CliRunner()


def _write_dataset(path: Path, expected_text: str = "Sources") -> Path:
    path.write_text(
        json.dumps(
            {
                "id": "research-smoke",
                "input": "Why are durable events useful?",
                "expected": {"text": expected_text, "tools": []},
                "metadata": {"demo": "research"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_list_and_dev_expose_reference_workflows_and_local_commands() -> None:
    listed = runner.invoke(app, ["list"])

    assert listed.exit_code == 0
    assert all(name in listed.stdout for name in ("research", "coding", "collaboration"))

    dev = runner.invoke(app, ["dev"])

    assert dev.exit_code == 0
    assert "admire serve --reload" in dev.stdout
    assert "pnpm.cmd --dir web dev" in dev.stdout


def test_list_and_dev_json_output_is_machine_readable() -> None:
    listed = runner.invoke(app, ["list", "--json"])

    assert listed.exit_code == 0
    assert listed.stderr == ""
    workflows = json.loads(listed.stdout)["workflows"]
    assert {workflow["id"] for workflow in workflows} == {
        "research",
        "coding",
        "collaboration",
    }

    dev = runner.invoke(app, ["dev", "--json"])

    assert dev.exit_code == 0
    assert dev.stderr == ""
    commands = json.loads(dev.stdout)["commands"]
    assert any("admire serve --reload" in command for command in commands)


def test_redact_masks_sensitive_keys_but_keeps_metrics() -> None:
    payload = {
        "api_key": "sk-secret",
        "modelApiKey": "sk-secret",
        "Authorization": "Bearer sk-secret",
        "nested": {"password": "hunter2", "input_tokens": 12},
        "items": [{"refresh_token": "tok"}, {"output_tokens": 7}],
    }

    redacted = _redact(payload)

    assert redacted["api_key"] == "***"
    assert redacted["modelApiKey"] == "***"
    assert redacted["Authorization"] == "***"
    assert redacted["nested"] == {"password": "***", "input_tokens": 12}
    assert redacted["items"] == [{"refresh_token": "***"}, {"output_tokens": 7}]


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


def test_evaluate_rejects_missing_and_malformed_dataset(tmp_path: Path) -> None:
    missing = runner.invoke(app, ["evaluate", str(tmp_path / "missing.jsonl")])

    assert missing.exit_code == 2
    assert missing.stdout == ""

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("{not json}\n", encoding="utf-8")

    invalid = runner.invoke(app, ["evaluate", str(malformed), "--data-dir", str(tmp_path / "data")])

    assert invalid.exit_code == 2
    assert invalid.stdout == ""
    assert "invalid dataset" in invalid.stderr


def test_evaluate_report_write_failure_keeps_stdout_machine_readable(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "reference.jsonl")

    completed = runner.invoke(
        app,
        [
            "evaluate",
            str(dataset),
            "--report",
            str(tmp_path),
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert completed.exit_code == 1
    assert json.loads(completed.stdout)["passed"] is True
    assert "could not write report" in completed.stderr


def test_evaluate_baseline_regression_fails_with_details(tmp_path: Path) -> None:
    baseline_dataset = _write_dataset(tmp_path / "baseline.jsonl")
    baseline_report = tmp_path / "baseline.json"
    recorded = runner.invoke(
        app,
        [
            "evaluate",
            str(baseline_dataset),
            "--report",
            str(baseline_report),
            "--data-dir",
            str(tmp_path / "baseline-data"),
        ],
    )
    assert recorded.exit_code == 0

    regressed_dataset = _write_dataset(
        tmp_path / "regressed.jsonl", expected_text="text the mock never emits"
    )

    compared = runner.invoke(
        app,
        [
            "evaluate",
            str(regressed_dataset),
            "--baseline",
            str(baseline_report),
            "--data-dir",
            str(tmp_path / "candidate-data"),
        ],
    )

    assert compared.exit_code == 1
    payload = json.loads(compared.stdout)
    assert payload["passed"] is False
    assert payload["baseline"]["passed"] is False
    assert payload["baseline"]["regressions"][0]["kind"] == "new_failure"
    assert "regressed" in compared.stderr
    assert "research-smoke" in compared.stderr


def test_evaluate_json_output_is_stable_and_unicode_safe(tmp_path: Path) -> None:
    dataset = tmp_path / "unicode.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "研究-案例",
                "input": "为什么可重放的事件流很有用?",
                "expected": {"text": "Sources", "tools": []},
                "metadata": {"demo": "research"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    runs = [
        runner.invoke(
            app,
            ["evaluate", str(dataset), "--data-dir", str(tmp_path / f"data-{index}")],
        )
        for index in range(2)
    ]

    assert all(completed.exit_code == 0 for completed in runs)
    assert all(completed.stderr == "" for completed in runs)
    assert "研究-案例" in runs[0].stdout
    first, second = (json.loads(completed.stdout) for completed in runs)
    assert first.keys() == second.keys()
    for key in ("name", "passed", "pass_rate", "mean_score", "threshold"):
        assert first[key] == second[key]
    first_case, second_case = first["results"][0], second["results"][0]
    assert first_case["case_id"] == second_case["case_id"] == "研究-案例"
    assert first_case["passed"] == second_case["passed"] is True
    assert first_case["score"] == second_case["score"]
    assert "Sources" in first_case["output"]["text"]


def test_evaluate_never_prints_configured_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADMIRE_MODEL_API_KEY", "sk-super-secret-value")
    dataset = _write_dataset(tmp_path / "reference.jsonl")

    completed = runner.invoke(app, ["evaluate", str(dataset), "--data-dir", str(tmp_path / "data")])

    assert completed.exit_code == 0
    assert "sk-super-secret-value" not in completed.stdout
    assert "sk-super-secret-value" not in completed.stderr
