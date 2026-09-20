# Evaluation methodology

The evaluation system measures the harness, not only the prose of a final answer. Each case
has a stable ID, input, optional fixture workspace, execution budget, and one or more graders.
Reports preserve per-grader evidence, trace/run IDs, latency, tokens, cost, and harness
version.

Deterministic graders are ordered by strength:

1. policy outcome and terminal status;
2. executable tests or artifact assertions;
3. required tool/delegation trajectory;
4. exact or contains checks on final text.

A candidate report can be compared with a previously saved report by passing `--baseline`;
the comparison fails on newly failing cases or a configured score regression and reports
improvements separately. On the CLI, `uv run admire evaluate --baseline report.json` prints
each regression to stderr, exits `1`, and can write the machine-readable comparison with
`--comparison comparison.json`; stdout always carries the candidate benchmark report as one
JSON document, even when the comparison fails. The required CI gate instead runs the
checked-in deterministic dataset directly and requires every case to pass. Randomized or
live-provider evals are informative and are never mixed into that offline gate.

The checked-in reference dataset contains three deterministic positive cases, one per demo.
Security and recovery negative paths (denied write, escaped path, stale approval, budget
exhaustion, concurrent resume, and approval recovery) live in focused pytest suites rather
than in that JSONL dataset.
