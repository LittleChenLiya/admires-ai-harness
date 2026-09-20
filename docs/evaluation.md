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
improvements separately. The required CI gate instead runs the checked-in deterministic
dataset directly and requires every case to pass. Randomized or live-provider evals are
informative and are never mixed into that offline gate.

## Baseline comparison semantics

`compare_reports` matches cases by `case_id` and refuses ambiguous input: reports with
duplicate case IDs are rejected (both at model validation and again inside the comparison),
and all tolerances must be finite fractions in `[0, 1]`. Per-case changes are classified as
`missing` (baseline case absent from the candidate, a regression unless explicitly allowed),
`new_case` (added coverage, reported as an improvement), `new_failure` (a previously
passing case now fails, always a regression regardless of tolerances), `score_drop` (a score
decrease beyond `case_score_tolerance`), and `improvement` (a previously failing case now
passes). Aggregate pass-rate and mean-score drops beyond their tolerances add synthetic
`__aggregate_pass_rate__`/`__aggregate_mean_score__` regressions, but aggregate tolerances
never hide a per-case regression. Empty reports compare cleanly with zero deltas. Reports
reject NaN and infinite scores, latencies, and costs, and `write_json`/`load_report`
preserve errors, latency, token usage, and cost. When a comparison fails, the `evaluate`
command prints the machine-readable comparison, summarizes each regression on stderr, and
exits with code 1.

The checked-in reference dataset contains three deterministic positive cases, one per demo.
Security and recovery negative paths (denied write, escaped path, stale approval, budget
exhaustion, concurrent resume, and approval recovery) live in focused pytest suites rather
than in that JSONL dataset.
