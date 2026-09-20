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

Baseline comparison semantics are deterministic and fail closed:

- Reports with duplicate case IDs are rejected, as are NaN or infinite scores, latencies,
  and tolerances; empty reports compare cleanly with zero-valued aggregates.
- A case present in the baseline but absent from the candidate is a `missing` regression
  unless missing cases are explicitly allowed; extra candidate cases are reported as
  `new_case` improvements.
- A previously passing case that now fails is a `new_failure` regression; a previously
  failing case that now passes is an `improvement`. Score drops beyond the per-case
  tolerance are `score_drop` regressions, so a single regressed case fails the gate even
  when the aggregate mean improves.
- Pass-rate and mean-score tolerances are absolute fractions in `[0, 1]` and are
  inclusive at the boundary. When a comparison fails, the CLI prints the regressed case
  IDs and kinds, then exits with a non-zero status.
- Report JSON round-trips losslessly: per-case errors, latency, token usage, and cost are
  preserved, so any report written by `admire evaluate --report` can be reused as a
  `--baseline` file.

The checked-in reference dataset contains three deterministic positive cases, one per demo.
Security and recovery negative paths (denied write, escaped path, stale approval, budget
exhaustion, concurrent resume, and approval recovery) live in focused pytest suites rather
than in that JSONL dataset.
