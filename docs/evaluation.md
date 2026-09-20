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

The checked-in reference dataset contains three deterministic positive cases, one per demo.
Security and recovery negative paths (denied write, escaped path, stale approval, budget
exhaustion, concurrent resume, and approval recovery) live in focused pytest suites rather
than in that JSONL dataset.

## Dataset format

Datasets are strict JSONL: one JSON object per line, decoded as UTF-8 with an optional
leading BOM. Each record maps to an `EvalCase`:

- `id` (string, required): stable case identifier, 1-256 characters, unique per dataset.
- `input` (string, required): the prompt or fixture input, at most 100,000 characters.
- `expected` (any JSON value, optional): reference output for deterministic graders.
- `metadata` (object, optional): arbitrary JSON annotations.
- `tags` (array of strings, optional): stored as an immutable set.

Unknown fields are rejected. Blank or whitespace-only lines and lines starting with `#`
are skipped, and surrounding whitespace on record lines is ignored, so datasets stay
diff-friendly. A single raw line may not exceed 1,000,000 characters. Files are streamed
line by line, so arbitrarily large datasets load in constant memory and every record keeps
its 1-based source line number.

Every loading failure raises `DatasetError` with the dataset path and source line number
(for example `evals/core.jsonl:17: invalid JSON: ...`). Messages describe the problem —
invalid JSON, a non-object record, a missing or mistyped field, an over-limit field, or a
duplicate `id` (which also names the line of the first definition) — but never echo record
payloads, so datasets may contain sensitive inputs without leaking them into logs or CI
output. Unicode content is fully supported and round-trips stably through
`write_jsonl`, which emits one canonical, diff-friendly case per line.
