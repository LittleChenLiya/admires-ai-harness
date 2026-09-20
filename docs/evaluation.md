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

Datasets are strict JSONL, loaded line by line so arbitrarily large files stream with
constant memory while every error keeps its 1-based source line number.

- Encoding is UTF-8; a leading byte-order mark is tolerated.
- Blank lines, whitespace-only lines, and lines starting with `#` are skipped; trailing
  whitespace on a record line is ignored.
- Each remaining line must be a single JSON object with required string fields `id`
  (non-empty) and `input`, plus optional `expected` (any JSON value), `metadata`
  (JSON object), and `tags` (array of unique strings). Unknown fields are rejected.
- Limits: `id` at most 256 characters, `input` at most 100,000 characters, and any
  single line at most 1,000,000 characters. Case IDs must be unique within a dataset.
- Unicode content (CJK, emoji, combining characters) is preserved verbatim and round-trips
  through `write_jsonl`/`load_jsonl` without normalization.

Every failure raises `DatasetError` formatted as `path:line: message` (file-level
failures such as an unreadable path omit the line). Messages name the offending field and
error kind but never echo the record's content, so sensitive prompts cannot leak through
logs or CI output.
