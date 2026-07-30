# Contributor quickstart

This is the shortest path from a fresh clone to a focused pull request. The default provider is
offline and deterministic, so no API key or paid model is required.

## 1. Prepare the repository

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), Node 22+, and pnpm 11.

```bash
git clone https://github.com/LittleChenLiya/admires-ai-harness.git
cd admires-ai-harness
uv sync --frozen --all-extras
pnpm --dir web install --frozen-lockfile
```

Run a deterministic smoke check:

```bash
uv run admire demo all
uv run admire evaluate evaluation/reference.jsonl
```

On Windows, use `pnpm.cmd` if the PowerShell execution policy blocks the `pnpm.ps1` shim. GNU
Make is optional; every Make target maps to the commands shown in the `Makefile`.

## 2. Pick and claim a task

1. Choose an unassigned
   [`good first issue`](https://github.com/LittleChenLiya/admires-ai-harness/labels/good%20first%20issue)
   or [`help wanted`](https://github.com/LittleChenLiya/admires-ai-harness/labels/help%20wanted) issue.
2. Comment that you would like to work on it and describe your intended approach in one or two
   sentences.
3. Wait for scope confirmation when the issue is labeled `needs design`; small, already-scoped
   tasks can start immediately.
4. Keep one pull request focused on one issue.

If the task you want is not listed, open a feature request first. Security vulnerabilities must
use the private reporting path in `SECURITY.md`.

## 3. Run the smallest useful checks

| Changed area         | Fast feedback                                             | Before opening a PR                        |
| -------------------- | --------------------------------------------------------- | ------------------------------------------ |
| Python module        | `uv run ruff check src tests` and a focused `pytest` file | `make check`                               |
| Runtime/recovery     | focused tests under `tests/unit/runtime`                  | `make test-runtime` and `make check`       |
| Safety/tooling       | focused unit and security tests                           | `make test-security` and `make check`      |
| API workflow         | `uv run python -m pytest tests/integration -v`            | `make check`                               |
| React console        | `pnpm --dir web test`                                     | `make web-check`                           |
| Visible browser flow | focused Playwright spec                                   | `make test-e2e` plus `make web-check`      |
| Documentation only   | check links and commands you changed                      | formatting checks relevant to edited files |

The full local gate is:

```bash
make check
```

## 4. Match tests to the contract

- Runtime changes cover the normal path plus budget, failure, or recovery behavior.
- Tool and sandbox changes include policy-denied and boundary-escape cases.
- Storage changes cover conflicts, ordering, and restart behavior.
- User-visible changes include a unit, API integration, or browser test at the lowest useful
  level.
- Documentation and the README capability matrix only claim behavior backed by executable
  evidence.

## 5. Open the pull request

Use a descriptive title, link the issue with `Closes #<number>`, and fill in the pull request
template. Include the commands you ran and their outcome. Screenshots are useful only for
visible interface changes.

Maintainers review for contract clarity, minimal scope, deterministic offline behavior, and
negative-path coverage. CI must pass before merge.
