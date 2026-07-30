# Contributing

Thank you for helping improve Admire's AI Harness. Python 3.12, Node 22+, uv, and pnpm are
required. Run `make install`, then `make check` before opening a pull request. On Windows, use
the equivalent `uv run ...` and `pnpm.cmd --dir web ...` commands if GNU Make is unavailable.

Keep changes focused. Runtime semantics need tests that demonstrate normal, budget/failure,
and recovery behavior. Tool changes need security-negative tests. User-visible flows should
include a browser or API integration test. Update the feature matrix only when linked evidence
exists.

Do not commit API keys, `.env`, generated databases, workspaces, or model outputs containing
private data. By contributing, you agree that your work is licensed under the repository's
MIT license.
