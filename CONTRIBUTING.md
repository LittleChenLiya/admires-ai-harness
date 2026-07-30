# Contributing

Thank you for helping improve Admire's AI Harness. Contributions are welcome across runtime
semantics, safety, evaluation, adapters, the trace console, tests, and documentation.

## Find the right starting point

- Follow the [contributor quickstart](docs/contributor-quickstart.md) for setup and focused checks.
- Use the [code map](docs/code-map.md) to find the smallest change surface.
- Pick a [`good first issue`](https://github.com/LittleChenLiya/admires-ai-harness/labels/good%20first%20issue)
  or a [`help wanted`](https://github.com/LittleChenLiya/admires-ai-harness/labels/help%20wanted) task.
- Read the [roadmap](ROADMAP.md) before proposing a larger abstraction.

Usage questions and early ideas belong in
[Discussions](https://github.com/LittleChenLiya/admires-ai-harness/discussions); reproducible bugs
and scoped work belong in Issues.

Comment on an issue before starting. For tasks labeled `needs design`, wait for agreement on
the contract and scope before investing in an implementation.

## Development setup

Python 3.12, Node 22+, uv, and pnpm 11 are required. The default model is deterministic,
offline, and needs no API key.

```bash
make install
uv run admire demo all
make check
```

On Windows, use the equivalent `uv run ...` and `pnpm.cmd --dir web ...` commands from the
`Makefile` if GNU Make is unavailable.

## Change expectations

- Keep one pull request focused on one problem and link the issue with `Closes #<number>`.
- Runtime semantics need tests for normal, budget/failure, and recovery behavior.
- Tool, policy, filesystem, process, and MCP changes need security-negative tests.
- User-visible flows should include a component, browser, or API integration test.
- New adapters normalize errors and preserve the no-key offline default.
- Update the README capability matrix only when linked executable evidence exists.
- Avoid drive-by refactors, dependency churn, or generated files unrelated to the issue.

Fill in the pull request template with what changed, why, and the exact checks you ran. CI must
pass before merge. A maintainer may ask to split a change when review or rollback would be safer
in smaller pieces.

## Security and data handling

Do not commit API keys, `.env`, generated databases, run workspaces, or model outputs containing
private data. Report vulnerabilities privately using the instructions in `SECURITY.md`; do not
open a public issue for an exploitable problem.

By contributing, you agree that your work is licensed under the repository's MIT license and
that participation follows the `CODE_OF_CONDUCT.md`.
