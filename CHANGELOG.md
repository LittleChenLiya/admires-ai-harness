# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Custom graph guide (`docs/custom-graphs.md`) and an executable offline example
  (`examples/custom_graph.py`) showing a conditional graph on the deterministic mock model,
  with unit tests covering both branches and failure paths.

## [1.0.0] - 2026-07-30

### Added

- Typed asynchronous state graphs with compile-time validation and conditional routing.
- Event-sourced execution with ordered streaming, checkpoint/resume, replay, execution leases,
  budgets, cancellation, and repeated-action protection.
- Validated multi-agent plan DAGs with parallel child runs and atomic budget reservation.
- Mock and OpenAI-compatible model adapters plus an optional allowlisted MCP stdio adapter.
- Typed tool registry, deterministic safety policy, one-shot human approval, workspace sandbox,
  and bounded process execution.
- Namespaced short- and long-term memory with deterministic offline RAG.
- Versioned evaluation datasets, deterministic graders, baseline comparison, and trace-derived
  metrics.
- Typer CLI, FastAPI REST/SSE service, and React run-inspection console.
- Research, coding, and collaboration demos that run offline with no API key or model cost.
- Python, Web, security, integration, browser, container, and CodeQL CI gates.
- Public roadmap, contributor quickstart, code map, and structured contribution templates.

[Unreleased]: https://github.com/LittleChenLiya/admires-ai-harness/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/LittleChenLiya/admires-ai-harness/releases/tag/v1.0.0
