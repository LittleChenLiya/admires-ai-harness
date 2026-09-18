# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Bounded, redacted trace export: `GET /api/runs/{id}/export` returns a versioned
  (`schema_version: "1"`) JSON document of a run's persisted events in sequence order, and
  the run inspector console gains an "Export trace" download action. Sensitive payload keys
  are recursively replaced with `[REDACTED]`; see `docs/security.md`.

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
