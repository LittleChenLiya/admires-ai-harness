# Roadmap

Admire's AI Harness is a learning-first reference implementation of an AI agent control
plane. The roadmap favors changes that make orchestration semantics easier to inspect, test,
and extend without weakening the offline or safe-by-default path.

The roadmap is directional, not a delivery promise. Open issues and accepted pull requests
are the source of truth for work in progress.

## Now: make v1 easy to learn and extend

- Give first-time contributors a short, reproducible path from clone to focused test.
- Improve trace-console ergonomics, especially stream recovery, filtering, and export.
- Add executable examples for custom graphs, model adapters, and policy-wrapped MCP tools.
- Expand deterministic evaluation fixtures and failure-path coverage.
- Keep the one-command, no-key Docker and local demo paths reliable.

## Next: strengthen extension boundaries

- Support application-defined state reducers without making graph updates implicit.
- Add durable tool-receipt storage for crash-safe side-effect deduplication.
- Export event-derived traces and metrics through an OpenTelemetry-compatible boundary.
- Provide an opt-in semantic embedding adapter while preserving the offline hashing baseline.
- Make storage and live-event delivery replaceable through documented ports.

## Later: production-oriented experiments

- Transactional database and outbox-backed event publication.
- Distributed worker ownership, leases, and heartbeats.
- Stronger tenant isolation and deployment guidance for untrusted workloads.
- Larger, versioned evaluation suites and cross-provider comparison reports.

These experiments are intentionally outside the v1 contract. They should land only with a
clear design, negative-path tests, migration notes, and an unchanged deterministic offline
experience.

## How to participate

- Start with a
  [`good first issue`](https://github.com/LittleChenLiya/admires-ai-harness/labels/good%20first%20issue)
  if this is your first contribution.
- Browse
  [`help wanted`](https://github.com/LittleChenLiya/admires-ai-harness/labels/help%20wanted)
  for larger, maintainer-supported work.
- Comment on an issue before starting so scope and ownership are visible.
- Open a feature request before implementing a new public abstraction or changing a safety,
  persistence, or recovery contract.

See the [contributor quickstart](docs/contributor-quickstart.md) for the shortest local path and
the [code map](docs/code-map.md) for where changes belong.
