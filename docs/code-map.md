# Code map

Use this map to find the smallest change surface before editing. The runtime is deliberately
split from adapters and user interfaces so its contracts remain testable without network or
framework dependencies.

## Repository map

| Path                                 | Responsibility                                                    | Primary evidence            |
| ------------------------------------ | ----------------------------------------------------------------- | --------------------------- |
| `src/admires_harness/runtime/`       | graph compilation, execution, plans, events, checkpoints, budgets | `tests/unit/runtime/`       |
| `src/admires_harness/contracts.py`   | versioned shared run, event, model, tool, and approval contracts  | unit and integration tests  |
| `src/admires_harness/ports.py`       | injected protocol boundaries used by the runtime                  | adapter tests               |
| `src/admires_harness/storage/`       | SQLite event and checkpoint persistence                           | `tests/unit/storage/`       |
| `src/admires_harness/services/`      | run lifecycle shared by CLI and HTTP                              | integration tests           |
| `src/admires_harness/adapters/`      | model-provider and MCP translations                               | focused adapter tests       |
| `src/admires_harness/tools/`         | typed tool registration and output/idempotency bounds             | `tests/unit/tools/`         |
| `src/admires_harness/safety/`        | policy, approval, workspace, and process boundaries               | unit and security tests     |
| `src/admires_harness/memory/`        | short-term memory, long-term store, chunking, embeddings, RAG     | `tests/unit/test_memory.py` |
| `src/admires_harness/evaluation/`    | datasets, graders, reports, and baseline comparison               | evaluation tests            |
| `src/admires_harness/observability/` | event-derived run views and metrics                               | projection tests            |
| `src/admires_harness/agents/`        | three reference graphs and deterministic demo behavior            | CLI/integration/e2e tests   |
| `src/admires_harness/api/`           | FastAPI REST, SSE, approval, and static-Web surface               | integration tests           |
| `web/src/`                           | React launch, run list, trace inspector, and approval UI          | Vitest component tests      |
| `web/tests/e2e/`                     | user-visible research and approval workflows                      | Playwright                  |
| `evaluation/`                        | checked-in deterministic benchmark cases                          | evaluation CI job           |

## A run through the system

1. The CLI or API asks `RunManager` to start a named reference graph.
2. `GraphRuntime` validates budgets, acquires execution ownership, and advances compiled nodes.
3. Every lifecycle transition is appended as a versioned `RunEvent` before live publication.
4. Checkpoints record the next node and state revision at node boundaries.
5. Model, memory, tool, and approval capabilities enter through injected ports.
6. Observability projections, replay, the API, and the Web console all read the persisted event
   language instead of maintaining separate histories.

## Change-routing guide

- A new reference workflow usually touches `agents/`, tests, the demo registry, and optionally
  the Web launcher. It should not add special cases to the runtime.
- A new provider belongs in `adapters/` behind an existing or deliberately revised port. It must
  normalize errors and preserve offline defaults.
- A new tool declares schema and safety metadata in `tools/`, then executes through the secured
  boundary. Do not authorize tools inside a prompt.
- A persistence change starts from the runtime store protocols, then an adapter and conflict or
  recovery tests.
- A new metric should be derivable from persisted events; avoid a second mutable metrics store.
- A public contract change needs a design issue, migration impact, and tests for old or invalid
  input where relevant.

Read `docs/architecture.md` for state and recovery semantics and `docs/security.md` before
changing tools, processes, paths, approvals, credentials, MCP, or network defaults.
