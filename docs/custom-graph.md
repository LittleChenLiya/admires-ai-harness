# Custom graph demo guide

This guide walks through `examples/custom_graph.py`, a small offline example that defines,
compiles, and runs a custom conditional graph with the deterministic mock model. It uses only
the public runtime contracts — no test helpers, network, credentials, or wall clock.

## What the example does

The graph is a support router with three nodes:

1. `classify` asks the mock model for a one-word category (`refund` or `general`) and
   stores it in state.
2. A conditional edge routes `refund` questions to `escalate`, which builds a
   deterministic ticket summary without another model call.
3. A fallback edge routes everything else to `answer`, which drafts a deterministic reply.

## Runtime contracts used

- `StateGraph` / `CompiledGraph` — the fluent builder and the validated immutable graph
  (`src/admires_harness/runtime/graph.py`).
- `NodeContext` / `NodeResult` — the node invocation context and the sole mutation and
  routing surface (`src/admires_harness/runtime/types.py`).
- `GraphRuntime` — the event-sourced execution engine; capabilities such as the model arrive
  through `dependencies` (`src/admires_harness/runtime/engine.py`).
- `InMemoryEventStore` / `InMemoryCheckpointStore` — durable-contract stores that keep the
  demo fully in process (`src/admires_harness/runtime/stores.py`).
- `MockModelProvider` — the deterministic offline model adapter
  (`src/admires_harness/adapters/models.py`).

See [the architecture guide](architecture.md) for state, event, and checkpoint semantics, and
[the extension guide](extending.md) for the broader rules for adding graphs.

## Define the nodes

Each node is an async callable that accepts `NodeContext` and returns `NodeResult`. Nodes
never mutate `context.state`; they declare changes through `NodeResult.update`, finish with
`NodeResult(finish=True, output=...)`, and read injected capabilities from
`context.dependencies`. The example fails fast with a clear error when the `model`
dependency is missing, and the runtime records that as a failed run.

## Declare the branches

Every possible destination must be declared as an edge. Conditional edges are evaluated in
declaration order and take precedence over the unconditional fallback edge:

```python
.add_edge("classify", "escalate", condition=lambda state: state.get("category") == "refund")
.add_edge("classify", "answer")  # fallback
```

Predicates receive a read-only deep copy of the state, so routing cannot mutate runtime state.

## Compile and validate

`StateGraph.compile()` validates eagerly at application startup: it rejects a missing entry
point, edges to unknown nodes, duplicate fallback edges, and unreachable nodes with
`GraphDefinitionError`. Compile once and share the resulting `CompiledGraph` between runs.

## Run it offline

`run_question` wires `GraphRuntime` with the in-memory stores and a
`MockModelProvider` whose rule-based responder classifies any question containing "refund" as
`refund` and everything else as `general`. Then:

```bash
uv run python examples/custom_graph.py
```

prints both branches deterministically:

```text
Q: Where is my refund?
   status=succeeded category=refund
   output=Escalated refund request as ticket REFUND-where-is-my-refund.
Q: How do I change my password?
   status=succeeded category=general
   output=Support answer: How do I change my password?
```

## Inputs, branches, and outputs

| Input (`initial_state`)                 | Branch    | Final state markers                             | Output                                                |
| ---------------------------------------------- | --------- | ----------------------------------------------- | ----------------------------------------------------- |
| `{"input": "Where is my refund?"}`          | escalate  | `category="refund"`, `resolution="escalated"` | `Escalated refund request as ticket REFUND-...`     |
| `{"input": "How do I change my password?"}` | answer    | `category="general"`, `resolution="answered"` | `Support answer: How do I change my password?`      |

Both runs succeed in two steps and record their events and checkpoints in the in-memory
stores. Repeating a run with the same input produces identical state and output.

## Test it

`tests/unit/test_custom_graph_example.py` covers both branch outcomes, run determinism,
compile-time rejection of invalid definitions, the missing-dependency failure path, and an
executable smoke test of `main()`:

```bash
uv run pytest tests/unit/test_custom_graph_example.py
```
