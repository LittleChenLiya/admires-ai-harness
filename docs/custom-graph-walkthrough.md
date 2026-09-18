# Custom graph walkthrough

This guide builds a small **inbox triage** graph from scratch using only the public
runtime contracts. The complete, executable code lives in
[`examples/custom_conditional_graph.py`](../examples/custom_conditional_graph.py); it
runs fully offline with the deterministic mock model — no network, credentials, or
wall-clock assumptions.

For the underlying scheduling and state semantics, see
[Graph semantics](architecture.md#graph-semantics); for contribution conventions, see
[Add a graph](extending.md#add-a-graph).

## Runtime contracts used

| Contract | Role | Source |
| --- | --- | --- |
| `StateGraph` | Fluent builder; `compile()` performs structural validation | [`runtime/graph.py`](../src/admires_harness/runtime/graph.py) |
| `CompiledGraph` | Immutable, validated graph executed by the runtime | [`runtime/graph.py`](../src/admires_harness/runtime/graph.py) |
| `END` | Terminal edge target exported by the runtime package | [`runtime/types.py`](../src/admires_harness/runtime/types.py) |
| `NodeContext` | Per-node capabilities: immutable `state`, `dependencies`, `emit()` | [`runtime/types.py`](../src/admires_harness/runtime/types.py) |
| `NodeResult` | The only mutation/routing surface: `update`, `output`, `finish` | [`runtime/types.py`](../src/admires_harness/runtime/types.py) |
| `GraphRuntime` | Event-sourced execution with checkpoints and budgets | [`runtime/engine.py`](../src/admires_harness/runtime/engine.py) |
| `InMemoryEventStore`, `InMemoryCheckpointStore` | Offline stores for examples and tests | [`runtime/stores.py`](../src/admires_harness/runtime/stores.py) |
| `MockModelProvider` | Deterministic offline model; scripted or responder-driven | [`adapters/models.py`](../src/admires_harness/adapters/models.py) |

All of these are importable from `admires_harness.runtime` (the model lives in
`admires_harness.adapters.models`). The example never imports private test helpers.

## Step 1: Write the nodes

Every node is an async callable that receives a `NodeContext` and returns a
`NodeResult`. Nodes must never mutate `context.state`; they publish changes through
`NodeResult.update`, which the runtime applies as a fresh top-level overwrite before
routing.

`classify` reads the `input` string, calls the injected mock model to draft a reply,
and records a `tone` key (`positive`, `negative`, or `unclear`) in state:

```python
async def classify(context: NodeContext) -> NodeResult:
    ...
    response = await model.complete([...])
    return NodeResult(
        update={"tone": tone, "draft_reply": response.content},
        model_calls=1,
        usage=response.usage,
    )
```

The three branch nodes (`positive_reply`, `negative_reply`, `clarify_request`) read
`draft_reply` from the immutable state, set `routed_to`, return the run `output`, and
finish with `finish=True`.

## Step 2: Declare the conditional edges

After compiling, the runtime calls `CompiledGraph.next_node()` after every node.
Conditional edges are predicates over the **new** state; they are checked in
declaration order, and an edge without a condition is the fallback. Returning `END`
ends the run.

```
classify ──tone == "positive"──▶ positive_reply    ──▶ END
         ──tone == "negative"──▶ negative_reply    ──▶ END
         ──fallback───────────▶ clarify_request    ──▶ END
```

```python
graph = (
    StateGraph("inbox-triage", version="1")
    .add_node("classify", classify)
    .add_node("positive_reply", positive_reply)
    .add_node("negative_reply", negative_reply)
    .add_node("clarify_request", clarify_request)
    .set_entry_point("classify")
    .add_edge("classify", "positive_reply",
              condition=lambda state: state.get("tone") == "positive")
    .add_edge("classify", "negative_reply",
              condition=lambda state: state.get("tone") == "negative")
    .add_edge("classify", "clarify_request")          # fallback, must be last
    .add_edge("positive_reply", END)
    .add_edge("negative_reply", END)
    .add_edge("clarify_request", END)
)
compiled = graph.compile()
```

`compile()` rejects an empty graph, missing/unknown entry point, unknown edge target,
duplicate fallback edges, and unreachable nodes — before a run starts.

## Step 3: Assemble the offline runtime

`GraphRuntime` takes an event store, a checkpoint store, and a `dependencies` mapping.
Nodes access injected capabilities through `context.dependencies`, so the mock model
is just data passed in at the boundary — no monkeypatching or test-only wiring:

```python
runtime = GraphRuntime(
    event_store=InMemoryEventStore(),
    checkpoint_store=InMemoryCheckpointStore(),
    dependencies={"model": MockModelProvider(responder=triage_responder)},
)
result = await runtime.run(compiled, {"input": user_input})
```

`MockModelProvider` accepts an optional `responder`. The example's `triage_responder`
is a pure keyword rule, so identical inputs always produce identical replies and
usage. There is no network access, no API key lookup, and nothing that depends on the
current time.

## Inputs, branch behavior, and outputs

The run input is a JSON-compatible mapping containing one key:

- `input` (string): the inbound message to triage.

After `classify`, state holds `tone` and `draft_reply`. The conditional edge selects:

| Keyword signal in `input` | `tone` | Next node | Final `state["routed_to"]` |
| --- | --- | --- | --- |
| e.g. `thanks`, `great`, `works` | `positive` | `positive_reply` | `positive` |
| e.g. `broken`, `fail`, `crash` | `negative` | `negative_reply` | `negative` |
| neither set | `unclear` | `clarify_request` (fallback) | `clarify` |

Each branch node terminates the run. The returned `RunResult` has
`status == "succeeded"`, the merged `state`, the string `output` (prefixed with the
branch name), and deterministic metrics (`steps == 2`, `model_calls == 1`).

If a node raises, the runtime emits `node.failed` and returns a `RunResult` with
`status == "failed"` and the error message — see the failure-path test in
[`tests/unit/examples/test_custom_conditional_graph.py`](../tests/unit/examples/test_custom_conditional_graph.py).

## Run it

```bash
# all three branch samples (positive, negative, fallback)
uv run python examples/custom_conditional_graph.py

# your own input
uv run python examples/custom_conditional_graph.py --message "The report failed to export"
```

Expected output for a positive sample:

```text
input:    Great, the deploy works, thanks for the quick fix!
tone:     positive -> branch positive
output:   [positive branch] Acknowledged, thank you for the positive feedback!
steps:    2
```

## Adapting the pattern

- Add a route by adding a node and a conditional edge; keep exactly one fallback edge
  per source or let explicit predicates cover every case and finish via `NodeResult(finish=True)`.
- Nodes can also choose a destination themselves with `NodeResult(goto=...)`, but the
  target must be a declared edge or the runtime raises `GraphRoutingError`.
- Swap the in-memory stores for the SQLite stores (or your own `EventStore` /
  `CheckpointStore` implementations) without touching the graph definition.
- Replace the mock responder with an `OpenAICompatibleProvider` using the exact same
  node code; only the injected dependency changes.
