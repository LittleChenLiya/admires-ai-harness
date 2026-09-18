# Custom graph walkthrough

This guide builds a small **support-ticket triager** that takes a real conditional
branch at runtime. The complete example lives in
[`examples/custom_graph_triage.py`](../examples/custom_graph_triage.py) and runs offline
with the deterministic mock model: no network, API key, clock, or randomness is involved.

## The runtime contracts

Everything below is public API, exported from `admires_harness.runtime`:

- [`StateGraph`](../src/admires_harness/runtime/graph.py) is the fluent builder. Edges
  carry an optional `condition` predicate; the edge without a condition is the fallback.
- [`CompiledGraph`](../src/admires_harness/runtime/graph.py) is the validated, immutable
  result of `compile()`. Compilation rejects empty graphs, missing entry points, unknown
  edge targets, duplicate nodes/fallback edges, and unreachable nodes.
- [`NodeContext`](../src/admires_harness/runtime/types.py) is the capability-limited
  argument of every node: read-only `state`, injected `dependencies`, `emit()` for
  domain events, and `delegate()` for child graphs. Nodes never mutate state directly.
- [`NodeResult`](../src/admires_harness/runtime/types.py) is the only mutation/routing
  surface: `update` merges new keys into state, `finish=True` ends the run, and `goto`
  selects a declared destination explicitly.
- [`GraphRuntime`](../src/admires_harness/runtime/engine.py) drives a compiled graph with
  events, checkpoints, and budgets. It contains no model implementation; capabilities are
  injected through `dependencies`.
- `InMemoryEventStore` and `InMemoryCheckpointStore`
  ([`runtime/stores.py`](../src/admires_harness/runtime/stores.py)) are the public
  reference stores, suitable for examples, tests, and fully in-process execution.
- [`MockModelProvider`](../src/admires_harness/adapters/models.py) is the deterministic
  offline model. Pass a `responder` callable for rule-based replies; its output never
  depends on network or wall-clock time.

## Inputs

A run starts from a plain JSON-compatible mapping. The triager takes one key:

```python
await runtime.run(graph.compile(), {"input": "[T-1] Production checkout is down ..."})
```

The `MockModelProvider` is injected as the `"model"` dependency, exactly as production
graphs inject a provider. Our responder parses the ticket text for deterministic keyword
markers (`outage`, `down`, `breach`, ...) and returns a synthetic `ModelResponse`:

```python
MockModelProvider(responder=triage_responder)
```

## Defining the nodes and the branch

Each node is an async callable taking a `NodeContext` and returning a `NodeResult`. The
`triage` node calls the model and writes `priority` into state:

```python
async def triage(context: NodeContext) -> NodeResult:
    response = await model.complete([...])
    return NodeResult(update={"priority": ..., "triage_reason": ...}, model_calls=1)
```

The branch is declared as **edges**, not inside a node. Predicates receive the state as
it exists *after* the node's update is merged. The first matching conditional edge wins;
the edge without a condition is the fallback:

```python
graph = (
    StateGraph("support-triage", version="1")
    .add_node("triage", triage)
    .add_node("escalate", escalate)
    .add_node("draft_reply", draft_reply)
    .set_entry_point("triage")
    .add_edge("triage", "escalate", condition=is_urgent, label="urgent")
    .add_edge("triage", "draft_reply", label="routine")  # fallback
    .add_edge("escalate", END)
    .add_edge("draft_reply", END)
)
```

`is_urgent(state)` is a plain `bool` predicate (`async` predicates are also supported):

```python
from collections.abc import Mapping

from admires_harness.contracts import JsonValue


def is_urgent(state: Mapping[str, JsonValue]) -> bool:
    return state.get("priority") == "urgent"
```

`END` is the public terminal sentinel exported from `admires_harness.runtime`.

## Branch behavior

After `triage` completes, `CompiledGraph.next_node()` evaluates the edges in order:

| `priority` in state | Selected edge  | Next node     | Outcome                                  |
| ------------------- | -------------- | ------------- | ---------------------------------------- |
| `"urgent"`          | `is_urgent`    | `escalate`    | pages on-call and finishes               |
| anything else       | fallback edge  | `draft_reply` | drafts a routine self-service reply      |

The predicate receives a deep-copied, read-only snapshot, so it cannot mutate live runtime
state. A node may instead override edges with `NodeResult(goto=...)`, but the target must
be a declared edge or the runtime raises `GraphRoutingError`.

## Outputs

Terminal nodes set `finish=True` and a JSON-compatible `output`. `GraphRuntime.run()`
returns a `RunResult` with the final `state`, `output`, `status`, and deterministic
counters (`steps`, `model_calls`, tokens, cost). Every transition is also recorded as a
`RunEvent` in the event store, so the branch taken is inspectable after the fact via
`runtime.events(run_id)` or `runtime.replay(run_id)`.

## Run it

From the repository root (no environment variables required):

```bash
uv run python examples/custom_graph_triage.py
```

Expected deterministic output (exact text, no timestamps):

```text
succeeded route=escalate priority=urgent  model_calls=1
  ESCALATED (urgent): [T-1] Production checkout is down for every customer in EU-West.
succeeded route=reply    priority=routine model_calls=1
  ROUTINE REPLY: Thanks, the team will answer: [T-2] Question: how do I export my runs to a CSV file? (reason: is a how-to or documentation question)
```

The script also works with the installed package using plain
`python examples/custom_graph_triage.py`. Automated coverage for both branches and the
invalid-definition/failure paths lives in
[`tests/unit/runtime/test_custom_graph_example.py`](../tests/unit/runtime/test_custom_graph_example.py).
See [`docs/extending.md`](extending.md) for the general graph-extension checklist.
