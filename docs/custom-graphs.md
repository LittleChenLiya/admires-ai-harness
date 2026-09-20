# Custom graphs

This guide shows how to define, compile, and run your own conditional graph with the
deterministic mock model. The companion example
[examples/custom_graph.py](../examples/custom_graph.py) is executable offline: no network,
credentials, or wall-clock dependence.

## Runtime contracts

The example only uses the public runtime surface:

- [StateGraph / CompiledGraph](../src/admires_harness/runtime/graph.py) — the fluent builder
  and the validated, immutable graph produced by ``compile()``.
- [NodeContext / NodeResult / END](../src/admires_harness/runtime/types.py) — the node
  signature, the sole mutation/routing surface, and the terminal edge target.
- [GraphRuntime](../src/admires_harness/runtime/engine.py) — the event-sourced executor,
  driven here with ``InMemoryEventStore`` and ``InMemoryCheckpointStore``.
- [MockModelProvider](../src/admires_harness/adapters/models.py) — the deterministic offline
  model; a rule-based ``responder`` replaces any network call.

See [docs/architecture.md](architecture.md) for state, event, and recovery semantics, and
[docs/extending.md](extending.md) for the general extension checklist.

## The example graph

```
classify ──(category == "technical")──▶ technical ──▶ END
    │
    └──(fallback)────────────────────▶ general ────▶ END
```

1. **Define nodes.** Each node is an async callable that receives a ``NodeContext`` and
   returns a ``NodeResult``. Nodes read state through ``context.state`` (read-only),
   reach injected capabilities through ``context.dependencies``, and merge changes back
   with ``NodeResult(update={...})``. They never mutate state in place.
2. **Declare routing.** ``add_edge("classify", "technical", condition=...)`` adds a
   conditional edge; the predicate receives a read-only snapshot of the state. The edge
   without a condition is the fallback and is taken only when no condition matches. Every
   possible destination must be a declared edge — ``compile()`` rejects unknown targets,
   duplicate fallbacks, and unreachable nodes with ``GraphDefinitionError``.
3. **Compile once.** ``build_graph()`` compiles at startup so structural errors fail
   early, before any run begins.
4. **Run.** ``GraphRuntime.run(graph, {"question": ...})`` executes the graph and
   returns a ``RunResult`` with status, final state, output, and usage metrics.

## Input

The initial state is a JSON-compatible mapping: ``{"question": "..."}``. The
``classify`` node asks the mock model for a category; the rule-based responder answers
``"technical"`` when the question mentions a keyword such as ``python``, ``error``,
or ``bug``, and ``"general"`` otherwise.

## Branch behavior

After ``classify`` merges ``{"category": ...}`` into the state, the runtime evaluates
the conditional edge. A technical question routes to the ``technical`` node; anything else
takes the fallback to ``general``. Each branch node calls the mock model once more and
finishes the run with ``NodeResult(output=..., finish=True)``. Every run takes exactly two
steps: classify, then one specialist answer.

## Output

``RunResult`` reports ``status`` (``succeeded``), the final ``state`` (including
``category``), the branch's ``output`` text, and deterministic metrics
(``steps``, ``model_calls``, token usage). The CLI prints a short summary and exits
non-zero if the run did not succeed.

## Run it

```bash
uv run python examples/custom_graph.py
uv run python examples/custom_graph.py --question "What is the meaning of life?"
```

Expected output for the default technical question:

```
status:   succeeded
category: technical
steps:    2
output:   Technical answer: reproduce, isolate the failing line, then patch it.
```

The same graph and both branches are covered by automated tests in
[tests/unit/test_custom_graph_example.py](../tests/unit/test_custom_graph_example.py),
including invalid-definition and model-failure paths.
