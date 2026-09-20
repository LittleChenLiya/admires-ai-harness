# Architecture

Admire's AI Harness is deliberately built without an agent orchestration framework. The
goal is to make the control plane inspectable: graph compilation, scheduling, state
reduction, persistence, policy checks, and evaluation all live in this repository.

## System map

```mermaid
flowchart LR
  subgraph Clients
    CLI["CLI"]
    WEB["React console"]
    SDK["Python API"]
  end
  subgraph Control["Harness control plane"]
    API["FastAPI + SSE"]
    ORCH["Run service"]
    GRAPH["Compiled state graph"]
    POLICY["Policy + approval gate"]
    EVENTS["Event broker"]
  end
  subgraph Capabilities
    MODEL["Mock / OpenAI-compatible model"]
    TOOLS["Typed tools / MCP"]
    MEMORY["Short-term memory + local RAG"]
    CHILD["Child agent graphs"]
  end
  subgraph Durable["Durable state"]
    LOG[("Append-only events")]
    CP[("CAS checkpoints")]
    MEM[("Namespaced memory")]
    ART[("Workspace artifacts")]
  end

  CLI --> ORCH
  SDK --> ORCH
  WEB --> API --> ORCH --> GRAPH
  GRAPH --> POLICY --> TOOLS
  GRAPH --> MODEL
  GRAPH --> MEMORY
  GRAPH --> CHILD
  ORCH --> LOG
  ORCH --> CP
  MEMORY --> MEM
  TOOLS --> ART
  ORCH --> EVENTS --> API
```

## One execution language

Every client consumes the same versioned `RunEvent` envelope. The runtime appends events
before they are published live; a disconnected console can therefore reconnect using its
last sequence number. Observability is not an extra logging path—trace summaries and run
views are projections of the persisted event stream.

The append-only log is the historical source of truth. Checkpoints are restart indexes: they
capture state and the next node at a graph-node boundary. A compare-and-swap revision stops
two workers from advancing the same run simultaneously. Replaying a run is read-only and
never invokes a model or tool.

## Graph semantics

A `StateGraph` contains named asynchronous nodes and declared edges. `compile()` rejects an
unknown entry point, unknown edge target, duplicate fallback, and unreachable node before a
run starts. A node receives an immutable view of state and returns a `NodeResult`. In v1 the
runtime deep-copies the prior state, then applies the node update as a shallow top-level
overwrite. Conditional edges route against that new state. Applications that need nested
aggregation must compute it inside a node; pluggable reducers are outside the v1 contract.

Runs have explicit terminal states. Step, token, cost, repeated-action, and wall-clock budgets
are enforced by the runtime rather than delegated to a prompt. A child agent is an
independent run: it gets fresh run and trace IDs, carries `parent_run_id` back to its
parent, and can only reserve a budget within the parent's remaining budget. Its
dependency/capability map is explicitly delegated by the parent graph. The parent records an
`agent.delegated` event before the child starts and always pairs it with exactly one
`agent.joined` event, whether the child succeeds, fails, times out, or is cancelled. The
join folds the child's metered usage into the parent's totals exactly once, so a resumed
parent cannot spend the child's budget a second time. A child's terminal state never becomes
the parent's terminal state; its failure is data that the parent must handle explicitly.

## Side effects and recovery

The harness does not claim universal exactly-once execution. A tool call is at-least-once
across process failure and carries a stable idempotency key. The built-in registry caches
receipts only in process; a production tool must durably store its own receipt if it needs
crash-safe deduplication. Read-only replay is always safe.

An approval is bound to the run, tool arguments digest, checkpoint revision, expiry, and a
single consumption. This prevents approval replay after state or arguments change.

## Dependency boundaries

The runtime depends only on small `Protocol` ports. SQLite, in-memory stores, model providers,
MCP, and FastAPI adapt to those ports. The deterministic mock provider is a first-class path,
not a test stub hidden behind the real service. That keeps every reference workflow and the
entire evaluation suite runnable without a network or paid credential.
