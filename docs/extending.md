# Extending the harness

Extensions should add capability without bypassing execution policy or creating a second
event language.

## Add a graph

1. Implement small async nodes that accept `NodeContext`.
2. Return a `NodeResult`; never mutate `context.state`.
3. Declare every possible destination as an edge.
4. Compile during application startup so structural errors fail early.
5. Add an offline deterministic end-to-end test and at least one failure-path test.

## Add a model provider

Implement `ModelProvider.complete()` and `stream()`. Normalize response content, tool calls,
usage, and retryability into the shared contracts. Fail fast when required configuration is
absent. Never expose authorization headers or raw provider responses in user-facing errors.

## Add a tool

A tool must have a unique name, JSON schema, side-effect category, risk level, output limit,
and an async handler returning `ToolResult`. A side-effecting tool must accept the runtime's
idempotency key and return a receipt. The registry applies policy before invoking the handler;
do not perform permission checks only inside prompts or UI code.

The registry deduplicates side-effecting calls per `(run_id, idempotency_key)`: concurrent
duplicates share a single in-flight handler invocation, and the first successful result is
cached as the receipt. Replays with identical arguments (JSON key order is canonicalized)
return the cached result; replays with different arguments fail with `idempotency_conflict`.
Handler failures and exceptions are never cached, so a retry after a failure re-invokes the
handler. Pure and read-only tools bypass receipts entirely and are never cached. Handlers
must therefore be safe to retry after failure and must not assume a successful result will
ever be recomputed within the same run and key.

Write tests for schema rejection, policy denial, handler failure, output bounds, and repeated
idempotency keys, including concurrent duplicates of the same key. Filesystem tools
additionally require traversal and symlink tests.

## Add an MCP server

MCP is an adapter, not a trusted tunnel. Configure the server explicitly, import its tool
schemas, and wrap the resulting executor in `SecuredToolExecutor` with matching safety specs
before injecting it into a graph. The three reference workflows deliberately do not auto-wire
MCP. Pin server packages and do not pass the entire process environment to an untrusted server.

## Add a grader

Prefer deterministic graders over model-judged grading: exact answer, required substring,
tool trajectory, artifact content, or policy outcome. When a model grader is unavoidable,
version its prompt and model and report it separately from deterministic scores.
