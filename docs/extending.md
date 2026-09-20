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

The built-in `OpenAICompatibleProvider` shows the expected normalization discipline; reuse
`StreamAccumulator` and the `_normalize_*` helpers in `adapters/models.py` where possible:

- Content may be missing, a plain string, an object, or a list of parts; all forms normalize
  to one string.
- Tool-call arguments must parse to a JSON object. Streamed argument fragments are
  concatenated per tool-call index and validated once at finalize; anything else raises
  `ModelProviderError` instead of reaching the runtime.
- Usage, finish reason, and model name are last-wins. Usage fields merge key-wise so
  cumulative or repeated usage events never double count tokens or cost, and missing usage
  degrades to zeroes.
- Byte-identical stream chunks (proxy or retry replays) are dropped; empty chunks are
  tolerated. `chunks_seen`/`chunks_merged` on the accumulator expose merge statistics.
- Provider failures become `ModelProviderError` with `status_code` and `retryable`
  classification. Timeouts and transport failures are retryable; error details that contain
  user prompts or credentials are replaced with a generic message.

Test providers deterministically with `httpx.MockTransport` or `MockModelProvider`; adapter
tests must never require real network access.

## Add a tool

A tool must have a unique name, JSON schema, side-effect category, risk level, output limit,
and an async handler returning `ToolResult`. A side-effecting tool must accept the runtime's
idempotency key and return a receipt. The registry applies policy before invoking the handler;
do not perform permission checks only inside prompts or UI code.

Write tests for schema rejection, policy denial, handler failure, output bounds, and repeated
idempotency keys. Filesystem tools additionally require traversal and symlink tests.

## Add an MCP server

MCP is an adapter, not a trusted tunnel. Configure the server explicitly, import its tool
schemas, and wrap the resulting executor in `SecuredToolExecutor` with matching safety specs
before injecting it into a graph. The three reference workflows deliberately do not auto-wire
MCP. Pin server packages and do not pass the entire process environment to an untrusted server.

## Add a grader

Prefer deterministic graders over model-judged grading: exact answer, required substring,
tool trajectory, artifact content, or policy outcome. When a model grader is unavoidable,
version its prompt and model and report it separately from deterministic scores.
