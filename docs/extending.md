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

Write tests for schema rejection, policy denial, handler failure, output bounds, and repeated
idempotency keys. Filesystem tools additionally require traversal and symlink tests.

## Add an MCP server

MCP is an adapter, not a trusted tunnel. Configure the server explicitly, import its tool
schemas, and wrap the resulting executor in `SecuredToolExecutor` with matching safety specs
before injecting it into a graph. The three reference workflows deliberately do not auto-wire
MCP. Pin server packages and do not pass the entire process environment to an untrusted server.

The adapter normalizes both dict-shaped and object-shaped SDK responses identically:

- Tool discovery defaults missing or invalid `inputSchema` values to `{"type": "object"}`,
  prefixes exposed names with the server name, enforces the `allow_tools` set, and rejects
  duplicate exposed names instead of silently overwriting them.
- Results concatenate text blocks into `ToolResult.content`, keep image, audio, resource,
  and resource-link blocks as JSON-safe mappings in `data["content"]`, and merge
  `structuredContent` into `data` (scalar structured values are wrapped under `"result"`).
  Bytes are base64-encoded and non-serializable values become typed placeholders, so no remote
  object `repr` ever leaks into a result.
- Failures stay distinguishable: `mcp_tool_not_found` (not retryable), `mcp_timeout` and
  `mcp_transport_error` (retryable, transport messages are reduced to the exception type
  name), and `mcp_tool_error` for server-side `isError` responses. Caller cancellation
  propagates as `CancelledError` and is never converted into a tool failure. Every remote
  call carries a deterministic `mcp:<sha256>` receipt derived from the run id and
  idempotency key.

## Add a grader

Prefer deterministic graders over model-judged grading: exact answer, required substring,
tool trajectory, artifact content, or policy outcome. When a model grader is unavoidable,
version its prompt and model and report it separately from deterministic scores.
