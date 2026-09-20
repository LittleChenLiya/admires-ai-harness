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

### Discovery and naming

`MCPToolExecutor.connect()` normalizes both dict-shaped and pydantic-style SDK responses.
Every `inputSchema` is coerced into a deterministic JSON Schema object (missing or invalid
schemas become `{"type": "object", "properties": {}}`), tool names are exposed as
`<server>__<tool>` (disable with `prefix_names=False`), duplicates are rejected, and
`allow_tools` restricts which remote tools are ever exposed or callable.

### Results and failures

Text blocks and embedded text resources join the `ToolResult.content` channel; images,
audio, resource links, and annotated blocks are preserved losslessly as JSON-safe
`data["content_blocks"]`, and `structuredContent` lands in `data`. Unserializable
payloads degrade to `<unserializable:TypeName>` markers — remote object reprs never leak.
Failures stay distinguishable: `mcp_timeout` and `mcp_transport_error` are retryable,
`mcp_tool_error` (remote `isError`) and `mcp_tool_not_found` are not, and caller
cancellation always propagates as `CancelledError`. Transport errors are sanitized to the
exception type name only. Every call carries a deterministic `mcp:<digest>` receipt derived
from the run id, idempotency key, server, and tool name.

The core package imports without MCP installed; only `stdio_mcp_executor` requires the
optional `admires-ai-harness[mcp]` extra and raises `MCPDependencyError` otherwise.

## Add a grader

Prefer deterministic graders over model-judged grading: exact answer, required substring,
tool trajectory, artifact content, or policy outcome. When a model grader is unavoidable,
version its prompt and model and report it separately from deterministic scores.
