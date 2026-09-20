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

Implement `ModelProvider.complete()` and `stream()`. Normalize provider output into the
shared `ModelResponse`/`ToolCall`/`ModelUsage` contracts:

- `content` may be missing, `None`, a plain string, or a list of text parts; collapse all
  forms to a single string.
- Streamed tool calls may arrive fragmented across chunks; merge fragments by chunk `index`,
  concatenate the JSON argument strings, and reject arguments that are not a JSON object.
- Treat `usage` as a latest-snapshot-wins value so repeated frames never double count tokens
  or cost; missing or malformed fields become zero.
- Convert provider exceptions to `ModelProviderError` with sanitized messages that never
  include credentials or request payloads, and mark timeouts plus 429/5xx responses retryable.

`StreamAccumulator` in `adapters/models.py` already implements this merge for
chat-completions streams (including empty keep-alive chunks and replayed duplicates); reuse
it instead of re-implementing chunk merging. Fail fast when required configuration is absent.
Never expose authorization headers or raw provider responses in user-facing errors.

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
