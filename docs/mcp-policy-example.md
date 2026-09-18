# Policy-wrapped MCP tools: runnable example

`examples/mcp_policy_demo/run_demo.py` is an executable, deterministic, fully offline
walkthrough of the only supported way to expose MCP tools to an agent: discovery through an
allowlist, namespaced tool names, and a `SecuredToolExecutor` boundary that applies
`SafetyPolicy` and one-shot `ApprovalGate` decisions before any call reaches the server.

Run it with:

```bash
uv sync --extra mcp
uv run python examples/mcp_policy_demo/run_demo.py
```

It uses a controlled local fixture (`examples/mcp_policy_demo/fixture_server.py`) launched
over real stdio through the production `stdio_mcp_executor`. No network access, credentials,
or external MCP server are involved, and the transcript is byte-for-byte reproducible. The
same wiring works against a real server; only the `command`/`args`/`env` change.

## What the example demonstrates

1. **Server allowlisting.** `stdio_mcp_executor(..., allow_tools={"weather", "write_report",
   "slow"})` filters discovery. The fixture also advertises `hidden_admin`, which never
   appears in the executor's schemas and can never be called through it.
2. **Namespaced discovery and calls.** Every tool is exposed as `fixture__<name>` and carries
   `x-mcp-server`/`x-mcp-tool` provenance, so two servers can never collide and a model can
   only address tools through their reviewed, namespaced names.
3. **Policy evaluation.** Each exposed name gets a `ToolSafetySpec`. Reads run directly;
   `fixture__write_report` is a `WRITE` side effect, so the default policy requires a
   one-shot human approval bound to the run, checkpoint revision, tool name, and exact
   arguments. A name with no registered metadata (`fixture__hidden_admin`) is denied even if
   it somehow reaches the boundary: default-deny does not depend on the allowlist.
4. **Normalized failures.** Every failure becomes a `ToolResult` with a stable
   `error_code`; transport internals never leak into content.

## Security boundaries

- The MCP server is **untrusted code in a child process**. The demo passes a minimal explicit
  environment (`FIXTURE_MODE` only) instead of inheriting the process environment, so secrets
  cannot leak into the server. The fixture refuses to start without its expected marker.
- The adapter is **not** an authorization boundary. `SecuredToolExecutor` is: policy is
  evaluated before execution, approvals are consumed atomically immediately before the side
  effect, and a consumed approval cannot be replayed (`approval_already_consumed`).
- Timeouts bound every discovery and call (`timeout_seconds`); a hung server degrades to a
  retryable `mcp_timeout` instead of blocking a run.
- Tool output is normalized into `ToolResult` (`content`, `data`, `receipt`) before it can
  enter model context or the event log. Receipts are deterministic digests scoped to the run,
  idempotency key, server, and tool.

## Failure mapping

| Scenario | `error_code` | Retryable | Source |
| --- | --- | --- | --- |
| Tool not in policy metadata | `policy_denied` | no | `SafetyPolicy` default-deny |
| Approval-gated call without approval | `approval_required` | no | `SecuredToolExecutor` |
| Reviewer rejected the request | `approval_denied` | no | `ApprovalGate` |
| Approval replayed after use | `approval_already_consumed` | no | `ApprovalGate` |
| Name passes policy but server does not expose it | `mcp_tool_not_found` | no | `MCPToolExecutor` |
| Call exceeds `timeout_seconds` | `mcp_timeout` | yes | `MCPToolExecutor` |
| Transport/SDK failure | `mcp_transport_error` (exception type only, no internals) | yes | `MCPToolExecutor` |
| Server returns `isError: true` | `mcp_tool_error` | no | `MCPToolExecutor` |
| `list_tools` response is not a tool list | `ValueError` at connect time (fail fast) | n/a | `MCPToolExecutor.connect` |
| Malformed call response (bad content blocks, non-dict structured data) | normalized to a safe `ToolResult`; garbage becomes `""`/`{"result": ...}` | n/a | `MCPToolExecutor` |

The demo prints one line per scenario; `tests/unit/test_mcp_policy_example.py` asserts each
mapping and runs the whole example offline as a smoke test.

## Configuring a real server

Install the optional transport (`uv sync --extra mcp` or `pip install
admires-ai-harness[mcp]`), then swap the fixture launch for the real, pinned server:

```python
async with stdio_mcp_executor(
    server_name="filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem@2026.1.0", "/data/reports"],
    env={"API_TOKEN": os.environ["REPORTS_API_TOKEN"]},  # inject secrets, never hardcode
    timeout_seconds=10.0,
    allow_tools=frozenset({"read_file", "list_directory"}),
) as mcp_executor:
    secured = SecuredToolExecutor(mcp_executor, policy=policy, approvals=approvals)
```

Rules that keep this safe:

- Pin the server package/version and treat the server as untrusted; run it inside the same
  container/VM isolation recommended for the harness itself.
- Pass a minimal explicit `env`. Read secrets from the environment or a secret manager at
  startup; never embed them in source, config checked into Git, or tool arguments.
- Keep `allow_tools` to the smallest useful set and register one `ToolSafetySpec` per
  exposed `server__tool` name with accurate side-effect and risk metadata.
- Never hand a raw `MCPToolExecutor` to a model-facing graph; always wrap it in
  `SecuredToolExecutor` so policy and approvals are enforced outside the model.
