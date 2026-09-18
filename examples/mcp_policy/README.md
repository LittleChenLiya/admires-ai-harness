# Policy-wrapped MCP tool (offline example)

This example shows the complete, safe-by-default path for exposing MCP tools to an
agent: explicit server allowlisting, namespaced discovery, policy evaluation, one-shot
human approvals, and normalized failure codes. It runs entirely offline against a
controlled local stdio fixture — no network access, no credentials, and no real external
MCP server.

## Run it

```bash
uv sync --extra mcp
uv run python examples/mcp_policy/run_example.py
```

Expected output (error codes are part of the stable contract):

```text
Policy-wrapped MCP example (offline stdio fixture)
server: warehouse  discovered tools: warehouse__echo, warehouse__record, warehouse__fetch_url, warehouse__always_fails, warehouse__slow, warehouse__broken

[PASS] discovery is allowlisted and namespaced -> error_code=- retryable=False
[PASS] read tool succeeds without approval -> error_code=- retryable=False
[PASS] write requires approval before first call -> error_code=approval_required retryable=False
[PASS] write succeeds after bound approval -> error_code=- retryable=False
[PASS] approval cannot be replayed -> error_code=approval_already_consumed retryable=False
[PASS] network capability is denied by policy -> error_code=policy_denied retryable=False
[PASS] non-allowlisted server tool is denied -> error_code=policy_denied retryable=False
[PASS] unknown namespaced tool is denied -> error_code=policy_denied retryable=False
[PASS] adapter refuses an unexposed tool -> error_code=mcp_tool_not_found retryable=False
[PASS] remote tool error is normalized -> error_code=mcp_tool_error retryable=False
[PASS] timeout is normalized as retryable -> error_code=mcp_timeout retryable=True
[PASS] malformed response is normalized -> error_code=mcp_malformed_response retryable=False
```

## What the fixture is

`fake_mcp_server.py` speaks the minimum JSON-RPC/2.0 MCP stdio wire protocol by hand,
using only the Python standard library. It advertises seven tools; the client allowlists
six of them:

- `echo` — read-only success;
- `record` — a write side effect, used to show approval gating;
- `fetch_url` — a network side effect, denied by the default policy;
- `internal_only` — advertised but excluded by `allow_tools`;
- `always_fails` — a well-formed `isError: true` result;
- `slow` — sleeps past the client timeout (`MCP_FAKE_DELAY_SECONDS`);
- `broken` — returns a malformed `content` array.

The client side still uses the real MCP SDK through the production
`stdio_mcp_executor` adapter, so this is a real subprocess + real stdio transport, just
with a deterministic server at the other end.

## Security boundaries

The example keeps the production layering intact. From the model outward:

1. **Policy (`SafetyPolicy`)** is evaluated outside the model for every call. Tools
   without a matching `ToolSafetySpec` are denied; `network` capability is off by
   default; `write` needs an approval; `process` and `critical` stay disabled.
2. **Approval (`ApprovalGate`)** issues one-shot decisions bound to run ID, tool name,
   exact arguments digest, checkpoint revision, and a TTL. The demo consumes one
   approval and proves a second call with the same token fails with
   `approval_already_consumed`.
3. **Discovery allowlist (`allow_tools`)** filters tools during `tools/list`:
   `internal_only` is advertised by the untrusted server but never enters the schema
   list and cannot be called through the secured executor.
4. **Namespacing (`warehouse__*`)** prevents two servers from colliding on a tool name
   and makes provenance visible in every policy spec.
5. **Adapter (`MCPToolExecutor`)** never forwards an unexposed name and always returns a
   `ToolResult`; SDK exceptions never reach the model as raw text.
6. **Process boundary**: the child receives a scrubbed environment (`PATH`, locale,
   `TEMP`, and the fixture's own delay knob only) — no API keys, cloud credentials, or
   tokens. For a real deployment, additionally pin the server package and run it under
   OS-level isolation (a dedicated user, container, or VM). The Python policy is defense
   in depth, not a multi-tenant sandbox.

## Failure mapping

All transport and protocol failures collapse into stable, non-leaking `ToolResult`
codes so agents can branch deterministically:

| Situation                            | `error_code`             | `retryable` | Set by                |
| ------------------------------------ | ------------------------ | ----------- | --------------------- |
| Policy denies (unknown/unregistered) | `policy_denied`          | no          | `SecuredToolExecutor` |
| Write without approval               | `approval_required`      | no          | `SecuredToolExecutor` |
| Approval rejected/expired/replayed   | `approval_<code>`        | no          | `SecuredToolExecutor` |
| Exposed name missing on the executor | `mcp_tool_not_found`     | no          | `MCPToolExecutor`     |
| Remote returned `isError: true`      | `mcp_tool_error`         | no          | `MCPToolExecutor`     |
| Deadline exceeded                    | `mcp_timeout`            | yes         | `MCPToolExecutor`     |
| Connection/process failure           | `mcp_transport_error`    | yes         | `MCPToolExecutor`     |
| Missing or misshapen `content`       | `mcp_malformed_response` | no          | `MCPToolExecutor`     |

Transport errors include only the exception class name (never its message), so a remote
endpoint cannot bounce connection strings or secrets into model context. Malformed
responses fail closed instead of rendering as empty successful content.

## Configuring a real server

The fake fixture is swapped for a pinned real command; nothing in the wrapping changes.

```python
async with stdio_mcp_executor(
    server_name="warehouse",
    command="uvx",  # or an absolute, pinned executable path
    args=("--from", "warehouse-mcp==1.4.2", "warehouse-mcp", "--stdio"),
    env={"PATH": os.environ["PATH"], "WAREHOUSE_HOME": "/var/lib/warehouse-mcp"},
    cwd="/srv/harness-mcp",
    timeout_seconds=15,
    allow_tools=frozenset({"stock.lookup", "shipment.create"}),
) as executor:
    policy = SafetyPolicy(
        specs_for(executor.schemas()),
        allow_network=False,  # enable only after review; approval still applies
    )
    secured = SecuredToolExecutor(executor, policy=policy, approvals=ApprovalGate())
```

Rules to keep:

- **Pin the server** with `==`, a lockfile, or an immutable image, and verify its
  provenance.
- **Pass an explicit, minimal `env`**. Never forward the whole parent environment. Give
  a server its own credentials via an env map or secret store scoped to that server;
  never put tokens in command arguments, logs, or this repository.
- **Allowlist tools** with `allow_tools`, then register one `ToolSafetySpec` per exposed
  `server__tool` name, classifying side effects and risk conservatively.
- **Keep `SecuredToolExecutor`** as the only model-facing boundary; never hand the raw
  executor to an agent or graph node.
- **Isolate the process** on the host and review the server like third-party code,
  because it is untrusted code running locally.

## Files and tests

- `fake_mcp_server.py` — stdlib-only JSON-RPC stdio fixture.
- `run_example.py` — executable walkthrough using `stdio_mcp_executor`,
  `MCPToolExecutor`, `SafetyPolicy`, `ApprovalGate`, and `SecuredToolExecutor`.
- Adapter normalization: `tests/unit/test_mcp_adapter.py`.
- Policy/approval/allowlist cases: `tests/unit/test_mcp_policy_example.py`.
- Offline subprocess smoke test: `tests/integration/test_mcp_policy_example_smoke.py`.

See [docs/security.md](../../docs/security.md) for the full trust model and
[docs/extending.md](../../docs/extending.md) for MCP extension rules.
