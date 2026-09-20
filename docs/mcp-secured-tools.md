# Policy-wrapped MCP tools

This guide walks through `examples/mcp_secured_stdio/`, an executable, fully offline
demo of exposing MCP tools to the harness without weakening the default-deny policy or
the approval boundary. It runs deterministically against a controlled local fixture
server: no network, no credentials, and no real external MCP server are involved.

Run it:

```bash
uv run python examples/mcp_secured_stdio/demo.py
```

Expected output ends with `DEMO RESULT: 11/11 scenarios matched expectations` and exit
code 0. The same scenarios are covered by `tests/integration/test_mcp_secured_demo.py`.

## What the example contains

| File | Role |
| --- | --- |
| `examples/mcp_secured_stdio/fake_server.py` | Deterministic stdio JSON-RPC fixture server (stdlib only). Advertises read, write, slow, failing, malformed, and dangerous tools. |
| `examples/mcp_secured_stdio/stdio_client.py` | Minimal stdio client implementing the structural `MCPClient` protocol from `admires_harness.adapters.mcp`. A stand-in for the official SDK so the demo needs no optional dependencies. |
| `examples/mcp_secured_stdio/demo.py` | Wires the production `MCPToolExecutor`, `SafetyPolicy`, `ApprovalGate`, and `SecuredToolExecutor` together and runs the scenarios below. |

The demo uses the same public adapter and policy contracts as production code. Nothing
in the example relaxes policy defaults, bypasses `SecuredToolExecutor`, or disables
approval checks.

## Security boundaries the demo enforces

- **Server allowlist.** `MCPToolExecutor.connect(..., allow_tools=...)` imports only
  the listed remote tools. The fixture advertises `admin_wipe`; it is never exposed.
- **Namespaced discovery.** Every exposed tool is prefixed (`demo__echo`), so tools
  from different servers cannot collide and policy metadata is unambiguous.
- **Policy before execution.** The executor is wrapped in `SecuredToolExecutor`. A
  tool with no registered `ToolSafetySpec` is denied (`policy_denied`); the network
  capability stays disabled (`allow_network=False` default), so a network spec is
  denied too.
- **One-shot approvals.** Writes require an approval bound to the run, checkpoint
  revision, tool name, and exact arguments. Replaying a consumed approval fails closed
  (`approval_already_consumed`).
- **Untrusted server.** The MCP server is an untrusted subprocess. Timeouts,
  transport failures, server-reported errors, and malformed results are normalized
  into stable `ToolResult` error codes instead of propagating exceptions or raw
  server details.

## Failure mapping

| Condition | `error_code` | Retryable |
| --- | --- | --- |
| Tool not registered in policy / capability disabled | `policy_denied` | no |
| Side-effecting call without a consumed approval | `approval_required` | no |
| Replayed, expired, or mis-bound approval | `approval_<code>` (e.g. `approval_already_consumed`) | no |
| Namespaced name not exposed by the server | `mcp_tool_not_found` | no |
| Server exceeded `timeout_seconds` | `mcp_timeout` | yes |
| Transport/subprocess failure | `mcp_transport_error` (message carries only the exception type) | yes |
| Server returned `isError: true` | `mcp_tool_error` | no |
| Malformed result shape | normalized: non-list content becomes empty text, non-mapping structured content is wrapped as `{"result": ...}` | n/a |

## Connecting a real server

Install the optional dependency (`pip install admires-ai-harness[mcp]`) and use the
production stdio bridge, keeping the same policy wrapping:

```python
from admires_harness.adapters.mcp import stdio_mcp_executor
from admires_harness.contracts import SideEffect
from admires_harness.safety import (
    ApprovalGate,
    SecuredToolExecutor,
    SafetyPolicy,
    ToolSafetySpec,
)

async with stdio_mcp_executor(
    server_name="files",
    command="uvx",
    args=["--from", "mcp-server-filesystem==0.4.1", "mcp-server-filesystem", "/data"],
    env={"PATH": "/usr/bin:/bin"},  # minimal environment, never os.environ
    timeout_seconds=30.0,
    allow_tools=frozenset({"read_file", "list_directory"}),
) as executor:
    policy = SafetyPolicy(
        [
            ToolSafetySpec(name="files__read_file", side_effect=SideEffect.READ),
            ToolSafetySpec(name="files__list_directory", side_effect=SideEffect.READ),
        ]
    )
    secured = SecuredToolExecutor(executor, policy=policy, approvals=ApprovalGate())
```

Operational rules:

- Pin the server package version and verify its provenance; the server process runs
  with your user's privileges unless you add OS-level isolation.
- Pass a minimal explicit `env`, never the whole process environment. Keep secrets
  out of config files; if a server genuinely needs a credential, inject it through a
  secret manager or environment variable reference resolved at deploy time.
- Register a `ToolSafetySpec` for every namespaced tool before use; undeclared tools
  are denied by default.
- Keep `SecuredToolExecutor` on the boundary. Do not hand a raw MCP executor to a
  model-facing agent, and do not move permission checks into prompts.
- The reference workflows deliberately do not auto-wire MCP; connecting a server is
  always an explicit, audited configuration step.
