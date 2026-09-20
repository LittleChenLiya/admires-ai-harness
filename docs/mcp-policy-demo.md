# Policy-wrapped MCP tools: offline demo and guide

This guide walks through `examples/mcp_policy_demo/`, an executable example that shows
how to expose Model Context Protocol (MCP) tools to an agent **through the harness
security boundary** instead of around it. It runs fully offline: no network, no
credentials, and no real MCP server. A controlled local fixture
(`examples/mcp_policy_demo/fake_server.py`) speaks framed JSON-RPC over stdio so every
run is deterministic.

Run it from the repository root:

```bash
python examples/mcp_policy_demo/demo.py
```

## What the demo shows

1. **Server allowlist at discovery.** `MCPToolExecutor.connect(..., allow_tools=...)`
   imports only the named remote tools. The fixture server advertises a `hidden` tool
   that never becomes callable because it is not on the allowlist.
2. **Namespaced tool names.** Every imported tool is exposed as `<server>__<tool>`
   (for example `demo__echo`), so tools from two servers cannot collide and a prompt
   cannot confuse a remote tool with a native one. Calls are mapped back to the remote
   name only inside the adapter.
3. **Policy evaluation before any transport call.** The executor is wrapped in
   `SecuredToolExecutor` with a `SafetyPolicy` that carries one `ToolSafetySpec` per
   exposed tool. A call to a tool with no registered spec (`demo__hidden`) fails closed
   with `policy_denied` and never touches the server.
4. **One-shot approvals for writes.** `demo__save_note` is declared
   `SideEffect.WRITE`, so the default policy requires human approval. The demo shows
   `approval_required`, an approved call that executes, and a reviewer-denied call
   (`approval_denied`). Approvals are bound to the run, checkpoint revision, tool name,
   and canonical arguments, and are consumed exactly once.
5. **Normalized failures.** Unknown tool, timeout, and malformed-transport responses
   all come back as ordinary `ToolResult` values with stable `error_code`s instead of
   exceptions (see the mapping table below).

## Security boundaries

- The demo uses the **same public contracts as production code**:
  `MCPToolExecutor`, `SafetyPolicy`/`ToolSafetySpec`, `ApprovalGate`, and
  `SecuredToolExecutor`. Nothing is monkey-patched and no default is weakened:
  unregistered tools are denied, write/process/network capabilities keep their default
  approval gates, and approvals remain one-shot.
- The MCP server is **untrusted input**. The allowlist, the namespace prefix, and the
  policy wrapper are the boundary; the model-facing agent only ever sees the secured
  executor. Do not hand a raw `MCPToolExecutor` to an agent.
- The fixture client (`examples/mcp_policy_demo/stdio_client.py`) exists only so the
  demo runs with zero optional dependencies. It satisfies the structural `MCPClient`
  protocol consumed by the adapter. It is not a general MCP client.

## Failure mapping

| Scenario | `error_code` | Retryable | Source |
| --- | --- | --- | --- |
| Tool has no registered safety spec / capability disabled | `policy_denied` | no | `SafetyPolicy` |
| Side-effecting call without an approval | `approval_required` | no | `SecuredToolExecutor` |
| Reviewer rejected the request | `approval_denied` | no | `ApprovalGate` |
| Approval expired, replayed, or bound to different arguments | `approval_<code>` | no | `ApprovalGate` |
| Tool not exposed by the server allowlist | `mcp_tool_not_found` | no | `MCPToolExecutor` |
| Server exceeded `timeout_seconds` | `mcp_timeout` | yes | `MCPToolExecutor` |
| Malformed frame or transport failure | `mcp_transport_error` | yes | `MCPToolExecutor` |
| Server returned `isError: true` | `mcp_tool_error` | no | `MCPToolExecutor` |

All failures are `ToolResult(ok=False, ...)` values, so graphs can branch on
`error_code` without exception handling, and the event log records a uniform shape.

## Configuring a real MCP server

The production path uses the optional official MCP SDK:

```bash
pip install "admires-ai-harness[mcp]"
```

```python
import os

from admires_harness.adapters.mcp import stdio_mcp_executor
from admires_harness.contracts import SideEffect
from admires_harness.safety import ApprovalGate, SafetyPolicy, SecuredToolExecutor, ToolSafetySpec

async with stdio_mcp_executor(
    server_name="files",
    command=os.environ["FILES_MCP_COMMAND"],  # e.g. "uvx", configured per deployment
    args=["--from", "files-mcp==1.2.3", "files-mcp"],  # pin the server package
    allow_tools=frozenset({"read_file"}),
    timeout_seconds=10.0,
) as executor:
    policy = SafetyPolicy(
        specs=(ToolSafetySpec(name="files__read_file", side_effect=SideEffect.READ),)
    )
    secured = SecuredToolExecutor(executor, policy=policy, approvals=ApprovalGate())
```

Guidelines for real deployments:

- **Pin the server package and version**; treat upgrades as code changes.
- **Keep the allowlist minimal** and register one `ToolSafetySpec` per exposed
  namespaced tool with an honest side-effect category and risk level.
- **Never embed secrets in the repo or in tool arguments.** Read server commands and
  tokens from the environment or a secret manager at runtime, and pass only the
  environment variables the server actually needs via the `env` parameter instead of
  inheriting the whole process environment.
- **Run the server process under OS-level isolation** (container/VM) for untrusted
  servers; the harness policy boundary is not a sandbox for the server process itself.

## Tests

- `tests/unit/test_mcp_policy_wrapper.py` covers allowlist filtering, namespacing,
  policy denial, approval require/consume/deny, and the normalized unknown, timeout,
  and malformed-response failures using an in-process fake client.
- `tests/integration/test_mcp_demo_smoke.py` runs the example end to end offline and
  asserts every scenario marker appears in its output.
