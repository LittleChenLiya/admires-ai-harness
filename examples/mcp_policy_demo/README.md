# MCP policy demo (offline)

Executable walkthrough of policy-wrapped MCP tools: server allowlist, namespaced
discovery and calls, policy evaluation with one-shot approvals, and normalized
denied/unknown/timeout/malformed failures. Runs deterministically with no network,
no credentials, and no real MCP server.

```bash
python examples/mcp_policy_demo/demo.py
```

See [docs/mcp-policy-demo.md](../../docs/mcp-policy-demo.md) for the full guide,
including how to configure a real MCP server without embedding secrets.
