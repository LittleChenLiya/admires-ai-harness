# Secured MCP stdio demo

Offline, deterministic demo of policy-wrapped MCP tools. Runs against a controlled
local fixture server with no network, no credentials, and no real MCP server.

```bash
uv run python examples/mcp_secured_stdio/demo.py
```

See [docs/mcp-secured-tools.md](../../docs/mcp-secured-tools.md) for the full guide,
security boundaries, and failure mapping.
