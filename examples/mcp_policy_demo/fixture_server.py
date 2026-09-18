"""Controlled local MCP stdio fixture server for the offline policy demo.

This process is a *fixture*, not a real integration. It speaks MCP over stdio using
the official SDK so the demo exercises the same ``stdio_mcp_executor`` production
path a real server would use, without network access, credentials, or side effects.
It requires the optional ``mcp`` extra (``uv sync --extra mcp``).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

if os.environ.get("FIXTURE_MODE") != "offline-demo":
    raise SystemExit("fixture server must be launched by examples/mcp_policy_demo/run_demo.py")

server: Server[Any] = Server("admires-fixture")

_WEATHER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}
_REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"title": {"type": "string"}},
    "required": ["title"],
    "additionalProperties": False,
}
_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "additionalProperties": False}


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="weather",
            description="Read-only fixture weather lookup.",
            inputSchema=_WEATHER_SCHEMA,
        ),
        types.Tool(
            name="write_report",
            description="Fixture write tool used to demonstrate the approval boundary.",
            inputSchema=_REPORT_SCHEMA,
        ),
        types.Tool(
            name="slow",
            description="Sleeps so the demo can show timeout normalization.",
            inputSchema=_EMPTY_SCHEMA,
        ),
        types.Tool(
            name="hidden_admin",
            description="Never allowlisted; the demo proves it stays unreachable.",
            inputSchema=_EMPTY_SCHEMA,
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    if name == "weather":
        city = str(arguments.get("city", ""))
        return [types.TextContent(type="text", text=f"sunny in {city}")]
    if name == "write_report":
        title = str(arguments.get("title", ""))
        return [types.TextContent(type="text", text=f"report stored: {title}")]
    if name == "slow":
        await asyncio.sleep(30)
        return [types.TextContent(type="text", text="slow result")]
    if name == "hidden_admin":
        return [types.TextContent(type="text", text="you should never see this")]
    raise ValueError(f"unknown fixture tool: {name}")


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
            raise_exceptions=False,
        )


if __name__ == "__main__":
    asyncio.run(main())
