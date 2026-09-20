"""Deterministic stdio MCP fixture server for the secured-tools demo.

This is a controlled local stand-in for a real MCP server. It speaks newline-delimited
JSON-RPC 2.0 over stdio (the MCP stdio framing) using only the standard library, so the
demo and its tests run offline with no credentials and no real external server. It is a
test fixture, not a production server.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

PROTOCOL_VERSION = "2025-06-18"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "echo",
        "description": "Echo text back. Read-only.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "write_note",
        "description": "Store a note. Side-effecting write.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "slow",
        "description": "Never answers within the demo timeout.",
        "inputSchema": {"type": "object"},
    },
    {
        "name": "fail",
        "description": "Always reports a tool-level error.",
        "inputSchema": {"type": "object"},
    },
    {
        "name": "malformed",
        "description": "Returns a result with an unexpected shape.",
        "inputSchema": {"type": "object"},
    },
    {
        "name": "admin_wipe",
        "description": "Dangerous tool the demo allowlist must hide.",
        "inputSchema": {"type": "object"},
    },
]


async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "echo":
        text = str(arguments.get("text", ""))
        return {
            "content": [{"type": "text", "text": f"echo: {text}"}],
            "structuredContent": {"echo": text},
            "isError": False,
        }
    if name == "write_note":
        text = str(arguments.get("text", ""))
        return {
            "content": [{"type": "text", "text": f"stored note ({len(text)} chars)"}],
            "structuredContent": {"stored": True},
            "isError": False,
        }
    if name == "slow":
        await asyncio.sleep(30)
        return {"content": [{"type": "text", "text": "too late"}], "isError": False}
    if name == "fail":
        return {
            "content": [{"type": "text", "text": "note store is read-only today"}],
            "isError": True,
        }
    if name == "malformed":
        # Deliberately off-shape: content should be a list of blocks and
        # structuredContent should be a mapping. The adapter must normalize both.
        return {
            "content": "unexpected-scalar",
            "structuredContent": ["not", "a", "mapping"],
            "isError": False,
        }
    if name == "admin_wipe":
        return {"content": [{"type": "text", "text": "wiped"}], "isError": False}
    raise ValueError(f"unknown tool: {name}")


async def dispatch(request: dict[str, Any]) -> tuple[Any, bool]:
    """Return (result, respond). Notifications get no response."""

    method = request.get("method")
    params = request.get("params") or {}
    if method == "initialize":
        return (
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-stdio-fixture", "version": "1.0.0"},
            },
            True,
        )
    if method == "notifications/initialized":
        return None, False
    if method == "tools/list":
        return {"tools": TOOLS}, True
    if method == "tools/call":
        return await call_tool(str(params.get("name", "")), params.get("arguments") or {}), True
    raise ValueError(f"unsupported method: {method}")


async def handle(line: bytes, write_lock: asyncio.Lock) -> None:
    writer = sys.stdout.buffer
    request: dict[str, Any] = {}
    try:
        request = json.loads(line)
        result, respond = await dispatch(request)
    except Exception as exc:
        request_id = request.get("id")
        if request_id is None:
            return
        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32603, "message": str(exc)},
        }
    else:
        if not respond:
            return
        message = {"jsonrpc": "2.0", "id": request.get("id"), "result": result}
    async with write_lock:
        writer.write(json.dumps(message).encode() + b"\n")
        writer.flush()


async def serve() -> None:
    write_lock = asyncio.Lock()
    pending: set[asyncio.Task[None]] = set()
    while True:
        # to_thread keeps this portable: asyncio cannot always connect pipes to
        # stdin on every platform/event loop.
        line = await asyncio.to_thread(sys.stdin.buffer.readline)
        if not line:
            for task in pending:
                task.cancel()
            return
        # Handle requests concurrently so one slow tool cannot block the others.
        task = asyncio.create_task(handle(line, write_lock))
        pending.add(task)
        task.add_done_callback(pending.discard)


if __name__ == "__main__":
    asyncio.run(serve())
