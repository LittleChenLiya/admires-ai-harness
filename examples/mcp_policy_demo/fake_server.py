"""Deterministic offline fake MCP server speaking framed JSON-RPC over stdio.

This fixture lets the policy-wrapping demo and its smoke test run with no
network, no credentials, and no real MCP server. It mirrors the stdio transport
framing (Content-Length headers) used by MCP, but it is intentionally not a
general-purpose server: every behavior is fixed so demo output is reproducible.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, BinaryIO

SERVER_INFO = {"name": "fake-mcp", "version": "0.1.0"}
PROTOCOL_VERSION = "2025-06-18"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "echo",
        "description": "Echo a message back.",
        "inputSchema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    },
    {
        "name": "save_note",
        "description": "Persist a short note (simulated write).",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "slow",
        "description": "Answer after a long delay to exercise timeouts.",
        "inputSchema": {"type": "object"},
    },
    {
        "name": "broken",
        "description": "Emit a malformed frame to exercise transport normalization.",
        "inputSchema": {"type": "object"},
    },
    {
        "name": "hidden",
        "description": "Server tool that the client allowlist must filter out.",
        "inputSchema": {"type": "object"},
    },
]


def _read_message(reader: BinaryIO) -> dict[str, Any] | None:
    content_length: int | None = None
    while True:
        line = reader.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            break
        key, _, value = line.partition(b":")
        if key.strip().lower() == b"content-length":
            content_length = int(value.strip())
    if content_length is None:
        return None
    return json.loads(reader.read(content_length))


def _write_message(writer: BinaryIO, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    with _WRITE_LOCK:
        writer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
        writer.flush()


_WRITE_LOCK = threading.Lock()


def _result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _tool_result(text: str, structured: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": structured,
        "isError": False,
    }


def _call_tool(name: str, arguments: dict[str, Any], writer: BinaryIO) -> dict[str, Any] | None:
    if name == "echo":
        message = str(arguments.get("message", ""))
        return _tool_result(f"echo: {message}", {"echoed": message})
    if name == "save_note":
        return _tool_result("note saved", {"saved": True})
    if name == "slow":
        time.sleep(5)
        return _tool_result("slow answer", {"answered": True})
    if name == "broken":
        # Malformed frame: the client must normalize this transport failure.
        with _WRITE_LOCK:
            writer.write(b"this is not a framed json-rpc message\r\n\r\n")
            writer.flush()
        return None
    if name == "hidden":
        return _tool_result("hidden tool executed", {"leaked": True})
    return _tool_result(f"unknown tool: {name}", {"isError": True})


def main() -> None:
    reader = sys.stdin.buffer
    writer = sys.stdout.buffer
    # Dispatch concurrently so the slow fixture tool does not block the others.
    with ThreadPoolExecutor(max_workers=4) as pool:
        while True:
            message = _read_message(reader)
            if message is None:
                return
            if message.get("id") is None:  # notification, e.g. notifications/initialized
                continue
            pool.submit(_handle, message, writer)


def _handle(message: dict[str, Any], writer: BinaryIO) -> None:
    method = message.get("method")
    request_id = message.get("id")
    if method == "initialize":
        response = _result(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            },
        )
    elif method == "tools/list":
        response = _result(request_id, {"tools": TOOLS})
    elif method == "tools/call":
        params = message.get("params") or {}
        called = _call_tool(str(params.get("name", "")), params.get("arguments") or {}, writer)
        if called is None:
            return
        response = _result(request_id, called)
    else:
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"unknown method: {method}"},
        }
    _write_message(writer, response)


if __name__ == "__main__":
    main()
