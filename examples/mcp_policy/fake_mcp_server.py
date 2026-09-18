#!/usr/bin/env python3
"""Deterministic, offline fake MCP stdio server for the policy example.

This script speaks the minimal JSON-RPC/2.0 MCP stdio wire protocol by hand. It uses
only the Python standard library so the example runs without installing the optional
``mcp`` extra on the *server* side (the client side still uses the real SDK through
``stdio_mcp_executor``). It never opens a socket and reads no configuration beyond the
spawn arguments.

Exposed tools (the example allowlists a subset of these):

* ``echo``      - returns the given text (read-only, always succeeds)
* ``record``    - returns the given note and marks it as a write side effect
* ``fetch_url`` - names a network side effect so policy denial can be demonstrated
* ``internal_only`` - exists on the server but is excluded from the client allowlist
* ``always_fails``  - returns a well-formed ``isError: true`` tool result
* ``slow``      - sleeps longer than the client timeout (``MCP_FAKE_DELAY_SECONDS``)
* ``broken``    - returns a malformed content array to test response normalization

Run it standalone only for debugging; the example driver spawns it automatically.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

PROTOCOL_VERSION = "2025-06-18"

TOOLS: tuple[dict[str, object], ...] = (
    {
        "name": "echo",
        "description": "Return the provided text unchanged.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "record",
        "description": "Pretend to persist a short note.",
        "inputSchema": {
            "type": "object",
            "properties": {"note": {"type": "string"}},
            "required": ["note"],
        },
    },
    {
        "name": "fetch_url",
        "description": "Pretend to fetch a URL (network side effect).",
        "inputSchema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "internal_only",
        "description": "Advertised but never allowlisted by the example client.",
        "inputSchema": {"type": "object"},
    },
    {
        "name": "always_fails",
        "description": "Return a well-formed MCP error result.",
        "inputSchema": {"type": "object"},
    },
    {
        "name": "slow",
        "description": "Sleep past the client timeout.",
        "inputSchema": {"type": "object"},
    },
    {
        "name": "broken",
        "description": "Return a malformed content sequence.",
        "inputSchema": {"type": "object"},
    },
)

_write_lock = threading.Lock()


def _send(message: dict[str, object]) -> None:
    encoded = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
    with _write_lock:
        sys.stdout.write(encoded + "\n")
        sys.stdout.flush()


def _text_result(text: str, *, is_error: bool = False) -> dict[str, object]:
    result: dict[str, object] = {"content": [{"type": "text", "text": text}]}
    if is_error:
        result["isError"] = True
    return result


def _call_tool(name: str, arguments: dict[str, object]) -> dict[str, object] | None:
    if name == "echo":
        return _text_result(str(arguments.get("text", "")))
    if name == "record":
        return _text_result(f"recorded: {arguments.get('note', '')}")
    if name == "fetch_url":
        return _text_result(f"would fetch {arguments.get('url', '')}")
    if name == "always_fails":
        return _text_result("remote tool refused the request", is_error=True)
    if name == "slow":
        time.sleep(float(os.environ.get("MCP_FAKE_DELAY_SECONDS", "5")))
        return _text_result("too late")
    if name == "broken":
        # Deliberately violates the MCP spec: content blocks must be objects.
        return {"content": ["not-an-object-block"]}
    return None


def _handle(message: dict[str, object]) -> None:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params")
    params = params if isinstance(params, dict) else {}

    if method == "initialize":
        offered = params.get("protocolVersion")
        _send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": offered if isinstance(offered, str) else PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "offline-policy-fixture", "version": "1.0.0"},
                },
            }
        )
        return
    if method == "ping":
        _send({"jsonrpc": "2.0", "id": request_id, "result": {}})
        return
    if method in {"notifications/initialized", "notifications/cancelled"}:
        return
    if method == "tools/list":
        _send({"jsonrpc": "2.0", "id": request_id, "result": {"tools": list(TOOLS)}})
        return
    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments")
        arguments = arguments if isinstance(arguments, dict) else {}
        result = _call_tool(str(name), arguments) if isinstance(name, str) else None
        if result is None:
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"unknown tool: {name!r}"},
                }
            )
        else:
            _send({"jsonrpc": "2.0", "id": request_id, "result": result})
        return
    if request_id is not None:
        _send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"method not found: {method!r}"},
            }
        )


def _serve_line(line: str) -> None:
    try:
        message = json.loads(line)
    except json.JSONDecodeError:
        return
    if isinstance(message, dict):
        _handle(message)


def main() -> None:
    # One thread per request keeps slow tools from blocking subsequent calls, which
    # makes timeout cases deterministic while preserving ordered JSON-RPC responses via
    # the write lock.
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if line:
            threading.Thread(target=_serve_line, args=(line,), daemon=True).start()


if __name__ == "__main__":
    main()
