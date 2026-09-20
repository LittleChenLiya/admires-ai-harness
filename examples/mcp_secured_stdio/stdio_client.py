"""Minimal stdio JSON-RPC client for the offline MCP demo.

This client exists so the demo and its tests run offline with zero third-party
dependencies: it speaks newline-delimited JSON-RPC 2.0 to a controlled local fixture
server and implements the structural MCPClient protocol consumed by the production
admires_harness.adapters.mcp.MCPToolExecutor.

Production deployments should use the official MCP SDK through
admires_harness.adapters.mcp.stdio_mcp_executor instead; see
docs/mcp-secured-tools.md.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from admires_harness.contracts import JsonValue


class StdioMCPClient:
    """Serialize JSON-RPC requests to one stdio server process.

    Implements list_tools/call_tool as required by
    admires_harness.adapters.mcp.MCPClient. Requests are serialized with a lock;
    responses are matched by id, so stale or unrelated frames are ignored.
    """

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process
        self._next_id = 0
        self._lock = asyncio.Lock()

    @classmethod
    async def start(cls, command: str, *args: str) -> StdioMCPClient:
        process = await asyncio.create_subprocess_exec(
            command,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        client = cls(process)
        await client._request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "mcp-secured-demo", "version": "1.0.0"},
            },
        )
        await client._notify("notifications/initialized", {})
        return client

    async def list_tools(self) -> Any:
        return await self._request("tools/list", {})

    async def call_tool(self, name: str, arguments: Mapping[str, JsonValue] | None = None) -> Any:
        return await self._request("tools/call", {"name": name, "arguments": dict(arguments or {})})

    async def aclose(self) -> None:
        if self._process.returncode is not None:
            return
        self._process.terminate()
        try:
            async with asyncio.timeout(5):
                await self._process.wait()
        except TimeoutError:
            self._process.kill()
            await self._process.wait()

    async def _notify(self, method: str, params: Mapping[str, Any]) -> None:
        async with self._lock:
            self._write({"jsonrpc": "2.0", "method": method, "params": dict(params)})

    async def _request(self, method: str, params: Mapping[str, Any]) -> Any:
        async with self._lock:
            self._next_id += 1
            request_id = self._next_id
            self._write(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": dict(params)}
            )
            assert self._process.stdout is not None
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    raise ConnectionError("MCP server closed stdio")
                try:
                    message = json.loads(line)
                except ValueError:
                    continue  # Skip malformed frames from the untrusted server.
                if not isinstance(message, dict) or message.get("id") != request_id:
                    continue
                if "error" in message:
                    error = message["error"]
                    code = error.get("code") if isinstance(error, dict) else "?"
                    raise RuntimeError(f"JSON-RPC error {code} for {method}")
                return message.get("result", {})

    def _write(self, message: Mapping[str, Any]) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(message).encode() + b"\n")
