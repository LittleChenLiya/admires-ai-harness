"""Minimal framed JSON-RPC stdio client for the local fake-server fixture.

It satisfies the structural MCPClient protocol consumed by the production
MCPToolExecutor adapter. Real deployments should use the official MCP SDK
through admires_harness.adapters.mcp.stdio_mcp_executor instead of this
fixture client.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Any

from admires_harness.contracts import JsonValue


class StdioTransportError(RuntimeError):
    """Raised when the server framing or payload cannot be parsed."""


class StdioJsonRpcClient:
    """Async JSON-RPC client over one child process stdio stream pair."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._reader_task = asyncio.create_task(self._read_loop())

    @classmethod
    async def start(cls, command: str, args: Sequence[str]) -> StdioJsonRpcClient:
        process = await asyncio.create_subprocess_exec(
            command,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        client = cls(process)
        await client._request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "mcp-policy-demo", "version": "0.1.0"},
            },
        )
        await client._notify("notifications/initialized", {})
        return client

    async def list_tools(self) -> Any:
        return await self._request("tools/list", {})

    async def call_tool(self, name: str, arguments: Mapping[str, JsonValue] | None = None) -> Any:
        return await self._request("tools/call", {"name": name, "arguments": dict(arguments or {})})

    async def close(self) -> None:
        self._reader_task.cancel()
        if self._process.returncode is None:
            self._process.terminate()
            try:
                async with asyncio.timeout(2):
                    await self._process.wait()
            except TimeoutError:
                self._process.kill()
                await self._process.wait()

    async def _read_loop(self) -> None:
        assert self._process.stdout is not None
        try:
            while True:
                content_length: int | None = None
                while True:
                    line = await self._process.stdout.readline()
                    if not line:
                        raise StdioTransportError("server closed the stream")
                    line = line.strip()
                    if not line:
                        break
                    key, separator, value = line.partition(b":")
                    if not separator or key.strip().lower() != b"content-length":
                        raise StdioTransportError("malformed frame header")
                    content_length = int(value.strip())
                if content_length is None:
                    raise StdioTransportError("missing Content-Length header")
                body = await self._process.stdout.readexactly(content_length)
                message = json.loads(body)
                request_id = message.get("id")
                future = self._pending.pop(request_id, None)
                if future is not None and not future.done():
                    if "error" in message:
                        future.set_exception(
                            StdioTransportError(
                                str(message["error"].get("message", "server error"))
                            )
                        )
                    else:
                        future.set_result(message.get("result"))
        except asyncio.CancelledError:
            self._fail_pending(StdioTransportError("transport closed"))
        except (StdioTransportError, ValueError, asyncio.IncompleteReadError) as exc:
            self._fail_pending(exc)

    def _fail_pending(self, error: BaseException) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

    async def _write(self, payload: dict[str, Any]) -> None:
        assert self._process.stdin is not None
        body = json.dumps(payload).encode("utf-8")
        self._process.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
        await self._process.stdin.drain()

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        return await future

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._write({"jsonrpc": "2.0", "method": method, "params": params})
