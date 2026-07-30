"""Model Context Protocol (MCP) tool adapter.

The executor depends on a tiny structural client protocol, keeping the core package
testable without launching a subprocess. ``stdio_mcp_executor`` bridges that protocol to
the optional official MCP Python SDK when it is installed.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, Protocol, cast
from uuid import UUID

from admires_harness.contracts import JsonValue, ToolCall, ToolResult


class MCPClient(Protocol):
    """The subset of an initialized MCP client session used by this adapter."""

    async def list_tools(self) -> Any: ...

    async def call_tool(
        self, name: str, arguments: Mapping[str, JsonValue] | None = None
    ) -> Any: ...


class MCPDependencyError(RuntimeError):
    """Raised when an optional MCP transport dependency is unavailable."""


def _read(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _as_json(value: Any) -> JsonValue:
    """Convert SDK/Pydantic objects to a JSON-safe value without leaking repr data."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    try:
        return cast(JsonValue, json.loads(json.dumps(value)))
    except (TypeError, ValueError):
        return str(value)


def _content_text(content: Any) -> str:
    blocks = (
        content if isinstance(content, Sequence) and not isinstance(content, (str, bytes)) else []
    )
    rendered: list[str] = []
    for block in blocks:
        text = _read(block, "text")
        if text is not None:
            rendered.append(str(text))
            continue
        resource = _read(block, "resource")
        if resource is not None:
            rendered.append(json.dumps(_as_json(resource), ensure_ascii=False, sort_keys=True))
    return "\n".join(rendered)


class MCPToolExecutor:
    """Expose tools discovered from one MCP server through the harness tool port."""

    def __init__(
        self,
        client: MCPClient,
        *,
        server_name: str,
        tools: Sequence[Any],
        timeout_seconds: float = 30.0,
        prefix_names: bool = True,
        allow_tools: frozenset[str] | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not server_name.strip():
            raise ValueError("server_name must not be empty")
        self._client = client
        self.server_name = server_name
        self.timeout_seconds = timeout_seconds
        self._definitions: dict[str, dict[str, Any]] = {}
        self._remote_names: dict[str, str] = {}
        for tool in tools:
            remote_name = str(_read(tool, "name", ""))
            if not remote_name or (allow_tools is not None and remote_name not in allow_tools):
                continue
            exposed_name = f"{server_name}__{remote_name}" if prefix_names else remote_name
            if exposed_name in self._definitions:
                raise ValueError(f"duplicate MCP tool name: {exposed_name}")
            raw_schema = _read(tool, "inputSchema", _read(tool, "input_schema", {}))
            schema = _as_json(raw_schema)
            if not isinstance(schema, dict):
                schema = {"type": "object"}
            self._remote_names[exposed_name] = remote_name
            self._definitions[exposed_name] = {
                "type": "function",
                "function": {
                    "name": exposed_name,
                    "description": str(_read(tool, "description", "")),
                    "parameters": schema,
                },
                "x-mcp-server": server_name,
                "x-mcp-tool": remote_name,
            }

    @classmethod
    async def connect(
        cls,
        client: MCPClient,
        *,
        server_name: str,
        timeout_seconds: float = 30.0,
        prefix_names: bool = True,
        allow_tools: frozenset[str] | None = None,
    ) -> MCPToolExecutor:
        async with asyncio.timeout(timeout_seconds):
            response = await client.list_tools()
        tools = _read(response, "tools", response)
        if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
            raise ValueError("MCP list_tools returned an invalid response")
        return cls(
            client,
            server_name=server_name,
            tools=tools,
            timeout_seconds=timeout_seconds,
            prefix_names=prefix_names,
            allow_tools=allow_tools,
        )

    def schemas(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._definitions.values())

    async def execute(
        self,
        call: ToolCall,
        *,
        run_id: UUID,
        idempotency_key: str,
    ) -> ToolResult:
        remote_name = self._remote_names.get(call.name)
        if remote_name is None:
            return ToolResult(
                ok=False,
                error_code="mcp_tool_not_found",
                content=f"Tool {call.name!r} is not exposed by MCP server {self.server_name!r}.",
            )
        try:
            async with asyncio.timeout(self.timeout_seconds):
                response = await self._client.call_tool(remote_name, call.arguments)
        except TimeoutError:
            return ToolResult(
                ok=False,
                error_code="mcp_timeout",
                content=f"MCP tool {call.name!r} timed out.",
                retryable=True,
            )
        except Exception as exc:  # SDK transports expose several implementation-specific errors.
            return ToolResult(
                ok=False,
                error_code="mcp_transport_error",
                content=f"MCP transport failed: {type(exc).__name__}",
                retryable=True,
            )

        failed = bool(_read(response, "isError", _read(response, "is_error", False)))
        structured = _as_json(
            _read(response, "structuredContent", _read(response, "structured_content", {}))
        )
        data: dict[str, JsonValue] = (
            structured if isinstance(structured, dict) else {"result": structured}
        )
        digest = hashlib.sha256(
            f"{run_id}:{idempotency_key}:{self.server_name}:{remote_name}".encode()
        ).hexdigest()
        return ToolResult(
            ok=not failed,
            content=_content_text(_read(response, "content", ())),
            data=data,
            error_code="mcp_tool_error" if failed else None,
            receipt=f"mcp:{digest}",
        )


@asynccontextmanager
async def stdio_mcp_executor(
    *,
    server_name: str,
    command: str,
    args: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
    cwd: str | None = None,
    timeout_seconds: float = 30.0,
    prefix_names: bool = True,
    allow_tools: frozenset[str] | None = None,
) -> AsyncIterator[MCPToolExecutor]:
    """Start a stdio MCP server using the optional official SDK.

    Importing the harness never requires MCP. Install ``admires-ai-harness[mcp]`` only
    when this transport is used.
    """

    try:
        mcp = importlib.import_module("mcp")
        stdio = importlib.import_module("mcp.client.stdio")
    except ImportError as exc:
        raise MCPDependencyError(
            "MCP support is optional; install admires-ai-harness[mcp]"
        ) from exc

    parameters = mcp.StdioServerParameters(
        command=command, args=list(args), env=dict(env) if env is not None else None, cwd=cwd
    )
    async with AsyncExitStack() as stack:
        read_stream, write_stream = await stack.enter_async_context(stdio.stdio_client(parameters))
        session = await stack.enter_async_context(mcp.ClientSession(read_stream, write_stream))
        async with asyncio.timeout(timeout_seconds):
            await session.initialize()
        executor = await MCPToolExecutor.connect(
            session,
            server_name=server_name,
            timeout_seconds=timeout_seconds,
            prefix_names=prefix_names,
            allow_tools=allow_tools,
        )
        yield executor
