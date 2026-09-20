"""Deterministic MCP adapter tests using pure-Python fake clients.

No MCP SDK or subprocess is required: the adapter only depends on a structural
client protocol, so both dict-shaped and object-shaped SDK responses are faked.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import sys
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest

from admires_harness.adapters import mcp as mcp_adapter
from admires_harness.adapters.mcp import MCPDependencyError, MCPToolExecutor
from admires_harness.contracts import ToolCall, ToolResult

RUN_ID = UUID("12345678-1234-5678-1234-567812345678")

WEATHER_TOOL = {
    "name": "weather",
    "description": "Get the weather",
    "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}},
}
HIDDEN_TOOL = {"name": "hidden", "inputSchema": {"type": "object"}}


class FakeMCPClient:
    """Structural stand-in for an initialized MCP client session."""

    def __init__(self, tools: Any = None, result: Any = None) -> None:
        self.tools = {"tools": [WEATHER_TOOL, HIDDEN_TOOL]} if tools is None else tools
        self.result = (
            {
                "content": [{"type": "text", "text": "sunny"}],
                "structuredContent": {"temperature": 24},
                "isError": False,
            }
            if result is None
            else result
        )
        self.called: list[tuple[str, object]] = []

    async def list_tools(self) -> Any:
        return self.tools

    async def call_tool(self, name: str, arguments: object = None) -> Any:
        self.called.append((name, arguments))
        return self.result


def make_call(name: str = "demo__weather", **arguments: Any) -> ToolCall:
    return ToolCall(name=name, arguments=arguments)


def assert_json_safe(result: ToolResult) -> None:
    json.dumps(result.model_dump(mode="json"), sort_keys=True)


async def connect(client: FakeMCPClient, **kwargs: Any) -> MCPToolExecutor:
    kwargs.setdefault("server_name", "demo")
    return await MCPToolExecutor.connect(client, **kwargs)


async def test_dict_responses_discover_allowlist_and_execute() -> None:
    client = FakeMCPClient()
    executor = await connect(client, allow_tools=frozenset({"weather"}))

    schemas = executor.schemas()
    assert [schema["function"]["name"] for schema in schemas] == ["demo__weather"]
    assert schemas[0]["function"]["parameters"] == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
    }
    assert schemas[0]["x-mcp-server"] == "demo"
    assert schemas[0]["x-mcp-tool"] == "weather"

    result = await executor.execute(
        make_call(city="Hangzhou"), run_id=RUN_ID, idempotency_key="step-1"
    )

    assert result.ok
    assert result.error_code is None
    assert not result.retryable
    assert result.content == "sunny"
    assert result.data == {"temperature": 24}
    assert result.receipt is not None and result.receipt.startswith("mcp:")
    assert client.called == [("weather", {"city": "Hangzhou"})]
    assert_json_safe(result)


async def test_object_responses_normalize_like_dicts() -> None:
    tools = SimpleNamespace(
        tools=[
            SimpleNamespace(
                name="weather",
                description="Get the weather",
                inputSchema={"properties": {"city": {"type": "string"}}},
            )
        ]
    )
    result = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="cloudy")],
        structured_content={"temperature": 21},
        is_error=False,
    )
    client = FakeMCPClient(tools=tools, result=result)
    executor = await connect(client)

    schemas = executor.schemas()
    assert [schema["function"]["name"] for schema in schemas] == ["demo__weather"]
    # A schema without an explicit type is normalized to an object schema.
    assert schemas[0]["function"]["parameters"] == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
    }

    executed = await executor.execute(make_call(), run_id=RUN_ID, idempotency_key="obj")
    assert executed.ok
    assert executed.content == "cloudy"
    assert executed.data == {"temperature": 21}
    assert_json_safe(executed)


async def test_model_dump_blocks_are_converted_without_repr_leak() -> None:
    class PydanticishBlock:
        def model_dump(self, mode: str = "json") -> dict[str, object]:
            assert mode == "json"
            return {"type": "text", "text": "dumped"}

    client = FakeMCPClient(
        result={"content": [PydanticishBlock()], "structuredContent": None, "isError": False}
    )
    executor = await connect(client)
    result = await executor.execute(make_call(), run_id=RUN_ID, idempotency_key="dump")

    assert result.ok
    assert result.content == "dumped"
    assert "object at 0x" not in json.dumps(result.model_dump(mode="json"))


async def test_empty_tool_list_yields_executor_without_tools() -> None:
    executor = await connect(FakeMCPClient(tools={"tools": []}))

    assert executor.schemas() == ()
    missing = await executor.execute(make_call(), run_id=RUN_ID, idempotency_key="empty")
    assert not missing.ok
    assert missing.error_code == "mcp_tool_not_found"
    assert not missing.retryable


@pytest.mark.parametrize(
    "response",
    [
        object(),
        {"tools": "not-a-list"},
        {"tools": None},
        "raw-string",
    ],
    ids=["no-tools-attribute", "tools-string", "tools-none", "bare-string"],
)
async def test_invalid_tool_list_rejected(response: Any) -> None:
    with pytest.raises(ValueError, match="invalid"):
        await connect(FakeMCPClient(tools=response))


async def test_duplicate_exposed_names_rejected() -> None:
    tools = {"tools": [WEATHER_TOOL, dict(WEATHER_TOOL)]}
    with pytest.raises(ValueError, match="duplicate MCP tool name: demo__weather"):
        await connect(FakeMCPClient(tools=tools))

    unprefixed = {"tools": [{"name": "a"}, {"name": "a"}]}
    with pytest.raises(ValueError, match="duplicate MCP tool name: a"):
        await connect(FakeMCPClient(tools=unprefixed), prefix_names=False)


async def test_allowlist_blocks_non_listed_remote_calls() -> None:
    client = FakeMCPClient()
    executor = await connect(client, allow_tools=frozenset({"weather"}))

    blocked = await executor.execute(
        make_call("demo__hidden"), run_id=RUN_ID, idempotency_key="deny"
    )
    assert not blocked.ok
    assert blocked.error_code == "mcp_tool_not_found"
    assert client.called == []


async def test_schema_normalization_defaults_are_stable() -> None:
    tools = {
        "tools": [
            {"name": "no_schema"},
            {"name": "bad_schema", "inputSchema": "not-a-schema"},
            {"name": "partial", "inputSchema": {"properties": {"x": {"type": "integer"}}}},
        ]
    }
    executor = await connect(FakeMCPClient(tools=tools))

    parameters = [schema["function"]["parameters"] for schema in executor.schemas()]
    assert parameters == [
        {"type": "object", "properties": {}},
        {"type": "object", "properties": {}},
        {"type": "object", "properties": {"x": {"type": "integer"}}},
    ]


async def test_non_dict_structured_content_is_wrapped() -> None:
    client = FakeMCPClient(result={"content": [], "structuredContent": [1, 2, 3], "isError": False})
    executor = await connect(client)
    result = await executor.execute(make_call(), run_id=RUN_ID, idempotency_key="list")

    assert result.ok
    assert result.data == {"result": [1, 2, 3]}
    assert_json_safe(result)


async def test_mixed_text_image_and_resource_content_is_lossless() -> None:
    content = [
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "annotated", "annotations": {"audience": ["user"]}},
        {"type": "image", "data": "aW1n", "mimeType": "image/png"},
        {"type": "resource_link", "uri": "file:///report.pdf", "name": "report"},
        {
            "type": "resource",
            "resource": {"uri": "file:///notes.txt", "text": "file body"},
        },
        {"type": "resource", "resource": {"uri": "file:///blob.bin", "blob": "AAE="}},
    ]
    client = FakeMCPClient(result={"content": content, "isError": False})
    executor = await connect(client)
    result = await executor.execute(make_call(), run_id=RUN_ID, idempotency_key="mixed")

    assert result.ok
    # Text blocks and embedded text resources share the text channel.
    assert result.content == "hello\nannotated\nfile body"
    # Everything non-text survives as JSON-safe blocks; nothing is dropped.
    assert result.data == {
        "content_blocks": [
            {"type": "text", "text": "annotated", "annotations": {"audience": ["user"]}},
            {"type": "image", "data": "aW1n", "mimeType": "image/png"},
            {"type": "resource_link", "uri": "file:///report.pdf", "name": "report"},
            {"type": "resource", "resource": {"uri": "file:///notes.txt", "text": "file body"}},
            {"type": "resource", "resource": {"uri": "file:///blob.bin", "blob": "AAE="}},
        ]
    }
    assert_json_safe(result)


async def test_unserializable_content_never_leaks_repr() -> None:
    class Opaque:
        def __repr__(self) -> str:  # pragma: no cover - must never be called
            raise AssertionError("repr must not be used")

    client = FakeMCPClient(result={"content": [Opaque()], "isError": False})
    executor = await connect(client)
    result = await executor.execute(make_call(), run_id=RUN_ID, idempotency_key="opaque")

    assert result.ok
    assert result.data == {
        "content_blocks": [{"type": "unknown", "value": "<unserializable:Opaque>"}]
    }
    assert_json_safe(result)


async def test_timeout_is_retryable_and_distinct() -> None:
    class SlowClient(FakeMCPClient):
        async def call_tool(self, name: str, arguments: object = None) -> Any:
            await asyncio.sleep(5)
            return await super().call_tool(name, arguments)

    executor = await connect(SlowClient(), timeout_seconds=0.01)
    result = await executor.execute(make_call(), run_id=RUN_ID, idempotency_key="slow")

    assert not result.ok
    assert result.error_code == "mcp_timeout"
    assert result.retryable
    assert_json_safe(result)


async def test_caller_cancellation_propagates() -> None:
    never = asyncio.Event()

    class HangingClient(FakeMCPClient):
        async def call_tool(self, name: str, arguments: object = None) -> Any:
            await never.wait()
            raise AssertionError("unreachable")

    executor = await connect(HangingClient(), timeout_seconds=30.0)
    task = asyncio.create_task(
        executor.execute(make_call(), run_id=RUN_ID, idempotency_key="cancel")
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_transport_exception_is_sanitized_and_retryable() -> None:
    client = FakeMCPClient()

    async def fail(name: str, arguments: object = None) -> Any:
        raise ConnectionError("private server details 10.0.0.1")

    client.call_tool = fail  # type: ignore[method-assign]
    executor = await connect(client)
    result = await executor.execute(make_call(), run_id=RUN_ID, idempotency_key="net")

    assert not result.ok
    assert result.error_code == "mcp_transport_error"
    assert result.retryable
    assert "ConnectionError" in result.content
    assert "private server details" not in result.content
    assert "10.0.0.1" not in result.content
    assert_json_safe(result)


async def test_tool_error_is_not_retryable_and_keeps_content() -> None:
    client = FakeMCPClient(
        result={"content": [{"type": "text", "text": "denied"}], "isError": True}
    )
    executor = await connect(client)
    failed = await executor.execute(make_call(), run_id=RUN_ID, idempotency_key="err")

    assert not failed.ok
    assert failed.error_code == "mcp_tool_error"
    assert not failed.retryable
    assert failed.content == "denied"
    assert failed.receipt is not None and failed.receipt.startswith("mcp:")

    object_client = FakeMCPClient(
        result=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="nope")],
            is_error=True,
            structured_content=None,
        )
    )
    object_executor = await connect(object_client)
    object_failed = await object_executor.execute(
        make_call(), run_id=RUN_ID, idempotency_key="err-obj"
    )
    assert not object_failed.ok
    assert object_failed.error_code == "mcp_tool_error"
    assert object_failed.content == "nope"


async def test_receipt_is_deterministic_per_idempotency_key() -> None:
    client = FakeMCPClient()
    executor = await connect(client)

    first = await executor.execute(make_call(), run_id=RUN_ID, idempotency_key="same")
    second = await executor.execute(make_call(), run_id=RUN_ID, idempotency_key="same")
    other = await executor.execute(make_call(), run_id=RUN_ID, idempotency_key="other")

    assert first.receipt == second.receipt
    assert first.receipt != other.receipt


async def test_missing_optional_dependency_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> Any:
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, package)

    monkeypatch.setattr(mcp_adapter.importlib, "import_module", fake_import)

    with pytest.raises(MCPDependencyError, match=r"admires-ai-harness\[mcp\]"):
        async with mcp_adapter.stdio_mcp_executor(server_name="demo", command="server"):
            pass


def test_core_package_imports_without_mcp_installed() -> None:
    importlib.import_module("admires_harness")
    importlib.import_module("admires_harness.adapters.mcp")
    if importlib.util.find_spec("mcp") is None:
        assert "mcp" not in sys.modules
