from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from admires_harness.adapters.mcp import (
    MCPDependencyError,
    MCPToolExecutor,
    stdio_mcp_executor,
)
from admires_harness.contracts import ToolCall


class FakeMCPClient:
    """Dict-shaped fake SDK client."""

    def __init__(self) -> None:
        self.called: list[tuple[str, object]] = []
        self.tools: Any = [
            {
                "name": "weather",
                "description": "Get the weather",
                "inputSchema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
            {"name": "hidden", "inputSchema": {"type": "object"}},
        ]
        self.result: Any = {
            "content": [{"type": "text", "text": "sunny"}],
            "structuredContent": {"temperature": 24},
            "isError": False,
        }

    async def list_tools(self) -> object:
        return {"tools": self.tools}

    async def call_tool(self, name: str, arguments: object = None) -> Any:
        self.called.append((name, arguments))
        return self.result


class ObjectFakeMCPClient(FakeMCPClient):
    """Object-shaped fake SDK client mirroring the official SDK's models."""

    async def list_tools(self) -> object:
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="weather",
                    description="Get the weather",
                    inputSchema={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                ),
                SimpleNamespace(name="hidden", description=None, inputSchema={"type": "object"}),
            ]
        )

    async def call_tool(self, name: str, arguments: object = None) -> Any:
        self.called.append((name, arguments))
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="sunny")],
            structuredContent={"temperature": 24},
            isError=False,
        )


def make_call(name: str = "demo__weather", **kwargs: Any) -> ToolCall:
    return ToolCall(name=name, **kwargs)


@pytest.mark.asyncio
async def test_mcp_adapter_discovers_allowlisted_tools_and_executes() -> None:
    client = FakeMCPClient()
    executor = await MCPToolExecutor.connect(
        client, server_name="demo", allow_tools=frozenset({"weather"})
    )

    schemas = executor.schemas()
    assert [schema["function"]["name"] for schema in schemas] == ["demo__weather"]
    result = await executor.execute(
        ToolCall(name="demo__weather", arguments={"city": "Hangzhou"}),
        run_id=uuid4(),
        idempotency_key="step-1",
    )

    assert result.ok
    assert result.content == "sunny"
    assert result.data == {"temperature": 24}
    assert result.receipt is not None and result.receipt.startswith("mcp:")
    assert client.called == [("weather", {"city": "Hangzhou"})]


@pytest.mark.asyncio
async def test_dict_and_object_sdk_shapes_normalize_identically() -> None:
    dict_executor = await MCPToolExecutor.connect(FakeMCPClient(), server_name="demo")
    object_executor = await MCPToolExecutor.connect(ObjectFakeMCPClient(), server_name="demo")

    assert dict_executor.schemas() == object_executor.schemas()
    schema = dict_executor.schemas()[0]
    assert schema["function"]["parameters"] == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
    }
    assert schema["x-mcp-server"] == "demo"
    assert schema["x-mcp-tool"] == "weather"

    run_id = uuid4()
    dict_result = await dict_executor.execute(make_call(), run_id=run_id, idempotency_key="same")
    object_result = await object_executor.execute(
        make_call(), run_id=run_id, idempotency_key="same"
    )
    assert dict_result == object_result
    assert object_result.ok
    assert object_result.content == "sunny"
    assert object_result.data == {"temperature": 24}


@pytest.mark.asyncio
async def test_empty_tool_list_yields_no_schemas() -> None:
    client = FakeMCPClient()
    client.tools = []
    executor = await MCPToolExecutor.connect(client, server_name="demo")
    assert executor.schemas() == ()


@pytest.mark.asyncio
async def test_invalid_tool_list_response_is_rejected() -> None:
    client = FakeMCPClient()

    async def bad_list_tools() -> object:
        return {"unexpected": 1}

    client.list_tools = bad_list_tools  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="invalid response"):
        await MCPToolExecutor.connect(client, server_name="demo")


@pytest.mark.asyncio
async def test_malformed_tool_entries_are_skipped() -> None:
    client = FakeMCPClient()
    client.tools = [None, {}, {"description": "no name"}, {"name": "kept"}]
    executor = await MCPToolExecutor.connect(client, server_name="demo")
    assert [schema["function"]["name"] for schema in executor.schemas()] == ["demo__kept"]
    assert executor.schemas()[0]["function"]["parameters"] == {"type": "object"}


@pytest.mark.asyncio
async def test_duplicate_exposed_names_are_rejected() -> None:
    client = FakeMCPClient()
    client.tools = [{"name": "weather"}, {"name": "weather"}]
    with pytest.raises(ValueError, match="duplicate MCP tool name"):
        await MCPToolExecutor.connect(client, server_name="demo")

    unprefixed = FakeMCPClient()
    unprefixed.tools = [{"name": "a"}, {"name": "a"}]
    with pytest.raises(ValueError, match="duplicate MCP tool name"):
        await MCPToolExecutor.connect(unprefixed, server_name="demo", prefix_names=False)


@pytest.mark.asyncio
async def test_input_schema_normalization_is_deterministic() -> None:
    client = FakeMCPClient()
    client.tools = [
        {"name": "no_schema"},
        {"name": "null_schema", "inputSchema": None},
        {"name": "partial_schema", "inputSchema": {"properties": {"x": {"type": "integer"}}}},
        {"name": "bytes_schema", "inputSchema": {"type": "object", "raw": b"\x01\x02"}},
    ]
    executor = await MCPToolExecutor.connect(client, server_name="demo")
    parameters = {
        schema["function"]["name"]: schema["function"]["parameters"]
        for schema in executor.schemas()
    }
    assert parameters["demo__no_schema"] == {"type": "object"}
    assert parameters["demo__null_schema"] == {"type": "object"}
    assert parameters["demo__partial_schema"] == {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
    }
    assert parameters["demo__bytes_schema"] == {"type": "object", "raw": "AQI="}
    json.dumps(parameters)  # schemas must stay JSON-serializable


@pytest.mark.asyncio
async def test_structured_content_and_non_text_blocks_are_preserved() -> None:
    client = FakeMCPClient()
    client.result = {
        "content": [
            {"type": "text", "text": "first"},
            {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
            {"type": "resource_link", "uri": "file:///report.pdf", "name": "report"},
            {"type": "text", "text": "second"},
        ],
        "structuredContent": {"rows": 3},
        "isError": False,
    }
    executor = await MCPToolExecutor.connect(client, server_name="demo")
    result = await executor.execute(make_call(), run_id=uuid4(), idempotency_key="mixed")

    assert result.ok
    assert result.content == "first\nsecond"
    assert result.data == {
        "rows": 3,
        "content": [
            {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
            {"type": "resource_link", "uri": "file:///report.pdf", "name": "report"},
        ],
    }
    json.dumps(result.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_scalar_structured_content_is_wrapped() -> None:
    client = FakeMCPClient()
    client.result = {"content": [], "structuredContent": [1, 2, 3], "isError": False}
    executor = await MCPToolExecutor.connect(client, server_name="demo")
    result = await executor.execute(make_call(), run_id=uuid4(), idempotency_key="scalar")
    assert result.ok
    assert result.data == {"result": [1, 2, 3]}


@pytest.mark.asyncio
async def test_mcp_adapter_rejects_unknown_and_normalizes_tool_error() -> None:
    client = FakeMCPClient()
    executor = await MCPToolExecutor.connect(client, server_name="demo")

    missing = await executor.execute(
        ToolCall(name="other"), run_id=uuid4(), idempotency_key="missing"
    )
    assert not missing.ok
    assert missing.error_code == "mcp_tool_not_found"
    assert not missing.retryable

    client.result = {"content": [{"text": "denied"}], "isError": True}
    failed = await executor.execute(
        ToolCall(name="demo__weather"), run_id=uuid4(), idempotency_key="denied"
    )
    assert not failed.ok
    assert failed.error_code == "mcp_tool_error"
    assert not failed.retryable
    assert failed.content == "denied"
    assert failed.receipt is not None and failed.receipt.startswith("mcp:")


@pytest.mark.asyncio
async def test_mcp_adapter_normalizes_timeout_and_transport_error() -> None:
    class SlowClient(FakeMCPClient):
        async def call_tool(self, name: str, arguments: object = None) -> Any:
            await asyncio.sleep(0.02)
            return await super().call_tool(name, arguments)

    slow = await MCPToolExecutor.connect(SlowClient(), server_name="demo", timeout_seconds=0.001)
    timed_out = await slow.execute(
        ToolCall(name="demo__weather"), run_id=uuid4(), idempotency_key="timeout"
    )
    assert timed_out.error_code == "mcp_timeout"
    assert timed_out.retryable
    assert timed_out.receipt is not None and timed_out.receipt.startswith("mcp:")

    client = FakeMCPClient()

    async def fail(name: str, arguments: object = None) -> Any:
        raise ConnectionError("private server details")

    client.call_tool = fail  # type: ignore[method-assign]
    executor = await MCPToolExecutor.connect(client, server_name="demo")
    failed = await executor.execute(
        ToolCall(name="demo__weather"), run_id=uuid4(), idempotency_key="network"
    )
    assert failed.error_code == "mcp_transport_error"
    assert failed.retryable
    assert "private server details" not in failed.content
    assert "0x" not in failed.content  # no remote object repr leaks
    assert failed.receipt is not None and failed.receipt.startswith("mcp:")


@pytest.mark.asyncio
async def test_caller_cancellation_propagates() -> None:
    started = asyncio.Event()

    class HangingClient(FakeMCPClient):
        async def call_tool(self, name: str, arguments: object = None) -> Any:
            started.set()
            await asyncio.Event().wait()
            return None  # pragma: no cover - unreachable

    executor = await MCPToolExecutor.connect(HangingClient(), server_name="demo")
    task = asyncio.create_task(
        executor.execute(make_call(), run_id=uuid4(), idempotency_key="cancel")
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_receipt_is_stable_per_idempotency_key() -> None:
    executor = await MCPToolExecutor.connect(FakeMCPClient(), server_name="demo")
    run_id = uuid4()
    first = await executor.execute(make_call(), run_id=run_id, idempotency_key="key-1")
    second = await executor.execute(make_call(), run_id=run_id, idempotency_key="key-1")
    third = await executor.execute(make_call(), run_id=run_id, idempotency_key="key-2")
    assert first.receipt is not None
    assert first.receipt == second.receipt
    assert first.receipt != third.receipt


@pytest.mark.asyncio
async def test_result_serialization_is_json_safe_without_repr_leaks() -> None:
    client = FakeMCPClient()
    client.result = {
        "content": [{"type": "text", "text": "ok"}],
        "structuredContent": {"blob": b"\x00\x01", "odd": {1, 2}},
        "isError": False,
    }
    executor = await MCPToolExecutor.connect(client, server_name="demo")
    result = await executor.execute(make_call(), run_id=uuid4(), idempotency_key="json")
    assert result.ok
    assert result.data["blob"] == "AAE="
    assert result.data["odd"] == "<non-json-serializable: set>"
    serialized = json.dumps(result.model_dump(mode="json"), sort_keys=True)
    assert "object at 0x" not in serialized


@pytest.mark.asyncio
async def test_stdio_transport_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "mcp", None)
    with pytest.raises(MCPDependencyError, match="optional"):
        async with stdio_mcp_executor(server_name="demo", command="noop"):
            pass  # pragma: no cover - unreachable


def test_core_package_imports_without_mcp_installed() -> None:
    import admires_harness.adapters.mcp as module

    assert module.MCPToolExecutor is not None
    # Importing the adapter module (done at collection time) must not require the
    # optional ``mcp`` package; only the stdio transport imports it lazily.
    assert callable(module.stdio_mcp_executor)
