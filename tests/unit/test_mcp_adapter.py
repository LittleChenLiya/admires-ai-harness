from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest

from admires_harness.adapters.mcp import MCPToolExecutor
from admires_harness.contracts import ToolCall


class FakeMCPClient:
    def __init__(self) -> None:
        self.called: list[tuple[str, object]] = []
        self.result: Any = {
            "content": [{"type": "text", "text": "sunny"}],
            "structuredContent": {"temperature": 24},
            "isError": False,
        }

    async def list_tools(self) -> object:
        return {
            "tools": [
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
        }

    async def call_tool(self, name: str, arguments: object = None) -> Any:
        self.called.append((name, arguments))
        return self.result


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
async def test_mcp_adapter_rejects_unknown_and_normalizes_tool_error() -> None:
    client = FakeMCPClient()
    executor = await MCPToolExecutor.connect(client, server_name="demo")

    missing = await executor.execute(
        ToolCall(name="other"), run_id=uuid4(), idempotency_key="missing"
    )
    assert not missing.ok
    assert missing.error_code == "mcp_tool_not_found"

    client.result = {"content": [{"text": "denied"}], "isError": True}
    failed = await executor.execute(
        ToolCall(name="demo__weather"), run_id=uuid4(), idempotency_key="denied"
    )
    assert not failed.ok
    assert failed.error_code == "mcp_tool_error"
    assert failed.content == "denied"


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

    client = FakeMCPClient()

    async def fail(name: str, arguments: object = None) -> Any:
        raise ConnectionError("private server details")

    client.call_tool = fail  # type: ignore[method-assign]
    executor = await MCPToolExecutor.connect(client, server_name="demo")
    failed = await executor.execute(
        ToolCall(name="demo__weather"), run_id=uuid4(), idempotency_key="network"
    )
    assert failed.error_code == "mcp_transport_error"
    assert "private server details" not in failed.content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        None,
        42,
        {"isError": False},
        {"content": "not-a-sequence"},
        {"content": [42]},
        {"content": ["raw-string-block"]},
    ],
)
async def test_mcp_adapter_normalizes_malformed_responses(result: Any) -> None:
    client = FakeMCPClient()
    client.result = result
    executor = await MCPToolExecutor.connect(
        client, server_name="demo", allow_tools=frozenset({"weather"})
    )

    outcome = await executor.execute(
        ToolCall(name="demo__weather"), run_id=uuid4(), idempotency_key="malformed"
    )

    assert not outcome.ok
    assert outcome.error_code == "mcp_malformed_response"
    assert not outcome.retryable


def test_mcp_adapter_accepts_empty_content_sequence() -> None:
    # An empty list is a structurally valid (if uninformative) MCP result; it must not
    # be misclassified as malformed.
    client = FakeMCPClient()
    client.result = {"content": []}

    async def scenario() -> None:
        executor = await MCPToolExecutor.connect(
            client, server_name="demo", allow_tools=frozenset({"weather"})
        )
        outcome = await executor.execute(
            ToolCall(name="demo__weather"), run_id=uuid4(), idempotency_key="empty"
        )
        assert outcome.ok
        assert outcome.content == ""

    asyncio.run(scenario())
