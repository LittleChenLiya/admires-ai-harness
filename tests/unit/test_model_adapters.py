from __future__ import annotations

import json

import httpx
import pytest

from admires_harness.adapters.models import (
    MockModelProvider,
    ModelProviderError,
    OpenAICompatibleProvider,
    create_model_provider,
)
from admires_harness.config import Settings
from admires_harness.contracts import Message, ModelResponse, Role


@pytest.mark.asyncio
async def test_mock_model_consumes_script_then_echoes_and_streams() -> None:
    model = MockModelProvider(
        [ModelResponse(content="scripted", model="fixture")], stream_chunk_size=4
    )
    messages = [Message(role=Role.USER, content="hello offline world")]

    assert (await model.complete(messages)).content == "scripted"
    chunks = [chunk async for chunk in model.stream(messages)]

    assert "".join(chunks) == "Mock response: hello offline world"
    assert len(model.calls) == 2


def test_mock_model_rejects_invalid_stream_options() -> None:
    with pytest.raises(ValueError, match="positive"):
        MockModelProvider(stream_chunk_size=0)
    with pytest.raises(ValueError, match="negative"):
        MockModelProvider(stream_delay=-1)


@pytest.mark.asyncio
async def test_openai_compatible_complete_maps_messages_tools_and_usage() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        captured["authorization"] = request.headers.get("Authorization")
        return httpx.Response(
            200,
            json={
                "model": "local-model",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"query": "agents"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "cost_usd": 0.01,
                },
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://models.example/v1/"
    )
    provider = OpenAICompatibleProvider(
        model="configured", api_key="secret", client=client, request_options={"seed": 7}
    )
    tools = [{"type": "function", "function": {"name": "search"}}]
    response = await provider.complete(
        [Message(role=Role.USER, content="research agents")], tools=tools
    )

    assert captured["model"] == "configured"
    assert captured["seed"] == 7
    assert captured["stream"] is False
    assert captured["tools"] == tools
    assert captured["authorization"] == "Bearer secret"
    assert response.model == "local-model"
    assert response.finish_reason == "tool_calls"
    assert response.usage.total_tokens == 13
    assert response.tool_calls[0].arguments == {"query": "agents"}
    await client.aclose()


@pytest.mark.asyncio
async def test_openai_compatible_stream_decodes_sse() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        body = (
            'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
            "event: ignored\n"
            'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=body)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://models.example/v1/"
    )
    provider = OpenAICompatibleProvider(model="test", client=client)
    chunks = [chunk async for chunk in provider.stream([Message(role=Role.USER, content="hello")])]

    assert chunks == ["hel", "lo"]
    await client.aclose()


@pytest.mark.asyncio
async def test_openai_compatible_normalizes_remote_errors() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://models.example/v1/"
    )
    provider = OpenAICompatibleProvider(model="test", client=client)

    with pytest.raises(ModelProviderError) as caught:
        await provider.complete([Message(role=Role.USER, content="hello")])

    assert str(caught.value) == "slow down"
    assert caught.value.status_code == 429
    assert caught.value.retryable
    await client.aclose()


@pytest.mark.asyncio
async def test_openai_compatible_rejects_invalid_tool_arguments() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": "bad", "function": {"name": "search", "arguments": "[1]"}}
                    ]
                }
            }
        ]
    }
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response)),
        base_url="https://models.example/v1/",
    )
    provider = OpenAICompatibleProvider(model="test", client=client)

    with pytest.raises(ModelProviderError, match="invalid tool call"):
        await provider.complete([Message(role=Role.USER, content="hello")])
    await client.aclose()


@pytest.mark.asyncio
async def test_model_factory_defaults_offline_and_selects_remote_provider() -> None:
    assert isinstance(create_model_provider(Settings()), MockModelProvider)

    remote = create_model_provider(
        Settings(
            model_provider="openai-compatible",
            model_name="local-model",
            model_base_url="http://localhost:1234/v1",
        )
    )
    assert isinstance(remote, OpenAICompatibleProvider)
    assert remote.model == "local-model"
    await remote.aclose()

    with pytest.raises(ValueError, match="unknown model provider"):
        create_model_provider(Settings(model_provider="mystery"))
