from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from admires_harness.adapters.models import (
    MockModelProvider,
    ModelProviderError,
    OpenAICompatibleProvider,
    StreamAccumulator,
    create_model_provider,
)
from admires_harness.config import Settings
from admires_harness.contracts import Message, ModelResponse, ModelUsage, Role, ToolCall


def _sse(*events: object, done: bool = True) -> str:
    lines = []
    for event in events:
        payload = event if isinstance(event, str) else json.dumps(event)
        lines.append(f"data: {payload}")
    if done:
        lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


def _provider(
    handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any
) -> tuple[OpenAICompatibleProvider, httpx.AsyncClient]:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://models.example/v1/",
    )
    provider = OpenAICompatibleProvider(model="configured", client=client, **kwargs)
    return provider, client


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


@pytest.mark.asyncio
async def test_complete_normalizes_object_and_part_list_content() -> None:
    payloads = [
        {"content": {"text": "object content"}},
        {"content": [{"type": "text", "text": "part "}, {"text": "list"}, {"type": "image"}]},
        {"content": None},
    ]
    responses = iter(
        httpx.Response(200, json={"choices": [{"message": message}]}) for message in payloads
    )
    provider, client = _provider(lambda request: next(responses))

    contents = [
        (await provider.complete([Message(role=Role.USER, content="hi")])).content for _ in payloads
    ]

    assert contents == ["object content", "part list", ""]
    await client.aclose()


@pytest.mark.asyncio
async def test_complete_missing_usage_and_finish_reason_defaults() -> None:
    provider, client = _provider(
        lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    )

    response = await provider.complete([Message(role=Role.USER, content="hi")])

    assert response.usage == ModelUsage()
    assert response.finish_reason == "stop"
    assert response.model == "configured"
    await client.aclose()


def test_stream_accumulator_merges_empty_and_duplicate_chunks() -> None:
    accumulator = StreamAccumulator(default_model="fallback")
    first = {"id": "chunk-1", "choices": [{"delta": {"content": "hel"}}]}

    assert accumulator.add(first) == "hel"
    assert accumulator.add({"id": "chunk-empty", "choices": []}) is None
    assert accumulator.add({}) is None
    assert accumulator.add(first) is None  # replayed event is dropped
    assert accumulator.add({"id": "chunk-2", "choices": [{"delta": {"content": "lo"}}]}) == "lo"

    response = accumulator.finalize()
    assert response.content == "hello"
    assert response.model == "fallback"
    assert accumulator.chunks_seen == 5
    assert accumulator.chunks_merged == 4


def test_stream_accumulator_concatenates_tool_call_arguments_across_chunks() -> None:
    accumulator = StreamAccumulator()
    accumulator.add(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "id": "call-9", "function": {"name": "search"}}]
                    }
                }
            ]
        }
    )
    accumulator.add(
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"que'}}]}}]}
    )
    accumulator.add(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": 'ry": "agents"}'}}]
                    }
                }
            ]
        }
    )
    accumulator.add({"choices": [{"finish_reason": "tool_calls"}]})

    response = accumulator.finalize()

    assert response.finish_reason == "tool_calls"
    assert response.tool_calls == (
        ToolCall(id="call-9", name="search", arguments={"query": "agents"}),
    )


def test_stream_accumulator_rejects_invalid_tool_arguments() -> None:
    accumulator = StreamAccumulator()
    accumulator.add(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"name": "search", "arguments": "[1,"}}
                        ]
                    }
                }
            ]
        }
    )
    accumulator.add(
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "2]"}}]}}]}
    )

    with pytest.raises(ModelProviderError, match="invalid tool call"):
        accumulator.finalize()


def test_stream_accumulator_requires_tool_call_name() -> None:
    accumulator = StreamAccumulator()
    accumulator.add(
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}}]}
    )

    with pytest.raises(ModelProviderError, match="invalid tool call"):
        accumulator.finalize()


def test_stream_accumulator_rejects_malformed_events() -> None:
    accumulator = StreamAccumulator()
    with pytest.raises(ModelProviderError, match="invalid stream event"):
        accumulator.add({"choices": "not-a-list"})
    with pytest.raises(ModelProviderError, match="invalid stream event"):
        accumulator.add({"choices": [{"delta": "not-a-mapping"}]})
    with pytest.raises(ModelProviderError, match="invalid stream event"):
        accumulator.add({"choices": [{"delta": {"tool_calls": [{"index": "x"}]}}]})


def test_stream_accumulator_usage_is_last_wins_never_double_counted() -> None:
    accumulator = StreamAccumulator()
    usage_chunk = {
        "id": "usage",
        "choices": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "cost_usd": 0.02},
    }
    accumulator.add({"id": "partial", "choices": [], "usage": {"prompt_tokens": 8}})
    accumulator.add(usage_chunk)
    accumulator.add(usage_chunk)  # replayed usage event must not re-apply

    response = accumulator.finalize()

    assert response.usage == ModelUsage(input_tokens=10, output_tokens=4, cost_usd=0.02)
    assert response.usage.total_tokens == 14


def test_stream_accumulator_missing_usage_degrades_to_zeroes() -> None:
    accumulator = StreamAccumulator()
    accumulator.add({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]})

    response = accumulator.finalize()

    assert response.usage == ModelUsage()
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_stream_merges_chunks_and_records_final_response() -> None:
    tool_start = {
        "id": "chatcmpl-1",
        "model": "remote-model",
        "choices": [
            {
                "delta": {
                    "tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "search"}}]
                }
            }
        ],
    }
    body = _sse(
        {"id": "chatcmpl-1", "model": "remote-model", "choices": [{"delta": {"content": "hel"}}]},
        {"id": "chatcmpl-1", "model": "remote-model", "choices": [{"delta": {"content": "lo"}}]},
        {"id": "chatcmpl-1", "model": "remote-model", "choices": [{"delta": {"content": "lo"}}]},
        tool_start,
        tool_start,  # duplicate replay must be dropped
        {
            "id": "chatcmpl-1",
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"q": 1}'}}]}}
            ],
        },
        {"id": "chatcmpl-1", "choices": [{"finish_reason": "tool_calls"}]},
        {
            "id": "chatcmpl-1",
            "choices": [],
            "usage": {"prompt_tokens": 7, "completion_tokens": 5, "cost_usd": 0.01},
        },
    )
    provider, client = _provider(lambda request: httpx.Response(200, text=body))

    chunks = [chunk async for chunk in provider.stream([Message(role=Role.USER, content="hi")])]

    assert chunks == ["hel", "lo"]
    final = provider.last_stream_response
    assert final is not None
    assert final.content == "hello"
    assert final.model == "remote-model"
    assert final.finish_reason == "tool_calls"
    assert final.usage == ModelUsage(input_tokens=7, output_tokens=5, cost_usd=0.01)
    assert final.tool_calls == (ToolCall(id="call-1", name="search", arguments={"q": 1}),)
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_rejects_invalid_sse_events() -> None:
    provider, client = _provider(lambda request: httpx.Response(200, text=_sse("not-json")))

    with pytest.raises(ModelProviderError, match="invalid stream event"):
        [chunk async for chunk in provider.stream([Message(role=Role.USER, content="hi")])]
    await client.aclose()


@pytest.mark.asyncio
async def test_provider_timeout_and_transport_errors_are_normalized() -> None:
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow", request=request)

    def transport_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    timeout_provider, timeout_client = _provider(timeout_handler)
    with pytest.raises(ModelProviderError) as timeout_error:
        await timeout_provider.complete([Message(role=Role.USER, content="hi")])
    assert str(timeout_error.value) == "model provider timed out"
    assert timeout_error.value.retryable
    assert timeout_error.value.status_code is None

    with pytest.raises(ModelProviderError, match="timed out"):
        [chunk async for chunk in timeout_provider.stream([Message(role=Role.USER, content="hi")])]

    down_provider, down_client = _provider(transport_handler)
    with pytest.raises(ModelProviderError) as down_error:
        await down_provider.complete([Message(role=Role.USER, content="hi")])
    assert str(down_error.value) == "model provider is unavailable"
    assert down_error.value.retryable
    await timeout_client.aclose()
    await down_client.aclose()


@pytest.mark.asyncio
async def test_error_detail_never_leaks_prompts_or_credentials() -> None:
    secret_prompt = "my confidential prompt"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={
                "error": {"message": f"backend failed on {secret_prompt} with key top-secret-key"}
            },
        )

    provider, client = _provider(handler, api_key="top-secret-key")

    with pytest.raises(ModelProviderError) as caught:
        await provider.complete([Message(role=Role.USER, content=secret_prompt)])

    assert str(caught.value) == "model provider returned HTTP 500"
    assert secret_prompt not in str(caught.value)
    assert "top-secret-key" not in str(caught.value)
    assert caught.value.status_code == 500
    assert caught.value.retryable
    await client.aclose()


@pytest.mark.asyncio
async def test_client_errors_are_not_retryable_and_detail_is_bounded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "x" * 500}})

    provider, client = _provider(handler)

    with pytest.raises(ModelProviderError) as caught:
        await provider.complete([Message(role=Role.USER, content="hi")])

    assert len(str(caught.value)) == 200
    assert caught.value.status_code == 400
    assert not caught.value.retryable
    await client.aclose()


def test_model_response_serialization_is_stable() -> None:
    response = ModelResponse(
        content="hello",
        tool_calls=(ToolCall(id="call-1", name="search", arguments={"q": "agents"}),),
        usage=ModelUsage(input_tokens=3, output_tokens=2, cost_usd=0.5),
        finish_reason="tool_calls",
        model="fixture",
    )

    assert response.model_dump() == {
        "content": "hello",
        "tool_calls": ({"id": "call-1", "name": "search", "arguments": {"q": "agents"}},),
        "usage": {"input_tokens": 3, "output_tokens": 2, "cost_usd": 0.5},
        "finish_reason": "tool_calls",
        "model": "fixture",
    }
    restored = ModelResponse.model_validate_json(response.model_dump_json())
    assert restored == response
    assert restored.model_dump_json() == response.model_dump_json()
