from __future__ import annotations

import json

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


def test_stream_accumulator_merges_fragmented_chunks_without_double_counting() -> None:
    accumulator = StreamAccumulator(default_model="fallback")
    opening = {"model": "stream-model", "choices": [{"delta": {"content": "Hel"}}]}
    final_usage = {"usage": {"prompt_tokens": 5, "completion_tokens": 2, "cost_usd": 0.5}}

    accumulator.add(opening)
    accumulator.add(opening)  # replayed duplicate is skipped
    accumulator.add({})  # empty keep-alive chunk is harmless
    accumulator.add({"choices": [{"delta": {"content": "lo"}}]})
    accumulator.add(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {"name": "search", "arguments": '{"que'},
                            }
                        ]
                    }
                }
            ]
        }
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
    accumulator.add({"usage": {"prompt_tokens": 5, "completion_tokens": 1}})
    accumulator.add(final_usage)  # latest snapshot wins, values are never summed
    accumulator.add(final_usage)  # repeated snapshot is skipped
    accumulator.add({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})

    response = accumulator.build()

    assert response == ModelResponse(
        content="Hello",
        tool_calls=(ToolCall(id="call-1", name="search", arguments={"query": "agents"}),),
        usage=ModelUsage(input_tokens=5, output_tokens=2, cost_usd=0.5),
        finish_reason="tool_calls",
        model="stream-model",
    )
    assert accumulator.chunks_seen == 10
    assert accumulator.chunks_merged == 8
    assert json.loads(response.model_dump_json()) == {
        "content": "Hello",
        "tool_calls": [{"id": "call-1", "name": "search", "arguments": {"query": "agents"}}],
        "usage": {"input_tokens": 5, "output_tokens": 2, "cost_usd": 0.5},
        "finish_reason": "tool_calls",
        "model": "stream-model",
    }


def test_stream_accumulator_normalizes_content_forms_and_defaults() -> None:
    accumulator = StreamAccumulator(default_model="fallback")
    accumulator.add({"choices": [{"delta": {"content": [{"text": "a"}, "b", {"type": "image"}]}}]})

    response = accumulator.build()

    assert response.content == "ab"
    assert response.model == "fallback"
    assert response.finish_reason == "stop"
    assert response.usage == ModelUsage()
    assert response.tool_calls == ()


def test_stream_accumulator_rejects_invalid_tool_arguments() -> None:
    malformed = StreamAccumulator()
    malformed.add(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"name": "search", "arguments": "{bad"}}
                        ]
                    }
                }
            ]
        }
    )
    with pytest.raises(ModelProviderError, match="invalid tool call"):
        malformed.build()

    non_object = StreamAccumulator()
    non_object.add(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"name": "search", "arguments": "[1]"}}
                        ]
                    }
                }
            ]
        }
    )
    with pytest.raises(ModelProviderError, match="invalid tool call"):
        non_object.build()

    missing_name = StreamAccumulator()
    missing_name.add({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-1"}]}}]})
    with pytest.raises(ModelProviderError, match="invalid tool call"):
        missing_name.build()


def test_stream_accumulator_rejects_invalid_chunk_shapes() -> None:
    with pytest.raises(ModelProviderError, match="invalid content"):
        StreamAccumulator().add({"choices": [{"delta": {"content": 42}}]})
    with pytest.raises(ModelProviderError, match="invalid stream event"):
        StreamAccumulator().add({"choices": "nope"})
    with pytest.raises(ModelProviderError, match="invalid stream event"):
        StreamAccumulator().add({"choices": ["not-a-mapping"]})


def _sse_body(*chunks: object) -> str:
    frames = [
        f"data: {chunk if isinstance(chunk, str) else json.dumps(chunk)}\n\n" for chunk in chunks
    ]
    return "".join([*frames, "data: [DONE]\n\n"])


@pytest.mark.asyncio
async def test_openai_compatible_complete_stream_merges_sse_chunks() -> None:
    opening = {"model": "stream-model", "choices": [{"delta": {"content": "Hel"}}]}
    final_usage = {"usage": {"prompt_tokens": 4, "completion_tokens": 2, "cost_usd": 0.25}}
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            text=_sse_body(
                opening,
                opening,  # replayed duplicate
                {},  # empty keep-alive chunk
                {"choices": [{"delta": {"content": "lo"}}]},
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-9",
                                        "function": {"name": "search", "arguments": '{"que'},
                                    },
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": 'ry": "agents"}'}}
                                ]
                            }
                        }
                    ]
                },
                {"usage": {"prompt_tokens": 4, "completion_tokens": 1}},
                final_usage,
                final_usage,  # repeated usage frame must not double count
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ),
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://models.example/v1/"
    )
    provider = OpenAICompatibleProvider(model="configured", client=client)

    response = await provider.complete_stream([Message(role=Role.USER, content="hello")])

    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}
    assert response == ModelResponse(
        content="Hello",
        tool_calls=(ToolCall(id="call-9", name="search", arguments={"query": "agents"}),),
        usage=ModelUsage(input_tokens=4, output_tokens=2, cost_usd=0.25),
        finish_reason="tool_calls",
        model="stream-model",
    )
    await client.aclose()


@pytest.mark.asyncio
async def test_openai_compatible_complete_defaults_for_missing_fields() -> None:
    payload = {
        "choices": [{"message": {"content": [{"type": "text", "text": "hi"}, {"text": " there"}]}}]
    }
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
        base_url="https://models.example/v1/",
    )
    provider = OpenAICompatibleProvider(model="configured", client=client)

    response = await provider.complete([Message(role=Role.USER, content="hello")])

    assert response.content == "hi there"
    assert response.model == "configured"
    assert response.finish_reason == "stop"
    assert response.usage == ModelUsage()
    assert response.tool_calls == ()
    await client.aclose()


@pytest.mark.asyncio
async def test_openai_compatible_normalizes_timeout_and_transport_errors() -> None:
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow upstream", request=request)

    timeout_client = httpx.AsyncClient(
        transport=httpx.MockTransport(timeout_handler), base_url="https://models.example/v1/"
    )
    provider = OpenAICompatibleProvider(model="test", client=timeout_client)
    messages = [Message(role=Role.USER, content="my secret prompt")]

    with pytest.raises(ModelProviderError) as caught:
        await provider.complete(messages)
    assert str(caught.value) == "model provider timed out"
    assert caught.value.retryable
    assert caught.value.status_code is None
    assert "my secret prompt" not in str(caught.value)

    with pytest.raises(ModelProviderError, match="timed out"):
        async for _ in provider.stream_events(messages):
            pass
    await timeout_client.aclose()

    def connect_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    connect_client = httpx.AsyncClient(
        transport=httpx.MockTransport(connect_handler), base_url="https://models.example/v1/"
    )
    unavailable = OpenAICompatibleProvider(model="test", client=connect_client)
    with pytest.raises(ModelProviderError) as caught:
        await unavailable.complete(messages)
    assert str(caught.value) == "model provider is unavailable"
    assert caught.value.retryable
    await connect_client.aclose()


@pytest.mark.asyncio
async def test_openai_compatible_errors_never_leak_credentials() -> None:
    api_key = "sk-live-secret"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": f"bad key {api_key} for request"}})

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://models.example/v1/"
    )
    provider = OpenAICompatibleProvider(model="test", api_key=api_key, client=client)

    with pytest.raises(ModelProviderError) as caught:
        await provider.complete([Message(role=Role.USER, content="hello")])

    assert api_key not in str(caught.value)
    assert "[redacted]" in str(caught.value)
    assert caught.value.status_code == 401
    assert not caught.value.retryable
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "retryable"),
    [(400, False), (401, False), (429, True), (500, True), (503, True)],
)
async def test_openai_compatible_classifies_http_errors(status: int, retryable: bool) -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(status, json={"error": {"message": "boom"}})
        ),
        base_url="https://models.example/v1/",
    )
    provider = OpenAICompatibleProvider(model="test", client=client)

    with pytest.raises(ModelProviderError) as caught:
        await provider.complete([Message(role=Role.USER, content="hello")])

    assert str(caught.value) == "boom"
    assert caught.value.status_code == status
    assert caught.value.retryable is retryable
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("body", ["data: {not json}\n\n", "data: [1, 2]\n\n"])
async def test_openai_compatible_rejects_malformed_stream_events(body: str) -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=body)),
        base_url="https://models.example/v1/",
    )
    provider = OpenAICompatibleProvider(model="test", client=client)

    with pytest.raises(ModelProviderError, match="invalid stream event"):
        async for _ in provider.stream_events([Message(role=Role.USER, content="hello")]):
            pass
    await client.aclose()


@pytest.mark.asyncio
async def test_mock_provider_records_frozen_call_statistics() -> None:
    model = MockModelProvider(["ok"])
    messages = [Message(role=Role.USER, content="hi")]
    tools = [{"type": "function", "function": {"name": "search"}}]

    response = await model.complete(messages, tools=tools)

    assert response.content == "ok"
    assert model.calls == [(tuple(messages), tuple(tools))]


def test_model_response_serialization_is_stable() -> None:
    response = ModelResponse(
        content="hi",
        tool_calls=(ToolCall(id="call-1", name="search", arguments={"q": "agents"}),),
        usage=ModelUsage(input_tokens=3, output_tokens=2, cost_usd=0.5),
        finish_reason="tool_calls",
        model="fixture",
    )

    assert json.loads(response.model_dump_json()) == {
        "content": "hi",
        "tool_calls": [{"id": "call-1", "name": "search", "arguments": {"q": "agents"}}],
        "usage": {"input_tokens": 3, "output_tokens": 2, "cost_usd": 0.5},
        "finish_reason": "tool_calls",
        "model": "fixture",
    }
    assert ModelResponse.model_validate_json(response.model_dump_json()) == response
