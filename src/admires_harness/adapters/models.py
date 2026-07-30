"""Model-provider adapters.

The mock provider is deliberately capable enough to drive complete offline demos.  The
OpenAI-compatible adapter only depends on HTTP, so it works with hosted OpenAI models as
well as local servers implementing the chat-completions protocol.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Any, cast

import httpx

from admires_harness.config import Settings
from admires_harness.contracts import (
    JsonValue,
    Message,
    ModelResponse,
    ModelUsage,
    Role,
    ToolCall,
)
from admires_harness.ports import ModelProvider

type MockResponder = Callable[[Sequence[Message], Sequence[Mapping[str, Any]]], ModelResponse]


class ModelProviderError(RuntimeError):
    """A normalized failure raised by a remote model provider."""

    def __init__(self, message: str, *, status_code: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class MockModelProvider:
    """Deterministic offline model with scripted or rule-based responses.

    A response script is consumed in FIFO order. Once it is exhausted, ``responder`` is
    used; the default responder simply echoes the most recent user message. Calls are
    recorded for assertions, tutorials, and evaluation reports.
    """

    name = "mock"

    def __init__(
        self,
        responses: Sequence[ModelResponse | str] = (),
        *,
        responder: MockResponder | None = None,
        stream_chunk_size: int = 12,
        stream_delay: float = 0.0,
    ) -> None:
        if stream_chunk_size < 1:
            raise ValueError("stream_chunk_size must be positive")
        if stream_delay < 0:
            raise ValueError("stream_delay cannot be negative")
        self._responses = deque(
            item
            if isinstance(item, ModelResponse)
            else ModelResponse(content=item, model=self.name)
            for item in responses
        )
        self._responder = responder or self._echo_response
        self._stream_chunk_size = stream_chunk_size
        self._stream_delay = stream_delay
        self.calls: list[tuple[tuple[Message, ...], tuple[Mapping[str, Any], ...]]] = []

    @staticmethod
    def _echo_response(
        messages: Sequence[Message], tools: Sequence[Mapping[str, Any]]
    ) -> ModelResponse:
        del tools
        prompt = next(
            (message.content for message in reversed(messages) if message.role == Role.USER), ""
        )
        return ModelResponse(
            content=f"Mock response: {prompt}",
            usage=ModelUsage(
                input_tokens=sum(len(message.content.split()) for message in messages),
                output_tokens=len(prompt.split()) + 2,
            ),
            model=MockModelProvider.name,
        )

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> ModelResponse:
        frozen_messages = tuple(messages)
        frozen_tools = tuple(tools)
        self.calls.append((frozen_messages, frozen_tools))
        if self._responses:
            return self._responses.popleft()
        return self._responder(frozen_messages, frozen_tools)

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> AsyncIterator[str]:
        response = await self.complete(messages, tools=tools)
        for offset in range(0, len(response.content), self._stream_chunk_size):
            if self._stream_delay:
                await asyncio.sleep(self._stream_delay)
            yield response.content[offset : offset + self._stream_chunk_size]


class OpenAICompatibleProvider:
    """Chat-completions adapter for OpenAI-compatible HTTP endpoints."""

    name = "openai-compatible"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
        extra_headers: Mapping[str, str] | None = None,
        request_options: Mapping[str, JsonValue] | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        self.model = model
        self._request_options = dict(request_options or {})
        self._owns_client = client is None
        headers = {"Accept": "application/json", **dict(extra_headers or {})}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._request_headers = headers
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/", timeout=timeout
        )

    async def __aenter__(self) -> OpenAICompatibleProvider:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @staticmethod
    def _message_payload(message: Message) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {"role": message.role.value, "content": message.content}
        if message.name is not None:
            payload["name"] = message.name
        if message.tool_call_id is not None:
            payload["tool_call_id"] = message.tool_call_id
        return payload

    def _payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[Mapping[str, Any]],
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            **self._request_options,
            "model": self.model,
            "messages": [self._message_payload(message) for message in messages],
            "stream": stream,
        }
        if tools:
            payload["tools"] = list(tools)
        return payload

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            try:
                detail = exc.response.json().get("error", {}).get("message")
            except (ValueError, AttributeError):
                detail = None
            message = str(detail or f"model provider returned HTTP {status}")
            raise ModelProviderError(
                message, status_code=status, retryable=status == 429 or status >= 500
            ) from exc

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> ModelResponse:
        try:
            response = await self._client.post(
                "chat/completions",
                json=self._payload(messages, tools, stream=False),
                headers=self._request_headers,
            )
        except httpx.TimeoutException as exc:
            raise ModelProviderError("model provider timed out", retryable=True) from exc
        except httpx.TransportError as exc:
            raise ModelProviderError("model provider is unavailable", retryable=True) from exc
        self._raise_for_status(response)
        try:
            data = cast(dict[str, Any], response.json())
            choice = cast(dict[str, Any], data["choices"][0])
            message = cast(dict[str, Any], choice["message"])
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ModelProviderError("model provider returned an invalid response") from exc

        tool_calls: list[ToolCall] = []
        for raw_call in message.get("tool_calls") or []:
            try:
                function = raw_call["function"]
                raw_arguments = function.get("arguments", "{}")
                arguments = (
                    json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                )
                if not isinstance(arguments, dict):
                    raise TypeError
                tool_calls.append(
                    ToolCall(
                        id=str(raw_call["id"]),
                        name=str(function["name"]),
                        arguments=arguments,
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ModelProviderError("model provider returned an invalid tool call") from exc

        usage = data.get("usage") or {}
        return ModelResponse(
            content=message.get("content") or "",
            tool_calls=tuple(tool_calls),
            usage=ModelUsage(
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
                cost_usd=float(usage.get("cost_usd", 0)),
            ),
            finish_reason=str(choice.get("finish_reason", "stop")),
            model=str(data.get("model", self.model)),
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> AsyncIterator[str]:
        try:
            async with self._client.stream(
                "POST",
                "chat/completions",
                json=self._payload(messages, tools, stream=True),
                headers=self._request_headers,
            ) as response:
                self._raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                        content = chunk["choices"][0]["delta"].get("content")
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                        raise ModelProviderError(
                            "model provider returned an invalid stream event"
                        ) from exc
                    if content:
                        yield str(content)
        except httpx.TimeoutException as exc:
            raise ModelProviderError("model provider timed out", retryable=True) from exc
        except httpx.TransportError as exc:
            raise ModelProviderError("model provider is unavailable", retryable=True) from exc


def create_model_provider(settings: Settings) -> ModelProvider:
    """Create the configured built-in provider with mock as the safe offline default."""

    provider = settings.model_provider.strip().casefold()
    if provider == "mock":
        return MockModelProvider()
    if provider in {"openai", "openai-compatible"}:
        return OpenAICompatibleProvider(
            model=settings.model_name,
            api_key=settings.model_api_key,
            base_url=settings.model_base_url,
        )
    raise ValueError(
        f"unknown model provider {settings.model_provider!r}; "
        "expected 'mock' or 'openai-compatible'"
    )
