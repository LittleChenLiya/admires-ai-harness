"""Model-provider adapters.

The mock provider is deliberately capable enough to drive complete offline demos.  The
OpenAI-compatible adapter only depends on HTTP, so it works with hosted OpenAI models as
well as local servers implementing the chat-completions protocol.

Normalization rules shared by both providers:

* ``content`` may be missing, ``None``, a plain string, or a list of text parts; all
  forms collapse to a single string.
* ``tool_calls`` may arrive whole (non-streaming) or fragmented across stream chunks;
  fragments are merged by chunk ``index`` and their JSON argument strings concatenated
  before parsing.  Arguments that are not a JSON object are rejected.
* ``usage`` may be missing or repeated; the latest reported snapshot wins so tokens and
  cost are never double counted.
* Provider exceptions are converted to :class:`ModelProviderError` with sanitized
  messages that never include credentials or request payloads.
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


def _content_to_text(content: Any) -> str:
    """Collapse provider content (missing, string, or text-part list) into a string."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping):
                text = part.get("text")
                if text is not None:
                    parts.append(str(text))
            else:
                raise ModelProviderError("model provider returned invalid content")
        return "".join(parts)
    raise ModelProviderError("model provider returned invalid content")


def _parse_tool_call(raw_call: Any) -> ToolCall:
    """Validate one non-streaming tool call and convert it to the shared contract."""

    try:
        if not isinstance(raw_call, Mapping):
            raise TypeError
        function = raw_call["function"]
        if not isinstance(function, Mapping):
            raise TypeError
        raw_arguments = function.get("arguments", "{}")
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        if not isinstance(arguments, dict):
            raise TypeError
        return ToolCall(
            id=str(raw_call["id"]),
            name=str(function["name"]),
            arguments=arguments,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ModelProviderError("model provider returned an invalid tool call") from exc


def _coerce_count(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _coerce_cost(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _normalize_usage(raw_usage: Any) -> ModelUsage:
    """Convert a provider usage object; missing or malformed fields become zero."""

    if not isinstance(raw_usage, Mapping):
        return ModelUsage()
    input_tokens = raw_usage.get("prompt_tokens", raw_usage.get("input_tokens", 0))
    output_tokens = raw_usage.get("completion_tokens", raw_usage.get("output_tokens", 0))
    return ModelUsage(
        input_tokens=_coerce_count(input_tokens),
        output_tokens=_coerce_count(output_tokens),
        cost_usd=_coerce_cost(raw_usage.get("cost_usd", 0)),
    )


class StreamAccumulator:
    """Merge chat-completion stream chunks into a single ``ModelResponse``.

    Chunks may carry content deltas, tool-call fragments, usage snapshots, a finish
    reason, or nothing at all (empty keep-alive chunks are ignored).  A chunk that is
    identical to the one applied immediately before it is treated as a replayed
    duplicate and skipped, so reconnections never double content or tokens.  Usage is
    tracked as the latest snapshot rather than a sum, which keeps token and cost
    accounting stable when providers repeat the final usage frame.
    """

    def __init__(self, *, default_model: str = "unknown") -> None:
        self._default_model = default_model
        self._content_parts: list[str] = []
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._usage = ModelUsage()
        self._finish_reason: str | None = None
        self._model: str | None = None
        self._previous_chunk: Mapping[str, Any] | None = None
        self.chunks_seen = 0
        self.chunks_merged = 0

    def add(self, chunk: Mapping[str, Any]) -> None:
        """Merge one raw provider chunk; duplicates and empty chunks are harmless."""

        self.chunks_seen += 1
        if chunk == self._previous_chunk:
            return
        self._previous_chunk = chunk
        self.chunks_merged += 1

        model = chunk.get("model")
        if model:
            self._model = str(model)
        usage = chunk.get("usage")
        if usage:
            self._usage = _normalize_usage(usage)

        choices = chunk.get("choices") or []
        if isinstance(choices, str) or not isinstance(choices, Sequence):
            raise ModelProviderError("model provider returned an invalid stream event")
        if not choices:
            return
        first = choices[0]
        if not isinstance(first, Mapping):
            raise ModelProviderError("model provider returned an invalid stream event")
        finish_reason = first.get("finish_reason")
        if finish_reason:
            self._finish_reason = str(finish_reason)
        delta = first.get("delta") or {}
        if not isinstance(delta, Mapping):
            raise ModelProviderError("model provider returned an invalid stream event")
        content = _content_to_text(delta.get("content"))
        if content:
            self._content_parts.append(content)
        for raw_call in delta.get("tool_calls") or []:
            self._merge_tool_call(raw_call)

    def _merge_tool_call(self, raw_call: Any) -> None:
        if not isinstance(raw_call, Mapping):
            raise ModelProviderError("model provider returned an invalid tool call")
        index = _coerce_count(raw_call.get("index", 0))
        entry = self._tool_calls.setdefault(index, {"id": None, "name": None, "arguments": []})
        if raw_call.get("id"):
            entry["id"] = str(raw_call["id"])
        function = raw_call.get("function") or {}
        if not isinstance(function, Mapping):
            raise ModelProviderError("model provider returned an invalid tool call")
        if function.get("name"):
            entry["name"] = str(function["name"])
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            entry["arguments"].append(arguments)
        elif isinstance(arguments, Mapping):
            entry["arguments"].append(json.dumps(arguments))
        elif arguments is not None:
            raise ModelProviderError("model provider returned an invalid tool call")

    def build(self) -> ModelResponse:
        """Materialize the merged response, validating tool-call arguments as JSON."""

        tool_calls: list[ToolCall] = []
        for index in sorted(self._tool_calls):
            entry = self._tool_calls[index]
            raw_arguments = "".join(entry["arguments"]) or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise ModelProviderError("model provider returned an invalid tool call") from exc
            if not isinstance(arguments, dict) or not entry["name"]:
                raise ModelProviderError("model provider returned an invalid tool call")
            call: dict[str, Any] = {"name": entry["name"], "arguments": arguments}
            if entry["id"]:
                call["id"] = entry["id"]
            tool_calls.append(ToolCall(**call))
        return ModelResponse(
            content="".join(self._content_parts),
            tool_calls=tuple(tool_calls),
            usage=self._usage,
            finish_reason=self._finish_reason or "stop",
            model=self._model or self._default_model,
        )


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
        self._api_key = api_key
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
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _sanitize(self, message: str) -> str:
        """Remove credentials from a message before it reaches callers or logs."""

        if self._api_key:
            message = message.replace(self._api_key, "[redacted]")
        return message

    def _raise_for_status(self, response: httpx.Response) -> None:
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
                self._sanitize(message),
                status_code=status,
                retryable=status == 429 or status >= 500,
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

        return ModelResponse(
            content=_content_to_text(message.get("content")),
            tool_calls=tuple(
                _parse_tool_call(raw_call) for raw_call in message.get("tool_calls") or []
            ),
            usage=_normalize_usage(data.get("usage")),
            finish_reason=str(choice.get("finish_reason", "stop")),
            model=str(data.get("model", self.model)),
        )

    async def stream_events(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> AsyncIterator[Mapping[str, Any]]:
        """Yield raw decoded stream chunks, normalizing transport failures."""

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
                    except json.JSONDecodeError as exc:
                        raise ModelProviderError(
                            "model provider returned an invalid stream event"
                        ) from exc
                    if not isinstance(chunk, Mapping):
                        raise ModelProviderError("model provider returned an invalid stream event")
                    yield cast(Mapping[str, Any], chunk)
        except httpx.TimeoutException as exc:
            raise ModelProviderError("model provider timed out", retryable=True) from exc
        except httpx.TransportError as exc:
            raise ModelProviderError("model provider is unavailable", retryable=True) from exc

    async def complete_stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> ModelResponse:
        """Consume the streaming endpoint and return one merged ``ModelResponse``."""

        accumulator = StreamAccumulator(default_model=self.model)
        async for chunk in self.stream_events(messages, tools=tools):
            accumulator.add(chunk)
        return accumulator.build()

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> AsyncIterator[str]:
        async for chunk in self.stream_events(messages, tools=tools):
            try:
                content = _content_to_text(
                    cast(Mapping[str, Any], chunk["choices"][0]["delta"]).get("content")
                )
            except (KeyError, IndexError, TypeError):
                continue
            if content:
                yield content


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
