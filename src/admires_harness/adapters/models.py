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


def _normalize_content(value: Any) -> str:
    """Normalize provider content that may be missing, an object, or a part list."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        text = value.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(value, Sequence):
        parts: list[str] = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(value)


def _normalize_usage(raw: Any) -> ModelUsage:
    """Normalize a usage object; missing or malformed usage degrades to zeroes."""

    if not isinstance(raw, Mapping):
        return ModelUsage()

    def _count(*keys: str) -> int:
        for key in keys:
            value = raw.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return max(0, int(value))
        return 0

    cost = raw.get("cost_usd")
    return ModelUsage(
        input_tokens=_count("prompt_tokens", "input_tokens"),
        output_tokens=_count("completion_tokens", "output_tokens"),
        cost_usd=max(0.0, float(cost))
        if isinstance(cost, (int, float)) and not isinstance(cost, bool)
        else 0.0,
    )


def _parse_tool_arguments(raw_arguments: Any) -> dict[str, JsonValue]:
    """Parse tool-call arguments, rejecting anything that is not a JSON object."""

    arguments = raw_arguments if raw_arguments not in (None, "") else "{}"
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise TypeError("tool call arguments must be a JSON object")
    return cast(dict[str, JsonValue], arguments)


def _normalize_tool_calls(raw_calls: Any) -> tuple[ToolCall, ...]:
    """Normalize complete-response tool calls, rejecting invalid arguments."""

    tool_calls: list[ToolCall] = []
    for raw_call in raw_calls or []:
        try:
            function = raw_call["function"]
            tool_calls.append(
                ToolCall(
                    id=str(raw_call["id"]),
                    name=str(function["name"]),
                    arguments=_parse_tool_arguments(function.get("arguments", "{}")),
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelProviderError("model provider returned an invalid tool call") from exc
    return tuple(tool_calls)


def _sanitize_detail(detail: Any, secrets: Sequence[str]) -> str | None:
    """Return a bounded provider error detail, or None if it leaks protected input."""

    if not detail:
        return None
    text = str(detail)
    for secret in secrets:
        if secret and secret in text:
            return None
    return text[:200]


class StreamAccumulator:
    """Merge chat-completion stream chunks into one stable ModelResponse.

    Chunks may arrive empty, malformed, or duplicated (retries and proxies re-send
    events). Content and tool-call argument fragments are concatenated in arrival
    order; finish reason and model are last-wins. Usage fields merge key-wise with
    later cumulative values replacing earlier ones, so token and cost totals are
    never double counted even when usage is reported repeatedly or split across
    chunks. Byte-identical chunks are dropped, which absorbs replayed events
    without disturbing providers that reuse one event id for a whole stream.
    """

    def __init__(self, *, default_model: str = "unknown") -> None:
        self._default_model = default_model
        self._content_parts: list[str] = []
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._usage_raw: dict[str, Any] = {}
        self._finish_reason: str | None = None
        self._model: str | None = None
        self._seen_fingerprints: set[str] = set()
        self.chunks_seen = 0
        self.chunks_merged = 0

    @staticmethod
    def _fingerprint(chunk: Mapping[str, Any]) -> str:
        try:
            return json.dumps(chunk, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return repr(chunk)

    def add(self, chunk: Mapping[str, Any]) -> str | None:
        """Merge one chunk and return its text delta, or None when nothing was emitted."""

        self.chunks_seen += 1
        fingerprint = self._fingerprint(chunk)
        if fingerprint in self._seen_fingerprints:
            return None
        self._seen_fingerprints.add(fingerprint)
        self.chunks_merged += 1

        model = chunk.get("model")
        if model:
            self._model = str(model)
        usage = chunk.get("usage")
        if isinstance(usage, Mapping):
            self._usage_raw.update(usage)

        choices = chunk.get("choices") or []
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
            raise ModelProviderError("model provider returned an invalid stream event")
        if choices:
            choice = choices[0]
            if not isinstance(choice, Mapping):
                raise ModelProviderError("model provider returned an invalid stream event")
            delta = choice.get("delta") or choice.get("message") or {}
            if not isinstance(delta, Mapping):
                raise ModelProviderError("model provider returned an invalid stream event")
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                self._finish_reason = str(finish_reason)
            self._merge_tool_call_deltas(delta.get("tool_calls"))
            text = _normalize_content(delta.get("content"))
            if text:
                self._content_parts.append(text)
                return text
        return None

    def _merge_tool_call_deltas(self, raw_calls: Any) -> None:
        for raw_call in raw_calls or []:
            if not isinstance(raw_call, Mapping):
                raise ModelProviderError("model provider returned an invalid stream event")
            try:
                index = int(raw_call.get("index", 0))
            except (TypeError, ValueError) as exc:
                raise ModelProviderError("model provider returned an invalid stream event") from exc
            slot = self._tool_calls.setdefault(index, {"id": None, "name": None, "arguments": []})
            if raw_call.get("id"):
                slot["id"] = str(raw_call["id"])
            function = raw_call.get("function") or {}
            if not isinstance(function, Mapping):
                raise ModelProviderError("model provider returned an invalid stream event")
            if function.get("name"):
                slot["name"] = str(function["name"])
            arguments = function.get("arguments")
            if isinstance(arguments, str) and arguments:
                slot["arguments"].append(arguments)
            elif isinstance(arguments, Mapping):
                slot["arguments"].append(json.dumps(arguments))

    def finalize(self) -> ModelResponse:
        """Build the merged response, rejecting tool calls with invalid arguments."""

        tool_calls: list[ToolCall] = []
        for index in sorted(self._tool_calls):
            slot = self._tool_calls[index]
            try:
                arguments = _parse_tool_arguments("".join(slot["arguments"]))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ModelProviderError("model provider returned an invalid tool call") from exc
            if not slot["name"]:
                raise ModelProviderError("model provider returned an invalid tool call")
            tool_calls.append(
                ToolCall(
                    id=slot["id"] or f"call-{index}",
                    name=slot["name"],
                    arguments=arguments,
                )
            )
        return ModelResponse(
            content="".join(self._content_parts),
            tool_calls=tuple(tool_calls),
            usage=_normalize_usage(self._usage_raw),
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
        self._request_options = dict(request_options or {})
        self._owns_client = client is None
        headers = {"Accept": "application/json", **dict(extra_headers or {})}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._request_headers = headers
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/", timeout=timeout
        )
        self.last_stream_response: ModelResponse | None = None

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

    def _secrets(self, messages: Sequence[Message]) -> tuple[str, ...]:
        secrets = [message.content for message in messages if message.content]
        authorization = self._request_headers.get("Authorization")
        if authorization:
            secrets.append(authorization.removeprefix("Bearer "))
        return tuple(secrets)

    @staticmethod
    def _raise_for_status(response: httpx.Response, *, secrets: Sequence[str] = ()) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            try:
                detail = exc.response.json().get("error", {}).get("message")
            except (ValueError, AttributeError):
                detail = None
            message = _sanitize_detail(detail, secrets) or f"model provider returned HTTP {status}"
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
        except httpx.HTTPError as exc:
            raise ModelProviderError("model provider request failed", retryable=True) from exc
        self._raise_for_status(response, secrets=self._secrets(messages))
        try:
            data = cast(dict[str, Any], response.json())
            choice = cast(dict[str, Any], data["choices"][0])
            message = cast(dict[str, Any], choice["message"])
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ModelProviderError("model provider returned an invalid response") from exc
        if not isinstance(message, dict):
            raise ModelProviderError("model provider returned an invalid response")

        return ModelResponse(
            content=_normalize_content(message.get("content")),
            tool_calls=_normalize_tool_calls(message.get("tool_calls")),
            usage=_normalize_usage(data.get("usage")),
            finish_reason=str(choice.get("finish_reason") or "stop"),
            model=str(data.get("model") or self.model),
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> AsyncIterator[str]:
        accumulator = StreamAccumulator(default_model=self.model)
        self.last_stream_response = None
        try:
            async with self._client.stream(
                "POST",
                "chat/completions",
                json=self._payload(messages, tools, stream=True),
                headers=self._request_headers,
            ) as response:
                self._raise_for_status(response, secrets=self._secrets(messages))
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise ModelProviderError(
                            "model provider returned an invalid stream event"
                        ) from exc
                    if not isinstance(chunk, dict):
                        raise ModelProviderError("model provider returned an invalid stream event")
                    content = accumulator.add(chunk)
                    if content:
                        yield content
            self.last_stream_response = accumulator.finalize()
        except httpx.TimeoutException as exc:
            raise ModelProviderError("model provider timed out", retryable=True) from exc
        except httpx.TransportError as exc:
            raise ModelProviderError("model provider is unavailable", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ModelProviderError("model provider request failed", retryable=True) from exc


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
