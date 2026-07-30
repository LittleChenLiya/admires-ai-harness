"""Bounded in-run conversational memory."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from admires_harness.contracts import Message


class ShortTermMemory:
    """Keep a bounded tail of messages without ever splitting a message.

    The budget is expressed in characters so enforcement is provider independent and
    deterministic.  A caller may reserve a system prompt separately; this class never
    silently summarizes or sends data to a model.
    """

    def __init__(self, *, max_messages: int = 50, max_characters: int = 32_000) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be positive")
        if max_characters < 1:
            raise ValueError("max_characters must be positive")
        self._max_messages = max_messages
        self._max_characters = max_characters
        self._messages: deque[Message] = deque()
        self._characters = 0

    def append(self, message: Message) -> None:
        """Append one message and evict the oldest entries to meet both limits."""

        self._messages.append(message)
        self._characters += len(message.content)
        self._trim()

    def extend(self, messages: Iterable[Message]) -> None:
        for message in messages:
            self.append(message)

    def snapshot(self) -> tuple[Message, ...]:
        return tuple(self._messages)

    def clear(self) -> None:
        self._messages.clear()
        self._characters = 0

    def __len__(self) -> int:
        return len(self._messages)

    @property
    def character_count(self) -> int:
        return self._characters

    def _trim(self) -> None:
        while self._messages and (
            len(self._messages) > self._max_messages or self._characters > self._max_characters
        ):
            removed = self._messages.popleft()
            self._characters -= len(removed.content)
