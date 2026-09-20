"""Human approval requests that cannot be reused for a different action."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime, timedelta
from uuid import UUID

from admires_harness.contracts import (
    ApprovalDecision,
    ApprovalRequest,
    JsonValue,
    RiskLevel,
)


class ApprovalError(RuntimeError):
    """A stable, machine-readable approval failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def approval_arguments_digest(arguments: Mapping[str, JsonValue]) -> str:
    """Hash arguments with a canonical encoding for exact approval binding."""

    encoded = json.dumps(
        arguments,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class ApprovalGate:
    """Issue, decide, and atomically consume one-shot approvals.

    An approval is bound to the run, checkpoint revision, tool name, and canonical
    arguments. Any resume from a changed checkpoint or argument mutation requires a
    new human decision. The implementation is intentionally storage-neutral; requests
    and decisions can be reconstructed from the event log after a restart.
    """

    def __init__(
        self,
        *,
        ttl: timedelta = timedelta(minutes=15),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if ttl <= timedelta(0):
            raise ValueError("approval ttl must be positive")
        self._ttl = ttl
        self._clock = clock or (lambda: datetime.now(UTC))
        self._requests: dict[UUID, ApprovalRequest] = {}
        self._decisions: dict[UUID, ApprovalDecision] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        *,
        run_id: UUID,
        checkpoint_revision: int,
        tool_name: str,
        arguments: Mapping[str, JsonValue],
        reason: str,
        risk: RiskLevel,
    ) -> ApprovalRequest:
        if checkpoint_revision < 0:
            raise ValueError("checkpoint_revision must not be negative")
        if not tool_name.strip():
            raise ValueError("tool_name must not be empty")
        if not reason.strip():
            raise ValueError("approval reason must not be empty")
        now = self._now()
        request = ApprovalRequest(
            run_id=run_id,
            checkpoint_revision=checkpoint_revision,
            tool_name=tool_name,
            arguments_digest=approval_arguments_digest(arguments),
            reason=reason,
            risk=risk,
            expires_at=now + self._ttl,
        )
        async with self._lock:
            self._requests[request.request_id] = request
        return request

    async def restore(
        self,
        request: ApprovalRequest,
        decision: ApprovalDecision | None = None,
    ) -> None:
        """Restore approval state from trusted persisted events."""

        if decision is not None and decision.request_id != request.request_id:
            raise ValueError("decision does not belong to request")
        async with self._lock:
            if request.request_id in self._requests:
                raise ApprovalError("already_exists", "approval request already exists")
            self._requests[request.request_id] = request
            if decision is not None:
                self._decisions[request.request_id] = decision

    async def resolve(self, decision: ApprovalDecision) -> ApprovalRequest:
        """Record exactly one human decision and return its request."""

        if not decision.decided_by.strip():
            raise ValueError("decided_by must not be empty")
        async with self._lock:
            request = self._require_request(decision.request_id)
            if request.consumed_at is not None:
                raise ApprovalError("already_consumed", "approval was already consumed")
            if decision.request_id in self._decisions:
                raise ApprovalError("already_decided", "approval was already decided")
            if self._now() >= request.expires_at:
                raise ApprovalError("expired", "approval request has expired")
            self._decisions[decision.request_id] = decision
            return request

    async def consume(
        self,
        request_id: UUID,
        *,
        run_id: UUID,
        checkpoint_revision: int,
        tool_name: str,
        arguments: Mapping[str, JsonValue],
        on_consumed: Callable[[ApprovalRequest], Awaitable[None]] | None = None,
    ) -> ApprovalDecision:
        """Verify all bindings and atomically consume an approved decision."""

        async with self._lock:
            request = self._require_request(request_id)
            decision = self._decisions.get(request_id)
            if decision is None:
                raise ApprovalError("not_decided", "approval has not been decided")
            if not decision.approved:
                raise ApprovalError("denied", "the human reviewer denied this action")
            if request.consumed_at is not None:
                raise ApprovalError("already_consumed", "approval was already consumed")
            if self._now() >= request.expires_at:
                raise ApprovalError("expired", "approval request has expired")
            if request.run_id != run_id:
                raise ApprovalError("binding_mismatch", "approval belongs to a different run")
            if request.checkpoint_revision != checkpoint_revision:
                raise ApprovalError("binding_mismatch", "checkpoint changed after approval")
            if request.tool_name != tool_name:
                raise ApprovalError("binding_mismatch", "approval is for a different tool")
            digest = approval_arguments_digest(arguments)
            if request.arguments_digest != digest:
                raise ApprovalError("binding_mismatch", "tool arguments changed after approval")
            consumed = request.model_copy(update={"consumed_at": self._now()})
            self._requests[request_id] = consumed
            if on_consumed is not None:
                # Persist the one-shot transition before the side effect starts. A
                # failed persistence callback prevents execution and fails closed.
                try:
                    await on_consumed(consumed)
                except Exception as exc:
                    self._requests[request_id] = request
                    raise ApprovalError(
                        "callback_failed",
                        "approval consumption callback failed; consumption rolled back",
                    ) from exc
            return decision

    async def get(self, request_id: UUID) -> tuple[ApprovalRequest, ApprovalDecision | None] | None:
        async with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                return None
            return request, self._decisions.get(request_id)

    def _require_request(self, request_id: UUID) -> ApprovalRequest:
        request = self._requests.get(request_id)
        if request is None:
            raise ApprovalError("not_found", "approval request was not found")
        return request

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("approval clock must return a timezone-aware datetime")
        return now.astimezone(UTC)
