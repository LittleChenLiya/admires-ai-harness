"""Policy-enforcing wrapper for native or MCP tool executors."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any
from uuid import UUID

from admires_harness.contracts import ApprovalRequest, JsonValue, ToolCall, ToolResult
from admires_harness.ports import ToolExecutor
from admires_harness.safety.approval import (
    ApprovalError,
    ApprovalGate,
    approval_arguments_digest,
)
from admires_harness.safety.policy import PolicyViolation, SafetyPolicy


class SecuredToolExecutor:
    """Apply policy and one-shot approvals before invoking an executor.

    The wrapper is the intended adapter boundary: native tools and MCP tools must not be
    handed directly to a model-facing agent. Approval consumption happens immediately
    before execution. If a process crashes after consumption but during the tool call, the
    stable idempotency key is the recovery mechanism; the approval must not be replayed.

    Every failure path returns a failed ToolResult with a distinct error_code instead of
    raising, so a misbehaving adapter can never bypass the policy or leave an approval
    consumed without an observable outcome:

    - policy_denied: the tool is unknown, denied, or its capability is disabled.
    - approval_required: the tool needs approval but none was supplied.
    - approval_<code>: the gate rejected the request (not_found, not_decided, denied,
      already_consumed, expired, binding_mismatch).
    - invalid_arguments: the arguments cannot be canonically encoded for binding.
    - approval_consume_failed: the consumption-persistence callback failed. The
      one-shot transition is rolled back first, so the result is retryable and the
      approval remains unconsumed.
    - tool_execution_failed: the wrapped executor raised instead of returning.
    """

    def __init__(
        self,
        executor: ToolExecutor,
        *,
        policy: SafetyPolicy,
        approvals: ApprovalGate,
    ) -> None:
        self._executor = executor
        self._policy = policy
        self._approvals = approvals

    def schemas(self) -> Sequence[Mapping[str, Any]]:
        return self._executor.schemas()

    async def execute(
        self,
        call: ToolCall,
        *,
        run_id: UUID,
        idempotency_key: str,
        checkpoint_revision: int = 0,
        approval_request_id: UUID | None = None,
        on_approval_consumed: Callable[[ApprovalRequest], Awaitable[None]] | None = None,
    ) -> ToolResult:
        decision = self._policy.evaluate(call.name)
        if not decision.allowed:
            return ToolResult(
                ok=False,
                content=decision.reason,
                error_code="policy_denied",
            )
        approval_consumed = False
        if decision.requires_approval:
            if approval_request_id is None:
                return ToolResult(
                    ok=False,
                    content="a valid one-shot human approval is required",
                    error_code="approval_required",
                )
            try:
                arguments = self._json_arguments(call.arguments)
            except ValueError as error:
                return ToolResult(ok=False, content=str(error), error_code="invalid_arguments")
            try:
                await self._approvals.consume(
                    approval_request_id,
                    run_id=run_id,
                    checkpoint_revision=checkpoint_revision,
                    tool_name=call.name,
                    arguments=arguments,
                    on_consumed=on_approval_consumed,
                )
            except ApprovalError as error:
                return ToolResult(
                    ok=False,
                    content=str(error),
                    error_code=f"approval_{error.code}",
                )
            except Exception as error:
                # The gate rolls the one-shot transition back when the persistence
                # callback fails, so the approval is still unconsumed and a retry
                # with the same request is safe.
                return ToolResult(
                    ok=False,
                    content=f"approval consumption was not persisted: {error}",
                    error_code="approval_consume_failed",
                    retryable=True,
                )
            approval_consumed = True
        try:
            self._policy.authorize(call.name, approval_consumed=approval_consumed)
        except PolicyViolation as error:
            return ToolResult(ok=False, content=error.reason, error_code="policy_denied")
        try:
            return await self._executor.execute(
                call, run_id=run_id, idempotency_key=idempotency_key
            )
        except Exception as error:
            return ToolResult(
                ok=False,
                content=str(error) or "tool execution failed",
                error_code="tool_execution_failed",
            )

    @staticmethod
    def _json_arguments(arguments: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        # Force the canonical encoding before any approval state is touched so
        # non-JSON values (for example NaN) fail as invalid arguments.
        approval_arguments_digest(arguments)
        return arguments
