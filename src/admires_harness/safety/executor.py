"""Policy-enforcing wrapper for native or MCP tool executors."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any
from uuid import UUID

from admires_harness.contracts import ApprovalRequest, JsonValue, ToolCall, ToolResult
from admires_harness.ports import ToolExecutor
from admires_harness.safety.approval import ApprovalError, ApprovalGate
from admires_harness.safety.policy import PolicyViolation, SafetyPolicy


class SecuredToolExecutor:
    """Apply policy and one-shot approvals before invoking an executor.

    The wrapper is the intended adapter boundary: native tools and MCP tools must not be
    handed directly to a model-facing agent. Approval consumption happens immediately
    before execution. If a process crashes after consumption but during the tool call, the
    stable idempotency key is the recovery mechanism; the approval must not be replayed.
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
                await self._approvals.consume(
                    approval_request_id,
                    run_id=run_id,
                    checkpoint_revision=checkpoint_revision,
                    tool_name=call.name,
                    arguments=self._json_arguments(call.arguments),
                    on_consumed=on_approval_consumed,
                )
            except ApprovalError as error:
                return ToolResult(
                    ok=False,
                    content=str(error),
                    error_code=f"approval_{error.code}",
                )
            approval_consumed = True
        try:
            self._policy.authorize(call.name, approval_consumed=approval_consumed)
        except PolicyViolation as error:
            return ToolResult(ok=False, content=error.reason, error_code="policy_denied")
        return await self._executor.execute(call, run_id=run_id, idempotency_key=idempotency_key)

    @staticmethod
    def _json_arguments(arguments: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        return arguments
