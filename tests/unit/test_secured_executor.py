from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest

from admires_harness.contracts import (
    ApprovalDecision,
    ApprovalRequest,
    RiskLevel,
    SideEffect,
    ToolCall,
    ToolResult,
)
from admires_harness.safety import (
    ApprovalGate,
    SafetyPolicy,
    SecuredToolExecutor,
    ToolSafetySpec,
)


class ScriptedExecutor:
    """ToolExecutor fake that records calls and replays scripted outcomes."""

    def __init__(self, outcomes: Sequence[ToolResult | Exception] = ()) -> None:
        self.calls: list[ToolCall] = []
        self.events: list[str] = []
        self._outcomes = list(outcomes)

    def schemas(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": "write"}}]

    async def execute(self, call: ToolCall, *, run_id: UUID, idempotency_key: str) -> ToolResult:
        del run_id, idempotency_key
        self.calls.append(call)
        self.events.append("executed")
        outcome = (
            self._outcomes.pop(0)
            if self._outcomes
            else ToolResult(ok=True, content="done", receipt="receipt-1")
        )
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_executor(
    backend: ScriptedExecutor,
    *,
    clock: Callable[[], datetime] | None = None,
    ttl: timedelta = timedelta(minutes=15),
) -> tuple[SecuredToolExecutor, ApprovalGate]:
    gate = ApprovalGate(ttl=ttl, clock=clock)
    policy = SafetyPolicy(
        [
            ToolSafetySpec(name="write", side_effect=SideEffect.WRITE),
            ToolSafetySpec(name="read", side_effect=SideEffect.READ),
        ]
    )
    return SecuredToolExecutor(backend, policy=policy, approvals=gate), gate


async def approved_request(
    gate: ApprovalGate,
    *,
    run_id: UUID,
    checkpoint_revision: int = 3,
    tool_name: str = "write",
    arguments: Mapping[str, Any] = {"path": "report.md"},
    approved: bool = True,
) -> ApprovalRequest:
    request = await gate.create(
        run_id=run_id,
        checkpoint_revision=checkpoint_revision,
        tool_name=tool_name,
        arguments=arguments,
        reason="write the report",
        risk=RiskLevel.MEDIUM,
    )
    await gate.resolve(
        ApprovalDecision(request_id=request.request_id, approved=approved, decided_by="human")
    )
    return request


@pytest.mark.asyncio
async def test_missing_approval_never_touches_backend() -> None:
    backend = ScriptedExecutor()
    executor, _ = make_executor(backend)
    call = ToolCall(name="write", arguments={"path": "report.md"})

    result = await executor.execute(call, run_id=uuid4(), idempotency_key="k1")

    assert not result.ok
    assert result.error_code == "approval_required"
    assert not result.retryable
    assert backend.calls == []


@pytest.mark.asyncio
async def test_valid_approval_consumes_once_before_execution() -> None:
    backend = ScriptedExecutor()
    executor, gate = make_executor(backend)
    run_id = uuid4()
    call = ToolCall(name="write", arguments={"path": "report.md"})
    request = await approved_request(gate, run_id=run_id)
    consumed: list[ApprovalRequest] = []

    async def on_consumed(request: ApprovalRequest) -> None:
        consumed.append(request)
        backend.events.append("consumed")

    result = await executor.execute(
        call,
        run_id=run_id,
        idempotency_key="k1",
        checkpoint_revision=3,
        approval_request_id=request.request_id,
        on_approval_consumed=on_consumed,
    )

    assert result.ok and result.receipt == "receipt-1"
    assert backend.calls == [call]
    assert [item.request_id for item in consumed] == [request.request_id]
    assert consumed[0].consumed_at is not None
    # The one-shot transition is persisted before the side effect starts.
    assert backend.events == ["consumed", "executed"]
    state = await gate.get(request.request_id)
    assert state is not None and state[0].consumed_at is not None


@pytest.mark.asyncio
async def test_replayed_approval_is_rejected_without_execution() -> None:
    backend = ScriptedExecutor()
    executor, gate = make_executor(backend)
    run_id = uuid4()
    call = ToolCall(name="write", arguments={"path": "report.md"})
    request = await approved_request(gate, run_id=run_id)

    first = await executor.execute(
        call,
        run_id=run_id,
        idempotency_key="k1",
        checkpoint_revision=3,
        approval_request_id=request.request_id,
    )
    replay = await executor.execute(
        call,
        run_id=run_id,
        idempotency_key="k2",
        checkpoint_revision=3,
        approval_request_id=request.request_id,
    )

    assert first.ok
    assert replay.error_code == "approval_already_consumed"
    assert not replay.retryable
    assert backend.calls == [call]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override",
    [
        {"arguments": {"path": "other.md"}},
        {"checkpoint_revision": 4},
        {"run_id": "different"},
    ],
    ids=["arguments", "checkpoint_revision", "run_id"],
)
async def test_binding_mismatch_blocks_execution_and_consumption(
    override: dict[str, object],
) -> None:
    backend = ScriptedExecutor()
    executor, gate = make_executor(backend)
    run_id = uuid4()
    arguments = {"path": "report.md"}
    request = await approved_request(gate, run_id=run_id, arguments=arguments)

    call_arguments = override.get("arguments", arguments)
    checkpoint_revision = override.get("checkpoint_revision", 3)
    actual_run_id = uuid4() if override.get("run_id") == "different" else run_id
    result = await executor.execute(
        ToolCall(name="write", arguments=call_arguments),  # type: ignore[arg-type]
        run_id=actual_run_id,  # type: ignore[arg-type]
        idempotency_key="k1",
        checkpoint_revision=checkpoint_revision,  # type: ignore[arg-type]
        approval_request_id=request.request_id,
    )

    assert result.error_code == "approval_binding_mismatch"
    assert backend.calls == []
    state = await gate.get(request.request_id)
    assert state is not None and state[0].consumed_at is None


@pytest.mark.asyncio
async def test_denied_approval_blocks_execution() -> None:
    backend = ScriptedExecutor()
    executor, gate = make_executor(backend)
    run_id = uuid4()
    request = await approved_request(gate, run_id=run_id, approved=False)

    result = await executor.execute(
        ToolCall(name="write", arguments={"path": "report.md"}),
        run_id=run_id,
        idempotency_key="k1",
        checkpoint_revision=3,
        approval_request_id=request.request_id,
    )

    assert result.error_code == "approval_denied"
    assert backend.calls == []


@pytest.mark.asyncio
async def test_expired_approval_blocks_execution() -> None:
    current = datetime(2026, 1, 1, tzinfo=UTC)
    backend = ScriptedExecutor()
    executor, gate = make_executor(backend, clock=lambda: current, ttl=timedelta(seconds=30))
    run_id = uuid4()
    request = await approved_request(gate, run_id=run_id)
    current = current + timedelta(seconds=31)

    result = await executor.execute(
        ToolCall(name="write", arguments={"path": "report.md"}),
        run_id=run_id,
        idempotency_key="k1",
        checkpoint_revision=3,
        approval_request_id=request.request_id,
    )

    assert result.error_code == "approval_expired"
    assert backend.calls == []
    state = await gate.get(request.request_id)
    assert state is not None and state[0].consumed_at is None


@pytest.mark.asyncio
async def test_unknown_and_denied_tools_fail_closed() -> None:
    backend = ScriptedExecutor()
    executor, _ = make_executor(backend)

    unknown = await executor.execute(ToolCall(name="missing"), run_id=uuid4(), idempotency_key="k1")

    assert unknown.error_code == "policy_denied"
    assert backend.calls == []


@pytest.mark.asyncio
async def test_non_json_arguments_fail_before_consumption() -> None:
    backend = ScriptedExecutor()
    executor, gate = make_executor(backend)
    run_id = uuid4()
    request = await approved_request(gate, run_id=run_id)

    result = await executor.execute(
        ToolCall(name="write", arguments={"path": float("nan")}),
        run_id=run_id,
        idempotency_key="k1",
        checkpoint_revision=3,
        approval_request_id=request.request_id,
    )

    assert result.error_code == "invalid_arguments"
    assert backend.calls == []
    state = await gate.get(request.request_id)
    assert state is not None and state[0].consumed_at is None


@pytest.mark.asyncio
async def test_handler_exception_is_caught_and_approval_stays_consumed() -> None:
    backend = ScriptedExecutor(outcomes=[RuntimeError("disk exploded")])
    executor, gate = make_executor(backend)
    run_id = uuid4()
    call = ToolCall(name="write", arguments={"path": "report.md"})
    request = await approved_request(gate, run_id=run_id)

    result = await executor.execute(
        call,
        run_id=run_id,
        idempotency_key="k1",
        checkpoint_revision=3,
        approval_request_id=request.request_id,
    )

    assert not result.ok
    assert result.error_code == "tool_execution_failed"
    assert not result.retryable
    assert "disk exploded" in result.content
    assert backend.calls == [call]
    # The approval was consumed before the crash; it must not be replayable.
    state = await gate.get(request.request_id)
    assert state is not None and state[0].consumed_at is not None
    retry = await executor.execute(
        call,
        run_id=run_id,
        idempotency_key="k2",
        checkpoint_revision=3,
        approval_request_id=request.request_id,
    )
    assert retry.error_code == "approval_already_consumed"
    assert backend.calls == [call]


@pytest.mark.asyncio
async def test_retryable_failure_result_passes_through() -> None:
    backend = ScriptedExecutor(
        outcomes=[
            ToolResult(ok=False, content="busy", error_code="tool_busy", retryable=True),
            ToolResult(ok=True, content="done"),
        ]
    )
    executor, gate = make_executor(backend)
    run_id = uuid4()
    call = ToolCall(name="write", arguments={"path": "report.md"})
    first_request = await approved_request(gate, run_id=run_id)

    failed = await executor.execute(
        call,
        run_id=run_id,
        idempotency_key="k1",
        checkpoint_revision=3,
        approval_request_id=first_request.request_id,
    )

    assert not failed.ok
    assert failed.error_code == "tool_busy"
    assert failed.retryable
    # A retry needs a fresh one-shot approval; the consumed one is not replayable.
    second_request = await approved_request(gate, run_id=run_id)
    retried = await executor.execute(
        call,
        run_id=run_id,
        idempotency_key="k1",
        checkpoint_revision=3,
        approval_request_id=second_request.request_id,
    )
    assert retried.ok
    assert backend.calls == [call, call]


@pytest.mark.asyncio
async def test_consume_persistence_failure_rolls_back_and_is_retryable() -> None:
    backend = ScriptedExecutor()
    executor, gate = make_executor(backend)
    run_id = uuid4()
    call = ToolCall(name="write", arguments={"path": "report.md"})
    request = await approved_request(gate, run_id=run_id)

    async def failing_callback(request: ApprovalRequest) -> None:
        raise OSError("event store unavailable")

    failed = await executor.execute(
        call,
        run_id=run_id,
        idempotency_key="k1",
        checkpoint_revision=3,
        approval_request_id=request.request_id,
        on_approval_consumed=failing_callback,
    )

    assert not failed.ok
    assert failed.error_code == "approval_consume_failed"
    assert failed.retryable
    assert backend.calls == []
    # The one-shot transition was rolled back: the approval is still usable.
    state = await gate.get(request.request_id)
    assert state is not None and state[0].consumed_at is None

    retried = await executor.execute(
        call,
        run_id=run_id,
        idempotency_key="k1",
        checkpoint_revision=3,
        approval_request_id=request.request_id,
    )
    assert retried.ok
    assert backend.calls == [call]


@pytest.mark.asyncio
async def test_concurrent_execute_consumes_approval_exactly_once() -> None:
    backend = ScriptedExecutor()
    executor, gate = make_executor(backend)
    run_id = uuid4()
    call = ToolCall(name="write", arguments={"path": "report.md"})
    request = await approved_request(gate, run_id=run_id)

    results = await asyncio.gather(
        *(
            executor.execute(
                call,
                run_id=run_id,
                idempotency_key=f"k{index}",
                checkpoint_revision=3,
                approval_request_id=request.request_id,
            )
            for index in range(8)
        )
    )

    succeeded = [result for result in results if result.ok]
    rejected = [result for result in results if not result.ok]
    assert len(succeeded) == 1
    assert {result.error_code for result in rejected} == {"approval_already_consumed"}
    assert backend.calls == [call]
