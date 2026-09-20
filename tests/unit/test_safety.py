from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
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
    ApprovalError,
    ApprovalGate,
    PolicyViolation,
    SafetyPolicy,
    SecuredToolExecutor,
    ToolSafetySpec,
    approval_arguments_digest,
)


def test_arguments_digest_is_canonical_and_rejects_nan() -> None:
    assert approval_arguments_digest({"b": 2, "a": 1}) == approval_arguments_digest(
        {"a": 1, "b": 2}
    )
    with pytest.raises(ValueError):
        approval_arguments_digest({"bad": float("nan")})


@pytest.mark.asyncio
async def test_approval_is_bound_and_single_use() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    gate = ApprovalGate(clock=lambda: now)
    run_id = uuid4()
    request = await gate.create(
        run_id=run_id,
        checkpoint_revision=4,
        tool_name="workspace.write",
        arguments={"path": "notes.txt", "content": "safe"},
        reason="write a report",
        risk=RiskLevel.MEDIUM,
    )
    decision = ApprovalDecision(
        request_id=request.request_id, approved=True, decided_by="reviewer", decided_at=now
    )
    await gate.resolve(decision)

    consumed = await gate.consume(
        request.request_id,
        run_id=run_id,
        checkpoint_revision=4,
        tool_name="workspace.write",
        arguments={"content": "safe", "path": "notes.txt"},
    )

    assert consumed == decision
    state = await gate.get(request.request_id)
    assert state is not None and state[0].consumed_at == now
    with pytest.raises(ApprovalError) as error:
        await gate.consume(
            request.request_id,
            run_id=run_id,
            checkpoint_revision=4,
            tool_name="workspace.write",
            arguments={"path": "notes.txt", "content": "safe"},
        )
    assert error.value.code == "already_consumed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"run_id": uuid4()},
        {"checkpoint_revision": 8},
        {"tool_name": "process.run"},
        {"arguments": {"path": "../../secret"}},
    ],
)
async def test_approval_rejects_changed_binding(changes: dict[str, object]) -> None:
    run_id = uuid4()
    gate = ApprovalGate()
    request = await gate.create(
        run_id=run_id,
        checkpoint_revision=7,
        tool_name="workspace.write",
        arguments={"path": "report.md"},
        reason="write report",
        risk=RiskLevel.HIGH,
    )
    await gate.resolve(
        ApprovalDecision(request_id=request.request_id, approved=True, decided_by="human")
    )
    binding: dict[str, object] = {
        "run_id": run_id,
        "checkpoint_revision": 7,
        "tool_name": "workspace.write",
        "arguments": {"path": "report.md"},
    }
    binding.update(changes)

    with pytest.raises(ApprovalError) as error:
        await gate.consume(request.request_id, **binding)  # type: ignore[arg-type]
    assert error.value.code == "binding_mismatch"


@pytest.mark.asyncio
async def test_approval_rejects_expiry_denial_and_double_decision() -> None:
    current = datetime(2026, 1, 1, tzinfo=UTC)

    def clock() -> datetime:
        return current

    gate = ApprovalGate(ttl=timedelta(seconds=1), clock=clock)
    request = await gate.create(
        run_id=uuid4(),
        checkpoint_revision=0,
        tool_name="network.get",
        arguments={},
        reason="fetch evidence",
        risk=RiskLevel.HIGH,
    )
    current += timedelta(seconds=1)
    with pytest.raises(ApprovalError) as expired:
        await gate.resolve(
            ApprovalDecision(request_id=request.request_id, approved=True, decided_by="human")
        )
    assert expired.value.code == "expired"

    gate = ApprovalGate()
    denied = await gate.create(
        run_id=uuid4(),
        checkpoint_revision=0,
        tool_name="x",
        arguments={},
        reason="because",
        risk=RiskLevel.LOW,
    )
    await gate.resolve(
        ApprovalDecision(request_id=denied.request_id, approved=False, decided_by="human")
    )
    with pytest.raises(ApprovalError) as denial:
        await gate.consume(
            denied.request_id,
            run_id=denied.run_id,
            checkpoint_revision=0,
            tool_name="x",
            arguments={},
        )
    assert denial.value.code == "denied"
    with pytest.raises(ApprovalError) as duplicate:
        await gate.resolve(
            ApprovalDecision(request_id=denied.request_id, approved=True, decided_by="human")
        )
    assert duplicate.value.code == "already_decided"


@pytest.mark.asyncio
async def test_resolved_approval_still_expires_before_consumption() -> None:
    current = datetime(2026, 1, 1, tzinfo=UTC)

    def clock() -> datetime:
        return current

    gate = ApprovalGate(ttl=timedelta(seconds=1), clock=clock)
    request = await gate.create(
        run_id=uuid4(),
        checkpoint_revision=2,
        tool_name="workspace.write",
        arguments={"path": "report.md"},
        reason="write the report",
        risk=RiskLevel.MEDIUM,
    )
    await gate.resolve(
        ApprovalDecision(request_id=request.request_id, approved=True, decided_by="human")
    )
    current += timedelta(seconds=1)

    with pytest.raises(ApprovalError) as expired:
        await gate.consume(
            request.request_id,
            run_id=request.run_id,
            checkpoint_revision=request.checkpoint_revision,
            tool_name=request.tool_name,
            arguments={"path": "report.md"},
        )

    assert expired.value.code == "expired"


def test_policy_denies_unknown_process_network_and_critical_by_default() -> None:
    policy = SafetyPolicy(
        [
            ToolSafetySpec(name="read", side_effect=SideEffect.READ),
            ToolSafetySpec(name="write", side_effect=SideEffect.WRITE),
            ToolSafetySpec(name="shell", side_effect=SideEffect.PROCESS),
            ToolSafetySpec(name="http", side_effect=SideEffect.NETWORK),
            ToolSafetySpec(name="danger", side_effect=SideEffect.NONE, risk=RiskLevel.CRITICAL),
        ]
    )

    assert policy.authorize("read").allowed
    with pytest.raises(PolicyViolation, match="valid one-shot"):
        policy.authorize("write")
    assert policy.authorize("write", approval_consumed=True).allowed
    for tool in ("shell", "http", "danger", "unregistered"):
        with pytest.raises(PolicyViolation):
            policy.authorize(tool, approval_consumed=True)


def test_policy_requires_explicit_duplicate_free_registration() -> None:
    policy = SafetyPolicy(denied_tools=["never"])
    spec = ToolSafetySpec(name="pure", side_effect=SideEffect.NONE)
    policy.register(spec)
    with pytest.raises(ValueError, match="already registered"):
        policy.register(spec)
    assert not policy.evaluate("never").allowed


class RecordingExecutor:
    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    def schemas(self) -> list[dict[str, object]]:
        return [{"type": "function", "function": {"name": "write"}}]

    async def execute(
        self, ToolCall: ToolCall, *, run_id: object, idempotency_key: str
    ) -> ToolResult:
        del run_id, idempotency_key
        self.calls.append(ToolCall)
        return ToolResult(ok=True, content="done", receipt="receipt-1")


@pytest.mark.asyncio
async def test_secured_executor_denies_unknown_and_requires_valid_approval() -> None:
    backend = RecordingExecutor()
    gate = ApprovalGate()
    policy = SafetyPolicy([ToolSafetySpec(name="write", side_effect=SideEffect.WRITE)])
    executor = SecuredToolExecutor(backend, policy=policy, approvals=gate)
    run_id = uuid4()
    call = ToolCall(name="write", arguments={"path": "report.md"})

    unknown = await executor.execute(
        ToolCall(name="unknown"), run_id=run_id, idempotency_key="unknown"
    )
    required = await executor.execute(call, run_id=run_id, idempotency_key="first")
    assert unknown.error_code == "policy_denied"
    assert required.error_code == "approval_required"
    assert backend.calls == []

    request = await gate.create(
        run_id=run_id,
        checkpoint_revision=3,
        tool_name=call.name,
        arguments=call.arguments,
        reason="write result",
        risk=RiskLevel.MEDIUM,
    )
    await gate.resolve(
        ApprovalDecision(request_id=request.request_id, approved=True, decided_by="reviewer")
    )
    result = await executor.execute(
        call,
        run_id=run_id,
        idempotency_key="approved",
        checkpoint_revision=3,
        approval_request_id=request.request_id,
    )
    replay = await executor.execute(
        call,
        run_id=run_id,
        idempotency_key="replay",
        checkpoint_revision=3,
        approval_request_id=request.request_id,
    )

    assert result.ok and result.receipt == "receipt-1"
    assert replay.error_code == "approval_already_consumed"
    assert backend.calls == [call]


class StubExecutor:
    """Deterministic executor double with scripted results and event recording."""

    def __init__(
        self,
        results: list[ToolResult] | None = None,
        error: Exception | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.calls: list[ToolCall] = []
        self._results = list(results or [])
        self._error = error
        self._events = events

    def schemas(self) -> list[dict[str, object]]:
        return []

    async def execute(self, call: ToolCall, *, run_id: UUID, idempotency_key: str) -> ToolResult:
        del run_id, idempotency_key
        self.calls.append(call)
        if self._events is not None:
            self._events.append("handler")
        if self._error is not None:
            raise self._error
        if self._results:
            return self._results.pop(0)
        return ToolResult(ok=True, content="done", receipt="receipt-1")


def _write_call(**arguments: object) -> ToolCall:
    return ToolCall(name="write", arguments=arguments or {"path": "report.md"})  # type: ignore[arg-type]


def _secured(
    backend: StubExecutor, gate: ApprovalGate | None = None
) -> tuple[SecuredToolExecutor, ApprovalGate]:
    gate = gate or ApprovalGate()
    policy = SafetyPolicy([ToolSafetySpec(name="write", side_effect=SideEffect.WRITE)])
    return SecuredToolExecutor(backend, policy=policy, approvals=gate), gate


async def _approved_request(
    gate: ApprovalGate,
    run_id: UUID,
    call: ToolCall,
    *,
    checkpoint_revision: int = 3,
    approved: bool = True,
) -> ApprovalRequest:
    request = await gate.create(
        run_id=run_id,
        checkpoint_revision=checkpoint_revision,
        tool_name=call.name,
        arguments=call.arguments,
        reason="write the report",
        risk=RiskLevel.MEDIUM,
    )
    await gate.resolve(
        ApprovalDecision(request_id=request.request_id, approved=approved, decided_by="reviewer")
    )
    return request


@pytest.mark.asyncio
async def test_secured_executor_consumes_approval_once_before_handler() -> None:
    events: list[str] = []
    backend = StubExecutor(events=events)
    executor, gate = _secured(backend)
    run_id = uuid4()
    call = _write_call()
    request = await _approved_request(gate, run_id, call)

    async def on_consumed(consumed: ApprovalRequest) -> None:
        events.append("consumed")

    result = await executor.execute(
        call,
        run_id=run_id,
        idempotency_key="approved",
        checkpoint_revision=3,
        approval_request_id=request.request_id,
        on_approval_consumed=on_consumed,
    )
    replay = await executor.execute(
        call,
        run_id=run_id,
        idempotency_key="replay",
        checkpoint_revision=3,
        approval_request_id=request.request_id,
    )

    assert result.ok and result.receipt == "receipt-1"
    assert replay.error_code == "approval_already_consumed"
    assert events == ["consumed", "handler"]
    assert backend.calls == [call]


@pytest.mark.asyncio
async def test_secured_executor_without_approval_never_calls_handler() -> None:
    backend = StubExecutor()
    executor, _ = _secured(backend)

    result = await executor.execute(_write_call(), run_id=uuid4(), idempotency_key="no-approval")

    assert result.error_code == "approval_required"
    assert backend.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override, expected_code",
    [
        ({"run_id": "other-run"}, "approval_binding_mismatch"),
        ({"checkpoint_revision": 4}, "approval_binding_mismatch"),
        ({"arguments": {"path": "other.md"}}, "approval_binding_mismatch"),
        ({"request_id": "other-request"}, "approval_not_found"),
    ],
)
async def test_secured_executor_rejects_changed_approval_binding(
    override: dict[str, object], expected_code: str
) -> None:
    backend = StubExecutor()
    executor, gate = _secured(backend)
    run_id = uuid4()
    call = _write_call()
    request = await _approved_request(gate, run_id, call)

    binding: dict[str, object] = {
        "run_id": run_id,
        "checkpoint_revision": 3,
        "approval_request_id": request.request_id,
    }
    binding.update(
        {
            "run_id": uuid4() if override.get("run_id") == "other-run" else binding["run_id"],
            "checkpoint_revision": override.get("checkpoint_revision", 3),
            "approval_request_id": (
                uuid4() if override.get("request_id") == "other-request" else request.request_id
            ),
        }
    )
    changed_call = _write_call(**override.get("arguments", call.arguments))  # type: ignore[arg-type]

    rejected = await executor.execute(
        changed_call,
        idempotency_key="changed",
        **binding,  # type: ignore[arg-type]
    )

    assert rejected.error_code == expected_code
    assert backend.calls == []
    # The failed attempt must not consume the approval: the original binding still works.
    if expected_code == "approval_binding_mismatch":
        recovered = await executor.execute(
            call,
            run_id=run_id,
            idempotency_key="recovered",
            checkpoint_revision=3,
            approval_request_id=request.request_id,
        )
        assert recovered.ok
        assert backend.calls == [call]


@pytest.mark.asyncio
async def test_secured_executor_maps_denied_and_expired_approvals() -> None:
    backend = StubExecutor()
    executor, gate = _secured(backend)
    run_id = uuid4()
    call = _write_call()

    denied = await _approved_request(gate, run_id, call, approved=False)
    denied_result = await executor.execute(
        call,
        run_id=run_id,
        idempotency_key="denied",
        checkpoint_revision=3,
        approval_request_id=denied.request_id,
    )
    assert denied_result.error_code == "approval_denied"

    current = datetime(2026, 1, 1, tzinfo=UTC)
    expiring_gate = ApprovalGate(ttl=timedelta(seconds=1), clock=lambda: current)
    expiring_executor, _ = _secured(backend, expiring_gate)
    stale = await _approved_request(expiring_gate, run_id, call)
    current += timedelta(seconds=2)
    expired_result = await expiring_executor.execute(
        call,
        run_id=run_id,
        idempotency_key="expired",
        checkpoint_revision=3,
        approval_request_id=stale.request_id,
    )
    assert expired_result.error_code == "approval_expired"
    assert backend.calls == []


@pytest.mark.asyncio
async def test_secured_executor_handler_exception_keeps_approval_consumed() -> None:
    backend = StubExecutor(error=RuntimeError("disk full"))
    executor, gate = _secured(backend)
    run_id = uuid4()
    call = _write_call()
    request = await _approved_request(gate, run_id, call)

    failed = await executor.execute(
        call,
        run_id=run_id,
        idempotency_key="failing",
        checkpoint_revision=3,
        approval_request_id=request.request_id,
    )
    replay = await executor.execute(
        call,
        run_id=run_id,
        idempotency_key="replay",
        checkpoint_revision=3,
        approval_request_id=request.request_id,
    )

    assert failed.error_code == "tool_execution_failed"
    assert "disk full" in failed.content
    # One-shot semantics: the crash after consumption must not replay the approval.
    assert replay.error_code == "approval_already_consumed"
    assert backend.calls == [call]


@pytest.mark.asyncio
async def test_secured_executor_passes_through_retryable_failure() -> None:
    backend = StubExecutor(
        results=[
            ToolResult(ok=False, content="busy", error_code="rate_limited", retryable=True),
            ToolResult(ok=True, content="done", receipt="receipt-2"),
        ]
    )
    executor, gate = _secured(backend)
    run_id = uuid4()
    call = _write_call()

    first_request = await _approved_request(gate, run_id, call)
    first = await executor.execute(
        call,
        run_id=run_id,
        idempotency_key="attempt-1",
        checkpoint_revision=3,
        approval_request_id=first_request.request_id,
    )
    second_request = await _approved_request(gate, run_id, call)
    second = await executor.execute(
        call,
        run_id=run_id,
        idempotency_key="attempt-2",
        checkpoint_revision=3,
        approval_request_id=second_request.request_id,
    )

    assert first.error_code == "rate_limited" and first.retryable
    assert second.ok and second.receipt == "receipt-2"
    assert backend.calls == [call, call]


@pytest.mark.asyncio
async def test_secured_executor_rolls_back_consumption_when_callback_fails() -> None:
    backend = StubExecutor()
    executor, gate = _secured(backend)
    run_id = uuid4()
    call = _write_call()
    request = await _approved_request(gate, run_id, call)

    async def failing_callback(consumed: ApprovalRequest) -> None:
        raise RuntimeError("event store unavailable")

    failed = await executor.execute(
        call,
        run_id=run_id,
        idempotency_key="callback-fails",
        checkpoint_revision=3,
        approval_request_id=request.request_id,
        on_approval_consumed=failing_callback,
    )

    assert failed.error_code == "approval_callback_failed"
    assert backend.calls == []
    # The rollback means the same approval can still authorize the retried execution.
    events: list[str] = []

    async def recording_callback(consumed: ApprovalRequest) -> None:
        events.append("consumed")

    recovered = await executor.execute(
        call,
        run_id=run_id,
        idempotency_key="callback-recovered",
        checkpoint_revision=3,
        approval_request_id=request.request_id,
        on_approval_consumed=recording_callback,
    )
    assert recovered.ok
    assert events == ["consumed"]
    assert backend.calls == [call]


@pytest.mark.asyncio
async def test_secured_executor_concurrent_execute_consumes_exactly_once() -> None:
    backend = StubExecutor()
    executor, gate = _secured(backend)
    run_id = uuid4()
    call = _write_call()
    request = await _approved_request(gate, run_id, call)

    first, second = await asyncio.gather(
        executor.execute(
            call,
            run_id=run_id,
            idempotency_key="concurrent-1",
            checkpoint_revision=3,
            approval_request_id=request.request_id,
        ),
        executor.execute(
            call,
            run_id=run_id,
            idempotency_key="concurrent-2",
            checkpoint_revision=3,
            approval_request_id=request.request_id,
        ),
    )

    outcomes = sorted(
        ("ok" if result.ok else result.error_code or "") for result in (first, second)
    )
    assert outcomes == ["approval_already_consumed", "ok"]
    assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_secured_executor_fails_closed_on_unexpected_gate_error() -> None:
    backend = StubExecutor()
    executor, gate = _secured(backend)
    run_id = uuid4()
    call = _write_call(path="report.md")
    request = await _approved_request(gate, run_id, call)
    # NaN cannot be hashed into the canonical arguments digest, so the gate raises an
    # unexpected ValueError. The executor must fail closed instead of propagating it.
    poisoned = ToolCall(name="write", arguments={"path": float("nan")})

    result = await executor.execute(
        poisoned,
        run_id=run_id,
        idempotency_key="poisoned",
        checkpoint_revision=3,
        approval_request_id=request.request_id,
    )

    assert result.error_code == "approval_error"
    assert backend.calls == []
