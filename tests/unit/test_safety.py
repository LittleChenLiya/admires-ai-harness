from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from admires_harness.contracts import (
    ApprovalDecision,
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
