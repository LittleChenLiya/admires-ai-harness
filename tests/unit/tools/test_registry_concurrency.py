"""Deterministic concurrency tests for idempotent side-effect receipts."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict

from admires_harness.contracts import JsonValue, SideEffect, ToolCall, ToolResult
from admires_harness.safety import ToolSafetySpec
from admires_harness.tools import ToolDefinition, ToolRegistry

FAN_OUT = 40


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    value: str
    count: int = 0


def _definition(
    handler: Callable[[BaseModel], ToolResult | Awaitable[ToolResult]],
    *,
    side_effect: SideEffect = SideEffect.WRITE,
    name: str = "demo.tool",
) -> ToolDefinition:
    return ToolDefinition(
        name,
        "demo tool",
        _Args,
        ToolSafetySpec(name=name, side_effect=side_effect),
        handler,
    )


def _registry(
    handler: Callable[[BaseModel], ToolResult | Awaitable[ToolResult]],
    *,
    side_effect: SideEffect = SideEffect.WRITE,
) -> ToolRegistry:
    return ToolRegistry([_definition(handler, side_effect=side_effect)])


def _call(arguments: dict[str, JsonValue]) -> ToolCall:
    return ToolCall(name="demo.tool", arguments=arguments)


async def _gather(
    registry: ToolRegistry,
    calls: list[ToolCall],
    *,
    run_id: UUID,
    key: str = "key",
) -> list[ToolResult]:
    return await asyncio.gather(
        *(registry.execute(call, run_id=run_id, idempotency_key=key) for call in calls)
    )


async def test_concurrent_identical_calls_invoke_handler_once():
    entered = asyncio.Event()
    release = asyncio.Event()
    invocations = 0

    async def handler(arguments: BaseModel) -> ToolResult:
        nonlocal invocations
        invocations += 1
        entered.set()
        await release.wait()
        return ToolResult(ok=True, content="done")

    registry = _registry(handler)
    run_id = uuid4()
    calls = [_call({"value": "same", "count": 1}) for _ in range(FAN_OUT)]
    results_task = asyncio.create_task(_gather(registry, calls, run_id=run_id))
    await asyncio.wait_for(entered.wait(), timeout=5)
    release.set()
    results = await results_task

    assert invocations == 1
    assert all(result.ok for result in results)
    assert all(result is results[0] for result in results)
    assert results[0].receipt


async def test_concurrent_mixed_arguments_split_shared_result_and_conflict():
    invocations = 0

    async def handler(arguments: BaseModel) -> ToolResult:
        nonlocal invocations
        invocations += 1
        await asyncio.sleep(0)
        return ToolResult(ok=True, content="done")

    registry = _registry(handler)
    run_id = uuid4()
    identical = [_call({"value": "same"}) for _ in range(FAN_OUT // 2)]
    conflicting = [_call({"value": f"other-{index}"}) for index in range(FAN_OUT // 2)]
    mixed = [call for pair in zip(identical, conflicting, strict=True) for call in pair]
    results = await _gather(registry, mixed, run_id=run_id)

    shared = results[::2]
    conflicts = results[1::2]
    assert invocations == 1
    assert all(result.ok and result is shared[0] for result in shared)
    assert all(result.error_code == "idempotency_conflict" for result in conflicts)


async def test_argument_json_key_order_does_not_create_conflict():
    invocations = 0

    def handler(arguments: BaseModel) -> ToolResult:
        nonlocal invocations
        invocations += 1
        return ToolResult(ok=True, content="done")

    registry = _registry(handler)
    run_id = uuid4()
    forward = _call({"value": "same", "count": 2})
    reversed_keys = _call({"count": 2, "value": "same"})
    results = await _gather(registry, [forward, reversed_keys], run_id=run_id)

    assert invocations == 1
    assert results[0].ok and results[1] == results[0]


async def test_failed_result_is_not_cached_and_retry_succeeds():
    invocations = 0

    async def handler(arguments: BaseModel) -> ToolResult:
        nonlocal invocations
        invocations += 1
        await asyncio.sleep(0)
        if invocations == 1:
            return ToolResult(ok=False, content="boom", error_code="backend_down")
        return ToolResult(ok=True, content=f"attempt-{invocations}")

    registry = _registry(handler)
    run_id = uuid4()
    first = await registry.execute(_call({"value": "x"}), run_id=run_id, idempotency_key="k")
    retry = await registry.execute(_call({"value": "x"}), run_id=run_id, idempotency_key="k")
    replay = await registry.execute(_call({"value": "x"}), run_id=run_id, idempotency_key="k")

    assert first.error_code == "backend_down" and not first.receipt
    assert retry.ok and retry.content == "attempt-2"
    assert replay is retry
    assert invocations == 2


async def test_raising_handler_shares_failure_and_allows_retry():
    invocations = 0

    async def handler(arguments: BaseModel) -> ToolResult:
        nonlocal invocations
        invocations += 1
        await asyncio.sleep(0)
        if invocations == 1:
            raise RuntimeError("exploded")
        return ToolResult(ok=True, content="recovered")

    registry = _registry(handler)
    run_id = uuid4()
    calls = [_call({"value": "x"}) for _ in range(FAN_OUT)]
    failures = await _gather(registry, calls, run_id=run_id, key="k")
    retry = await registry.execute(_call({"value": "x"}), run_id=run_id, idempotency_key="k")

    assert invocations == 2
    assert all(result.error_code == "tool_execution_failed" for result in failures)
    assert all(result.retryable and not result.receipt for result in failures)
    assert retry.ok and retry.content == "recovered" and retry.receipt


async def test_concurrent_failures_are_shared_within_one_batch():
    invocations = 0

    async def handler(arguments: BaseModel) -> ToolResult:
        nonlocal invocations
        invocations += 1
        await asyncio.sleep(0)
        return ToolResult(ok=False, content="down", error_code="backend_down")

    registry = _registry(handler)
    run_id = uuid4()
    calls = [_call({"value": "x"}) for _ in range(FAN_OUT)]
    results = await _gather(registry, calls, run_id=run_id, key="k")

    assert invocations == 1
    assert all(result is results[0] for result in results)


async def test_receipts_are_scoped_per_run_and_per_key():
    invocations = 0

    def handler(arguments: BaseModel) -> ToolResult:
        nonlocal invocations
        invocations += 1
        return ToolResult(ok=True, content=f"call-{invocations}")

    registry = _registry(handler)
    run_a, run_b = uuid4(), uuid4()
    first = await registry.execute(_call({"value": "x"}), run_id=run_a, idempotency_key="k")
    other_run = await registry.execute(_call({"value": "x"}), run_id=run_b, idempotency_key="k")
    other_key = await registry.execute(_call({"value": "x"}), run_id=run_a, idempotency_key="j")

    assert invocations == 3
    assert len({first.receipt, other_run.receipt, other_key.receipt}) == 1
    assert first.content != other_run.content != other_key.content


async def test_read_only_tools_are_never_cached():
    invocations = 0

    def handler(arguments: BaseModel) -> ToolResult:
        nonlocal invocations
        invocations += 1
        return ToolResult(ok=True, content=f"read-{invocations}")

    for side_effect in (SideEffect.NONE, SideEffect.READ):
        invocations = 0
        registry = _registry(handler, side_effect=side_effect)
        run_id = uuid4()
        calls = [_call({"value": "x"}) for _ in range(FAN_OUT)]
        results = await _gather(registry, calls, run_id=run_id, key="k")

        assert invocations == FAN_OUT
        assert all(result.ok and not result.receipt for result in results)
        assert len({result.content for result in results}) == FAN_OUT


async def test_success_is_not_faked_for_failed_handler_receipt():
    def handler(arguments: BaseModel) -> ToolResult:
        return ToolResult(ok=False, content="no", error_code="backend_down")

    registry = _registry(handler)
    run_id = uuid4()
    for _ in range(3):
        result = await registry.execute(_call({"value": "x"}), run_id=run_id, idempotency_key="k")
        assert not result.ok and not result.receipt
    conflict_attempt = await registry.execute(
        _call({"value": "y"}), run_id=run_id, idempotency_key="k"
    )
    assert conflict_attempt.error_code == "backend_down"


async def test_duplicate_registration_is_rejected():
    definition = _definition(lambda arguments: ToolResult(ok=True))
    registry = ToolRegistry([definition])
    try:
        registry.register(definition)
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("duplicate registration must raise ValueError")
