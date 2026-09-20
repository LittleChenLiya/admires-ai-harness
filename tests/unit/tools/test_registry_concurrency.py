"""Deterministic concurrency tests for the tool registry idempotency receipts."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from pydantic import BaseModel, ConfigDict

from admires_harness.contracts import JsonValue, RiskLevel, SideEffect, ToolCall, ToolResult
from admires_harness.safety import ToolSafetySpec
from admires_harness.tools import ToolDefinition, ToolRegistry
from admires_harness.tools.registry import ToolHandler


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: int = 0
    label: str = ""


def _definition(
    handler: ToolHandler,
    *,
    side_effect: SideEffect = SideEffect.WRITE,
    name: str = "test.tool",
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="test tool",
        input_model=_Args,
        safety=ToolSafetySpec(name=name, side_effect=side_effect, risk=RiskLevel.MEDIUM),
        handler=handler,
    )


def _call(arguments: dict[str, JsonValue], *, name: str = "test.tool") -> ToolCall:
    return ToolCall(name=name, arguments=arguments)


async def _pump_loop(cycles: int = 200) -> None:
    """Let every ready task run to its next await; deterministic, not time-based."""
    for _ in range(cycles):
        await asyncio.sleep(0)


class _GatedHandler:
    """Blocks inside the handler until released so callers overlap deterministically."""

    def __init__(self) -> None:
        self.calls = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, arguments: BaseModel) -> ToolResult:
        self.calls += 1
        invocation = self.calls
        self.entered.set()
        await self.release.wait()
        return ToolResult(ok=True, content=f"call-{invocation}")


async def _await_entered(handler: _GatedHandler) -> None:
    await asyncio.wait_for(handler.entered.wait(), timeout=5)
    await _pump_loop()


@pytest.mark.asyncio
async def test_concurrent_duplicates_share_one_handler_invocation() -> None:
    handler = _GatedHandler()
    registry = ToolRegistry([_definition(handler)])
    run_id = uuid4()
    tasks = [
        asyncio.create_task(
            registry.execute(_call({"value": 7}), run_id=run_id, idempotency_key="dup")
        )
        for _ in range(48)
    ]
    await _await_entered(handler)
    handler.release.set()
    results = await asyncio.gather(*tasks)

    assert handler.calls == 1
    first = results[0]
    assert first.ok and first.receipt
    assert all(result == first for result in results)


@pytest.mark.asyncio
async def test_concurrent_calls_dedupe_per_idempotency_key() -> None:
    handler = _GatedHandler()
    registry = ToolRegistry([_definition(handler)])
    run_id = uuid4()
    keys = ("key-a", "key-b", "key-c")
    keyed_tasks = [
        (
            key,
            asyncio.create_task(
                registry.execute(_call({"value": 1}), run_id=run_id, idempotency_key=key)
            ),
        )
        for key in keys
        for _ in range(12)
    ]
    await _await_entered(handler)
    handler.release.set()
    results = [await task for _, task in keyed_tasks]

    assert handler.calls == len(keys)
    for key in keys:
        grouped = [
            result for (want, _), result in zip(keyed_tasks, results, strict=True) if want == key
        ]
        assert len(grouped) == 12
        assert all(result == grouped[0] for result in grouped)
    assert len({result.content for result in results}) == len(keys)


@pytest.mark.asyncio
async def test_concurrent_conflicting_arguments_return_conflict() -> None:
    handler = _GatedHandler()
    registry = ToolRegistry([_definition(handler)])
    run_id = uuid4()
    winner = asyncio.create_task(
        registry.execute(_call({"value": 1}), run_id=run_id, idempotency_key="clash")
    )
    await _await_entered(handler)
    conflicts = await asyncio.wait_for(
        asyncio.gather(
            *(
                registry.execute(_call({"value": 2}), run_id=run_id, idempotency_key="clash")
                for _ in range(8)
            )
        ),
        timeout=5,
    )
    handler.release.set()
    result = await winner

    assert result.ok
    assert handler.calls == 1
    assert all(conflict.error_code == "idempotency_conflict" for conflict in conflicts)
    assert all(not conflict.ok for conflict in conflicts)


@pytest.mark.asyncio
async def test_completed_receipt_conflict_and_replay_are_stable() -> None:
    handler = _GatedHandler()
    handler.release.set()
    registry = ToolRegistry([_definition(handler)])
    run_id = uuid4()
    first = await registry.execute(_call({"value": 1}), run_id=run_id, idempotency_key="k")
    replay = await registry.execute(_call({"value": 1}), run_id=run_id, idempotency_key="k")
    conflict = await registry.execute(_call({"value": 2}), run_id=run_id, idempotency_key="k")

    assert handler.calls == 1
    assert replay == first
    assert conflict.error_code == "idempotency_conflict"


@pytest.mark.asyncio
async def test_failed_result_is_not_cached_and_retry_reinvokes() -> None:
    calls = 0

    def handler(arguments: BaseModel) -> ToolResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ToolResult(ok=False, content="transient", error_code="upstream_error")
        return ToolResult(ok=True, content="recovered")

    registry = ToolRegistry([_definition(handler)])
    run_id = uuid4()
    failed = await registry.execute(_call({"value": 1}), run_id=run_id, idempotency_key="k")
    retried = await registry.execute(_call({"value": 1}), run_id=run_id, idempotency_key="k")
    replay = await registry.execute(_call({"value": 1}), run_id=run_id, idempotency_key="k")

    assert failed.error_code == "upstream_error" and failed.receipt is None
    assert retried.ok and retried.receipt
    assert replay == retried
    assert calls == 2


@pytest.mark.asyncio
async def test_raising_handler_is_not_cached_and_retry_reinvokes() -> None:
    calls = 0
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(arguments: BaseModel) -> ToolResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await release.wait()
            raise RuntimeError("boom")
        return ToolResult(ok=True, content="recovered")

    registry = ToolRegistry([_definition(handler)])
    run_id = uuid4()
    tasks = [
        asyncio.create_task(
            registry.execute(_call({"value": 1}), run_id=run_id, idempotency_key="k")
        )
        for _ in range(16)
    ]
    await asyncio.wait_for(entered.wait(), timeout=5)
    await _pump_loop()
    release.set()
    failures = await asyncio.gather(*tasks)

    assert calls == 1
    assert all(failure.error_code == "tool_execution_failed" for failure in failures)
    assert all("boom" in failure.content for failure in failures)

    retried = await registry.execute(_call({"value": 1}), run_id=run_id, idempotency_key="k")
    replay = await registry.execute(_call({"value": 1}), run_id=run_id, idempotency_key="k")
    assert retried.ok and retried.receipt
    assert replay == retried
    assert calls == 2


@pytest.mark.asyncio
async def test_receipts_are_scoped_per_run_and_key() -> None:
    calls = 0

    def handler(arguments: BaseModel) -> ToolResult:
        nonlocal calls
        calls += 1
        return ToolResult(ok=True, content=f"call-{calls}")

    registry = ToolRegistry([_definition(handler)])
    run_a, run_b = uuid4(), uuid4()
    first = await registry.execute(_call({"value": 1}), run_id=run_a, idempotency_key="k")
    other_run = await registry.execute(_call({"value": 1}), run_id=run_b, idempotency_key="k")
    other_key = await registry.execute(_call({"value": 1}), run_id=run_a, idempotency_key="j")
    replay = await registry.execute(_call({"value": 1}), run_id=run_a, idempotency_key="k")

    assert calls == 3
    assert replay == first
    assert len({first.content, other_run.content, other_key.content}) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("side_effect", [SideEffect.NONE, SideEffect.READ])
async def test_pure_and_read_only_tools_are_never_cached(side_effect: SideEffect) -> None:
    calls = 0

    def handler(arguments: BaseModel) -> ToolResult:
        nonlocal calls
        calls += 1
        return ToolResult(ok=True, content=f"call-{calls}")

    registry = ToolRegistry([_definition(handler, side_effect=side_effect)])
    run_id = uuid4()
    results = [
        await registry.execute(_call({"value": 1}), run_id=run_id, idempotency_key="same")
        for _ in range(3)
    ]

    assert calls == 3
    assert len({result.content for result in results}) == 3
    assert all(result.receipt is None for result in results)


@pytest.mark.asyncio
async def test_argument_json_key_order_is_canonicalized() -> None:
    calls = 0

    def handler(arguments: BaseModel) -> ToolResult:
        nonlocal calls
        calls += 1
        return ToolResult(ok=True, content="done")

    registry = ToolRegistry([_definition(handler)])
    run_id = uuid4()
    first = await registry.execute(
        _call({"value": 1, "label": "x"}), run_id=run_id, idempotency_key="k"
    )
    replay = await registry.execute(
        _call({"label": "x", "value": 1}), run_id=run_id, idempotency_key="k"
    )

    assert calls == 1
    assert replay == first


@pytest.mark.asyncio
async def test_success_receipt_preserves_result_contract() -> None:
    def handler(arguments: BaseModel) -> ToolResult:
        return ToolResult(ok=True, content="done", data={"rows": 1}, artifact_paths=("out/a.txt",))

    registry = ToolRegistry([_definition(handler)])
    run_id = uuid4()
    first = await registry.execute(_call({"value": 1}), run_id=run_id, idempotency_key="k")
    replay = await registry.execute(_call({"value": 1}), run_id=run_id, idempotency_key="k")

    assert first.receipt is not None and first.receipt.startswith("sha256:")
    assert replay == first
    assert replay.artifact_paths == ("out/a.txt",)
    assert replay.data == {"rows": 1}


@pytest.mark.asyncio
async def test_handler_supplied_receipt_is_respected() -> None:
    def handler(arguments: BaseModel) -> ToolResult:
        return ToolResult(ok=True, content="done", receipt="tool-owned-receipt")

    registry = ToolRegistry([_definition(handler)])
    result = await registry.execute(_call({}), run_id=uuid4(), idempotency_key="k")

    assert result.receipt == "tool-owned-receipt"


def test_duplicate_registration_is_rejected() -> None:
    def handler(arguments: BaseModel) -> ToolResult:
        return ToolResult(ok=True)

    registry = ToolRegistry([_definition(handler)])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_definition(handler))


def test_definition_and_safety_names_must_match() -> None:
    def handler(arguments: BaseModel) -> ToolResult:
        return ToolResult(ok=True)

    definition = ToolDefinition(
        name="one.name",
        description="mismatched",
        input_model=_Args,
        safety=ToolSafetySpec(name="other.name", side_effect=SideEffect.WRITE),
        handler=handler,
    )
    with pytest.raises(ValueError, match="names must match"):
        ToolRegistry([definition])
