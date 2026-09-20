"""Stable data contracts shared by the runtime and every adapter.

The event envelope is deliberately boring and versioned. CLI output, SSE, persistence,
replay, tracing, and evaluations all consume the same representation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    BUDGET_EXCEEDED = "budget_exceeded"

    @property
    def terminal(self) -> bool:
        return self in {
            self.SUCCEEDED,
            self.FAILED,
            self.CANCELLED,
            self.TIMED_OUT,
            self.BUDGET_EXCEEDED,
        }


class EventType(StrEnum):
    RUN_CREATED = "run.created"
    RUN_STARTED = "run.started"
    RUN_STATUS = "run.status"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    NODE_STARTED = "node.started"
    NODE_COMPLETED = "node.completed"
    NODE_FAILED = "node.failed"
    MODEL_STARTED = "model.started"
    MODEL_TOKEN = "model.token"
    MODEL_COMPLETED = "model.completed"
    TOOL_REQUESTED = "tool.requested"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_RESOLVED = "approval.resolved"
    APPROVAL_CONSUMED = "approval.consumed"
    CHECKPOINT_SAVED = "checkpoint.saved"
    MEMORY_WRITTEN = "memory.written"
    MEMORY_RETRIEVED = "memory.retrieved"
    AGENT_DELEGATED = "agent.delegated"
    AGENT_JOINED = "agent.joined"
    LOG = "log"


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class SideEffect(StrEnum):
    NONE = "none"
    READ = "read"
    WRITE = "write"
    PROCESS = "process"
    NETWORK = "network"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Message(StrictModel):
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class ToolCall(StrictModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class ModelUsage(StrictModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0, allow_inf_nan=False)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelResponse(StrictModel):
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: ModelUsage = Field(default_factory=ModelUsage)
    finish_reason: str = "stop"
    model: str = "unknown"


class RunBudget(StrictModel):
    max_steps: int = Field(default=30, ge=1)
    max_tokens: int = Field(default=100_000, ge=1)
    max_cost_usd: float = Field(default=10.0, ge=0)
    timeout_seconds: float = Field(default=300.0, gt=0)
    max_repeated_actions: int = Field(default=3, ge=1)


class RunEvent(StrictModel):
    schema_version: Literal["1"] = "1"
    event_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    sequence: int = Field(ge=0)
    type: EventType
    timestamp: datetime = Field(default_factory=utc_now)
    parent_run_id: UUID | None = None
    node_id: str | None = None
    trace_id: UUID
    span_id: UUID = Field(default_factory=uuid4)
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class ToolResult(StrictModel):
    ok: bool
    content: str = ""
    data: dict[str, JsonValue] = Field(default_factory=dict)
    error_code: str | None = None
    retryable: bool = False
    artifact_paths: tuple[str, ...] = ()
    receipt: str | None = None

    @model_validator(mode="after")
    def error_results_have_codes(self) -> ToolResult:
        if not self.ok and not self.error_code:
            raise ValueError("failed tool results require error_code")
        return self


class ApprovalRequest(StrictModel):
    request_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    checkpoint_revision: int = Field(ge=0)
    tool_name: str
    arguments_digest: str
    reason: str
    risk: RiskLevel
    expires_at: datetime
    consumed_at: datetime | None = None


class ApprovalDecision(StrictModel):
    request_id: UUID
    approved: bool
    decided_by: str
    decided_at: datetime = Field(default_factory=utc_now)
    comment: str | None = None


class Checkpoint(StrictModel):
    run_id: UUID
    revision: int = Field(ge=0)
    graph_name: str
    graph_version: str
    next_node: str | None
    status: RunStatus
    state: dict[str, JsonValue]
    elapsed_active_seconds: float = Field(default=0.0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)


class TraceSummary(StrictModel):
    run_id: UUID
    status: RunStatus
    started_at: datetime
    ended_at: datetime | None = None
    steps: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)
