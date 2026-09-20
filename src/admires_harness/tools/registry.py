"""Schema-validating tool registry with idempotent side-effect receipts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from admires_harness.contracts import (
    JsonValue,
    RiskLevel,
    SideEffect,
    ToolCall,
    ToolResult,
)
from admires_harness.safety import SandboxViolation, ToolSafetySpec, WorkspaceSandbox

ToolHandler = Callable[[BaseModel], ToolResult | Awaitable[ToolResult]]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    input_model: type[BaseModel]
    safety: ToolSafetySpec
    handler: ToolHandler

    def schema(self) -> Mapping[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }


class ToolRegistry:
    """Native tool adapter implementing the runtime ToolExecutor port."""

    def __init__(self, definitions: Sequence[ToolDefinition] = ()) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        self._receipts: dict[tuple[UUID, str], tuple[str, ToolResult]] = {}
        for definition in definitions:
            self.register(definition)

    @property
    def safety_specs(self) -> tuple[ToolSafetySpec, ...]:
        return tuple(definition.safety for definition in self._definitions.values())

    def register(self, definition: ToolDefinition) -> None:
        if not definition.name or definition.name != definition.safety.name:
            raise ValueError("tool definition and safety names must match")
        if definition.name in self._definitions:
            raise ValueError(f"tool already registered: {definition.name}")
        self._definitions[definition.name] = definition

    def schemas(self) -> Sequence[Mapping[str, Any]]:
        return tuple(definition.schema() for definition in self._definitions.values())

    async def execute(self, call: ToolCall, *, run_id: UUID, idempotency_key: str) -> ToolResult:
        definition = self._definitions.get(call.name)
        if definition is None:
            return ToolResult(ok=False, content="tool is not registered", error_code="unknown_tool")
        try:
            arguments = definition.input_model.model_validate(call.arguments)
        except ValidationError as exc:
            return ToolResult(
                ok=False,
                content=exc.errors(include_url=False, include_input=False)[0]["msg"],
                error_code="invalid_arguments",
            )
        digest = hashlib.sha256(
            json.dumps(call.arguments, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        key = (run_id, idempotency_key)
        previous = self._receipts.get(key)
        if previous is not None:
            if previous[0] != digest:
                return ToolResult(
                    ok=False,
                    content="idempotency key was already used with different arguments",
                    error_code="idempotency_conflict",
                )
            return previous[1]
        try:
            raw = definition.handler(arguments)
            result = await raw if isinstance(raw, Awaitable) else raw
        except SandboxViolation as exc:
            return ToolResult(ok=False, content=str(exc), error_code=exc.code)
        except Exception as exc:
            return ToolResult(ok=False, content=str(exc), error_code="tool_execution_failed")
        if definition.safety.side_effect not in {SideEffect.NONE, SideEffect.READ}:
            receipt = result.receipt or f"sha256:{digest}"
            result = result.model_copy(update={"receipt": receipt})
            self._receipts[key] = (digest, result)
        return result


class _PathInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str = Field(default=".", min_length=1, max_length=1_000)


class _SearchInput(_PathInput):
    query: str = Field(min_length=1, max_length=1_000)
    limit: int = Field(default=20, ge=1, le=100)


class _WriteInput(_PathInput):
    content: str = Field(max_length=1_000_000)
    overwrite: bool = False


def workspace_tools(workspace: WorkspaceSandbox) -> ToolRegistry:
    """Create bounded filesystem tools rooted in one run workspace."""

    def read(arguments: BaseModel) -> ToolResult:
        parsed = _PathInput.model_validate(arguments)
        return ToolResult(ok=True, content=workspace.read_text(parsed.path))

    def list_files(arguments: BaseModel) -> ToolResult:
        parsed = _PathInput.model_validate(arguments)
        files = workspace.list_files(parsed.path)
        return ToolResult(ok=True, content="\n".join(files), data={"count": len(files)})

    def search(arguments: BaseModel) -> ToolResult:
        parsed = _SearchInput.model_validate(arguments)
        matches: list[JsonValue] = []
        for path in workspace.list_files(parsed.path):
            try:
                content = workspace.read_text(path)
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(content.splitlines(), start=1):
                if parsed.query.casefold() in line.casefold():
                    matches.append({"path": path, "line": line_number, "text": line[:500]})
                    if len(matches) >= parsed.limit:
                        break
            if len(matches) >= parsed.limit:
                break
        return ToolResult(
            ok=True, content=json.dumps(matches, ensure_ascii=False), data={"matches": matches}
        )

    def write(arguments: BaseModel) -> ToolResult:
        parsed = _WriteInput.model_validate(arguments)
        path = workspace.write_text(parsed.path, parsed.content, overwrite=parsed.overwrite)
        return ToolResult(ok=True, content=f"wrote {path.name}", artifact_paths=(parsed.path,))

    definitions = (
        ToolDefinition(
            "workspace.read",
            "Read one UTF-8 file inside the isolated workspace.",
            _PathInput,
            ToolSafetySpec(name="workspace.read", side_effect=SideEffect.READ),
            read,
        ),
        ToolDefinition(
            "workspace.list",
            "List bounded regular files inside the isolated workspace.",
            _PathInput,
            ToolSafetySpec(name="workspace.list", side_effect=SideEffect.READ),
            list_files,
        ),
        ToolDefinition(
            "workspace.search",
            "Search bounded UTF-8 workspace files.",
            _SearchInput,
            ToolSafetySpec(name="workspace.search", side_effect=SideEffect.READ),
            search,
        ),
        ToolDefinition(
            "workspace.write",
            "Write one file inside the isolated workspace.",
            _WriteInput,
            ToolSafetySpec(
                name="workspace.write", side_effect=SideEffect.WRITE, risk=RiskLevel.MEDIUM
            ),
            write,
        ),
    )
    return ToolRegistry(definitions)
