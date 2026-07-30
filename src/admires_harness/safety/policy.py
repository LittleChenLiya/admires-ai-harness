"""Central authorization policy for tools and their side effects."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import Field

from admires_harness.contracts import RiskLevel, SideEffect, StrictModel


class ToolSafetySpec(StrictModel):
    """Security metadata registered alongside a tool implementation."""

    name: str = Field(min_length=1, max_length=128)
    side_effect: SideEffect
    risk: RiskLevel = RiskLevel.LOW
    description: str = Field(default="", max_length=2_000)
    requires_approval: bool | None = None


class PolicyDecision(StrictModel):
    allowed: bool
    requires_approval: bool = False
    reason: str
    risk: RiskLevel


class PolicyViolation(PermissionError):
    def __init__(self, tool_name: str, reason: str) -> None:
        super().__init__(f"tool '{tool_name}' denied: {reason}")
        self.tool_name = tool_name
        self.reason = reason


class SafetyPolicy:
    """Allow only registered capabilities and gate meaningful side effects.

    Defaults permit low-risk pure/read tools, require approval for writes, and deny
    process/network/critical tools. Deployments opt into each stronger capability
    explicitly; enabling it still leaves human approval on by default.
    """

    def __init__(
        self,
        specs: Iterable[ToolSafetySpec] = (),
        *,
        allow_write: bool = True,
        allow_process: bool = False,
        allow_network: bool = False,
        allow_critical: bool = False,
        denied_tools: Iterable[str] = (),
    ) -> None:
        self._specs: dict[str, ToolSafetySpec] = {}
        self._denied_tools = frozenset(denied_tools)
        self.allow_write = allow_write
        self.allow_process = allow_process
        self.allow_network = allow_network
        self.allow_critical = allow_critical
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSafetySpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"tool security metadata already registered: {spec.name}")
        self._specs[spec.name] = spec

    def evaluate(self, tool_name: str) -> PolicyDecision:
        if tool_name in self._denied_tools:
            return PolicyDecision(
                allowed=False, reason="tool is explicitly denied", risk=RiskLevel.CRITICAL
            )
        spec = self._specs.get(tool_name)
        if spec is None:
            return PolicyDecision(
                allowed=False,
                reason="tool has no registered security metadata",
                risk=RiskLevel.CRITICAL,
            )
        if spec.risk is RiskLevel.CRITICAL and not self.allow_critical:
            return PolicyDecision(
                allowed=False, reason="critical-risk tools are disabled", risk=spec.risk
            )
        capability_flags = {
            SideEffect.WRITE: self.allow_write,
            SideEffect.PROCESS: self.allow_process,
            SideEffect.NETWORK: self.allow_network,
        }
        if spec.side_effect in capability_flags and not capability_flags[spec.side_effect]:
            return PolicyDecision(
                allowed=False,
                reason=f"{spec.side_effect.value} capability is disabled",
                risk=spec.risk,
            )
        default_approval = spec.side_effect in {
            SideEffect.WRITE,
            SideEffect.PROCESS,
            SideEffect.NETWORK,
        } or spec.risk in {RiskLevel.HIGH, RiskLevel.CRITICAL}
        requires_approval = (
            spec.requires_approval if spec.requires_approval is not None else default_approval
        )
        return PolicyDecision(
            allowed=True,
            requires_approval=requires_approval,
            reason=("human approval required" if requires_approval else "allowed by policy"),
            risk=spec.risk,
        )

    def authorize(self, tool_name: str, *, approval_consumed: bool = False) -> PolicyDecision:
        decision = self.evaluate(tool_name)
        if not decision.allowed:
            raise PolicyViolation(tool_name, decision.reason)
        if decision.requires_approval and not approval_consumed:
            raise PolicyViolation(tool_name, "a valid one-shot human approval is required")
        return decision
