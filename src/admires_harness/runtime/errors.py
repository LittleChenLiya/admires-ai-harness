"""Typed failures raised by the graph runtime."""

from __future__ import annotations


class RuntimeErrorBase(Exception):
    """Base class for failures callers may handle without parsing text."""


class GraphDefinitionError(RuntimeErrorBase, ValueError):
    """A graph is incomplete, ambiguous, or references an unknown node."""


class GraphRoutingError(RuntimeErrorBase):
    """A node requested a transition that the compiled graph cannot satisfy."""


class RunAlreadyExistsError(RuntimeErrorBase):
    """A fresh execution attempted to reuse an existing run identifier."""


class RunNotFoundError(RuntimeErrorBase):
    """No durable checkpoint exists for the requested run."""


class TerminalRunError(RuntimeErrorBase):
    """A terminal run cannot be resumed."""


class RunLeaseActiveError(RuntimeErrorBase):
    """Another worker currently owns the durable execution lease."""


class IncompatibleCheckpointError(RuntimeErrorBase):
    """The supplied graph is not compatible with the stored checkpoint."""


class CheckpointConflictError(RuntimeErrorBase):
    """Optimistic checkpoint revision validation failed."""


class DelegationError(RuntimeErrorBase):
    """One or more delegated agents could not complete their plan step."""


class BudgetExceededError(RuntimeErrorBase):
    """A run or delegated step exhausted its declared budget."""
