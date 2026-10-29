"""Domain layer Exceptions for Offline Hierarchical RAG Core."""

from __future__ import annotations

from typing import Sequence


class DomainError(Exception):
    """Base exception for all domain layer errors."""


class DomainValidationError(DomainError):
    """Raised when a domain invariant or validation rule is violated."""


class InvalidValueObjectError(DomainValidationError):
    """Raised when a Value Object is constructed with invalid values."""


class HierarchyRuleViolationError(DomainValidationError):
    """Raised when organizational hierarchy validation rules (R-01 to R-08) are violated."""

    def __init__(self, rule_id: str, message: str) -> None:
        self.rule_id = rule_id
        super().__init__(f"[{rule_id}] {message}")


class InvariantViolationError(DomainValidationError):
    """Raised when an Entity or Aggregate Root state transition violates business rules."""


class MissingPromptVariableError(InvariantViolationError):
    """Raised when one or more required variables are missing during prompt assembly."""

    def __init__(
        self,
        prompt_id: str,
        missing_variables: Sequence[str],
        message: str | None = None,
    ) -> None:
        self.prompt_id = prompt_id
        self.missing_variables = tuple(missing_variables)
        default_msg = (
            f"Missing required prompt variables for '{prompt_id}': {list(self.missing_variables)}"
        )
        super().__init__(message or default_msg)


class EntityNotFoundError(DomainError):
    """Raised when a requested domain entity does not exist in a repository."""
