"""Domain layer Exceptions for Offline Hierarchical RAG Core."""

from __future__ import annotations


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


class EntityNotFoundError(DomainError):
    """Raised when a requested domain entity does not exist in a repository."""
