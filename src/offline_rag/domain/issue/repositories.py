"""Repository interface for the Issue Management Bounded Context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from offline_rag.domain.issue.entities import Issue
from offline_rag.domain.shared.value_objects import NodeIdentifier


class IssueRepository(ABC):
    """Abstract repository for Issue aggregates and their SubIssues."""

    @abstractmethod
    def save(self, issue: Issue) -> None:
        """Persist or update an Issue aggregate."""

    @abstractmethod
    def get_by_id(self, issue_id: str) -> Issue | None:
        """Retrieve an Issue by its identifier."""

    @abstractmethod
    def list_by_node(self, node_id: NodeIdentifier) -> Sequence[Issue]:
        """Return all Issues belonging to the given Node."""
