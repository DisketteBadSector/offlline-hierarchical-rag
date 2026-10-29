"""Executor Management Bounded Context: Executor Profile, Expertise, and Repository."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from offline_rag.domain.shared.base import AggregateRoot
from offline_rag.domain.shared.errors import DomainValidationError
from offline_rag.domain.shared.value_objects import NodeIdentifier, normalize_persian_text


class Executor(AggregateRoot):
    """Aggregate Root representing an individual, team, or unit capable of addressing Issues (Section 13.2)."""

    def __init__(
        self,
        executor_id: str,
        name: str,
        node_id: NodeIdentifier,
        organizational_role: str,
        expertise_domains: Iterable[str],
        past_project_ids: Iterable[str] = (),
        authored_knowledge_ids: Iterable[str] = (),
        availability_status: str = "available",
    ) -> None:
        if not isinstance(executor_id, str) or not executor_id.strip():
            raise DomainValidationError("Executor.executor_id cannot be empty.")
        if not isinstance(name, str) or not name.strip():
            raise DomainValidationError("Executor.name cannot be empty.")
        if not isinstance(node_id, NodeIdentifier):
            raise DomainValidationError("Executor.node_id must be a NodeIdentifier.")
        if not isinstance(organizational_role, str) or not organizational_role.strip():
            raise DomainValidationError("Executor.organizational_role cannot be empty.")

        cleaned_domains = tuple(
            normalize_persian_text(d)
            for d in expertise_domains
            if isinstance(d, str) and d.strip()
        )
        if not cleaned_domains:
            raise DomainValidationError(
                f"Executor '{executor_id}' must have at least one expertise domain."
            )

        super().__init__(executor_id.strip())
        self._executor_id = executor_id.strip()
        self._name = name.strip()
        self._node_id = node_id
        self._organizational_role = organizational_role.strip()
        self._expertise_domains: list[str] = list(dict.fromkeys(cleaned_domains))
        self._past_project_ids: list[str] = [
            p.strip() for p in past_project_ids if isinstance(p, str) and p.strip()
        ]
        self._authored_knowledge_ids: list[str] = [
            k.strip() for k in authored_knowledge_ids if isinstance(k, str) and k.strip()
        ]
        self._availability_status = availability_status.strip() or "available"

    @property
    def executor_id(self) -> str:
        return self._executor_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def node_id(self) -> NodeIdentifier:
        return self._node_id

    @property
    def organizational_role(self) -> str:
        return self._organizational_role

    @property
    def expertise_domains(self) -> tuple[str, ...]:
        return tuple(self._expertise_domains)

    @property
    def past_project_ids(self) -> tuple[str, ...]:
        return tuple(self._past_project_ids)

    @property
    def authored_knowledge_ids(self) -> tuple[str, ...]:
        return tuple(self._authored_knowledge_ids)

    @property
    def availability_status(self) -> str:
        return self._availability_status

    def matches_expertise(self, query_domain: str) -> bool:
        target = normalize_persian_text(query_domain).lower()
        if not target:
            return False
        return any(
            target in dom.lower() or dom.lower() in target for dom in self._expertise_domains
        )

    def add_expertise(self, domain: str) -> None:
        cleaned = normalize_persian_text(domain)
        if not cleaned:
            raise DomainValidationError("Expertise domain cannot be empty.")
        if cleaned not in self._expertise_domains:
            self._expertise_domains.append(cleaned)


class ExecutorRepository(ABC):
    """Abstract repository for Executor aggregates."""

    @abstractmethod
    def save(self, executor: Executor) -> None:
        """Persist or update an Executor."""

    @abstractmethod
    def get_by_id(self, executor_id: str) -> Executor | None:
        """Retrieve an Executor by identifier."""

    @abstractmethod
    def list_by_node(self, node_id: NodeIdentifier) -> Sequence[Executor]:
        """List all Executors belonging to an organizational Node."""

    @abstractmethod
    def find_by_expertise(
        self, domain_keyword: str, node_id: NodeIdentifier | None = None
    ) -> Sequence[Executor]:
        """Find Executors matching an expertise keyword (optionally scoped to a Node)."""


__all__ = ["Executor", "ExecutorRepository"]
