"""Issue-Knowledge-Executor (IKE) Bounded Context: 12 Relationship Types and Repository (Section 13)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence

from offline_rag.domain.shared.base import AggregateRoot
from offline_rag.domain.shared.errors import DomainValidationError
from offline_rag.domain.shared.value_objects import (
    NodeIdentifier,
    SimilarityScore,
    VersionTag,
)


class IKERelationshipType(str, Enum):
    """All 12 relationship types in the Issue-Knowledge-Executor model (Section 13.1)."""

    ISSUE_RELATED_TO_KNOWLEDGE = "issue_related_to_knowledge"
    ISSUE_DEPENDS_ON_KNOWLEDGE = "issue_depends_on_knowledge"
    ISSUE_LACKS_KNOWLEDGE = "issue_lacks_knowledge"
    ISSUE_SIMILAR_TO_ISSUE = "issue_similar_to_issue"
    ISSUE_DECOMPOSED_INTO_SUBISSUE = "issue_decomposed_into_subissue"
    ISSUE_BLOCKED_BY_ISSUE = "issue_blocked_by_issue"
    EXECUTOR_HAS_DOMAIN_EXPERTISE = "executor_has_domain_expertise"
    EXECUTOR_WORKED_ON_PROJECT = "executor_worked_on_project"
    EXECUTOR_LINKED_TO_KNOWLEDGE = "executor_linked_to_knowledge"
    EXECUTOR_CAN_PURSUE_ISSUE = "executor_can_pursue_issue"
    KNOWLEDGE_SUPPORTS_SOLUTION = "knowledge_supports_solution"
    KNOWLEDGE_CONSTRAINS_OR_CONFLICTS_SOLUTION = "knowledge_constrains_or_conflicts_solution"


class IKERelationship(AggregateRoot):
    """Typed, versioned, confidence-scored link in the Issue-Knowledge-Executor graph."""

    def __init__(
        self,
        relationship_id: str,
        node_id: NodeIdentifier,
        relationship_type: IKERelationshipType | str,
        source_entity_id: str,
        target_entity_id: str,
        confidence: SimilarityScore,
        created_in_version: VersionTag,
        evidence_reference: str,
    ) -> None:
        if not isinstance(relationship_id, str) or not relationship_id.strip():
            raise DomainValidationError("IKERelationship.relationship_id cannot be empty.")
        if not isinstance(node_id, NodeIdentifier):
            raise DomainValidationError("IKERelationship.node_id must be a NodeIdentifier.")
        try:
            rel_type = (
                relationship_type
                if isinstance(relationship_type, IKERelationshipType)
                else IKERelationshipType(str(relationship_type))
            )
        except ValueError as exc:
            raise DomainValidationError(
                f"Invalid IKERelationshipType: {relationship_type}"
            ) from exc

        if not isinstance(source_entity_id, str) or not source_entity_id.strip():
            raise DomainValidationError("IKERelationship.source_entity_id cannot be empty.")
        if not isinstance(target_entity_id, str) or not target_entity_id.strip():
            raise DomainValidationError("IKERelationship.target_entity_id cannot be empty.")
        if not isinstance(confidence, SimilarityScore):
            raise DomainValidationError("IKERelationship.confidence must be a SimilarityScore.")
        if not isinstance(created_in_version, VersionTag):
            raise DomainValidationError("IKERelationship.created_in_version must be a VersionTag.")
        if not isinstance(evidence_reference, str) or not evidence_reference.strip():
            raise DomainValidationError("IKERelationship.evidence_reference cannot be empty.")

        super().__init__(relationship_id.strip())
        self._relationship_id = relationship_id.strip()
        self._node_id = node_id
        self._relationship_type = rel_type
        self._source_entity_id = source_entity_id.strip()
        self._target_entity_id = target_entity_id.strip()
        self._confidence = confidence
        self._created_in_version = created_in_version
        self._evidence_reference = evidence_reference.strip()

    @property
    def relationship_id(self) -> str:
        return self._relationship_id

    @property
    def node_id(self) -> NodeIdentifier:
        return self._node_id

    @property
    def relationship_type(self) -> IKERelationshipType:
        return self._relationship_type

    @property
    def source_entity_id(self) -> str:
        return self._source_entity_id

    @property
    def target_entity_id(self) -> str:
        return self._target_entity_id

    @property
    def confidence(self) -> SimilarityScore:
        return self._confidence

    @property
    def created_in_version(self) -> VersionTag:
        return self._created_in_version

    @property
    def evidence_reference(self) -> str:
        return self._evidence_reference


class IKERepository(ABC):
    """Abstract repository for Issue-Knowledge-Executor relationships."""

    @abstractmethod
    def save(self, relationship: IKERelationship) -> None:
        """Persist or update an IKERelationship."""

    @abstractmethod
    def get_by_id(self, relationship_id: str) -> IKERelationship | None:
        """Retrieve an IKERelationship by ID."""

    @abstractmethod
    def list_by_node(self, node_id: NodeIdentifier) -> Sequence[IKERelationship]:
        """List all IKERelationships within a Node."""

    @abstractmethod
    def list_by_source(self, source_entity_id: str) -> Sequence[IKERelationship]:
        """List all IKERelationships originating from a given source entity ID."""


__all__ = ["IKERelationship", "IKERelationshipType", "IKERepository"]
