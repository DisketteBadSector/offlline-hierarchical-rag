"""Repository interface for the Knowledge Management Bounded Context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from offline_rag.domain.knowledge.entities import KnowledgeChunk, KnowledgeItem
from offline_rag.domain.shared.value_objects import NodeIdentifier


class KnowledgeRepository(ABC):
    """Abstract repository for KnowledgeItem aggregates and their active chunks."""

    @abstractmethod
    def save(self, item: KnowledgeItem) -> None:
        """Persist or update a KnowledgeItem aggregate."""

    @abstractmethod
    def get_by_id(self, item_id: str) -> KnowledgeItem | None:
        """Retrieve a KnowledgeItem by its identifier."""

    @abstractmethod
    def list_by_node(self, node_id: NodeIdentifier) -> Sequence[KnowledgeItem]:
        """Return all KnowledgeItems owned by the specified organizational Node."""

    @abstractmethod
    def find_by_active_content_hash(self, content_hash: str) -> Sequence[KnowledgeItem]:
        """Find KnowledgeItems whose active version has the given content hash (exact duplicate check)."""

    @abstractmethod
    def list_active_chunks_by_node(self, node_id: NodeIdentifier) -> Sequence[KnowledgeChunk]:
        """Return all chunks belonging to the active versions of a Node's KnowledgeItems."""
