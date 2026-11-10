"""Exact/Near-Duplicate and Overlap detection domain models (Phase 5).

This module defines the domain-level value objects and the exact-duplicate
detector domain service. Near-duplicate and overlap detection that requires
embedding vectors lives in the Application layer
(``KnowledgeDeduplicationService``) because it depends on
``EmbeddingModelPort``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from offline_rag.domain.knowledge.entities import KnowledgeItem
from offline_rag.domain.knowledge.repositories import KnowledgeRepository
from offline_rag.domain.shared.errors import InvalidValueObjectError
from offline_rag.domain.shared.value_objects import SimilarityScore


class DuplicateKind(str, Enum):
    """Classification of a duplicate/overlap finding."""

    EXACT = "exact"
    NEAR = "near"
    OVERLAP = "overlap"


@dataclass(frozen=True)
class DuplicateMatch:
    """A single duplicate or overlap finding against an existing KnowledgeItem version."""

    kind: DuplicateKind
    existing_item_id: str
    existing_version_id: str
    similarity: SimilarityScore
    matched_chunk_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, DuplicateKind):
            raise InvalidValueObjectError("DuplicateMatch.kind must be a DuplicateKind.")
        if not isinstance(self.existing_item_id, str) or not self.existing_item_id.strip():
            raise InvalidValueObjectError("DuplicateMatch.existing_item_id cannot be empty.")
        if not isinstance(self.existing_version_id, str) or not self.existing_version_id.strip():
            raise InvalidValueObjectError("DuplicateMatch.existing_version_id cannot be empty.")
        if not isinstance(self.similarity, SimilarityScore):
            raise InvalidValueObjectError("DuplicateMatch.similarity must be a SimilarityScore.")
        cleaned_chunks = tuple(
            c.strip() for c in self.matched_chunk_ids if isinstance(c, str) and c.strip()
        )
        object.__setattr__(self, "existing_item_id", self.existing_item_id.strip())
        object.__setattr__(self, "existing_version_id", self.existing_version_id.strip())
        object.__setattr__(self, "matched_chunk_ids", cleaned_chunks)


@dataclass(frozen=True)
class DuplicateDetectionResult:
    """Outcome of a duplicate/overlap check for one candidate KnowledgeItem version."""

    query_content_hash: str
    is_exact_duplicate: bool
    exact_matches: tuple[str, ...] = ()
    near_duplicate_matches: tuple[DuplicateMatch, ...] = ()
    overlap_matches: tuple[DuplicateMatch, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.query_content_hash, str) or not self.query_content_hash.strip():
            raise InvalidValueObjectError(
                "DuplicateDetectionResult.query_content_hash cannot be empty."
            )
        if self.is_exact_duplicate and not self.exact_matches:
            raise InvalidValueObjectError(
                "DuplicateDetectionResult with is_exact_duplicate=True "
                "must provide at least one exact_match."
            )
        object.__setattr__(self, "query_content_hash", self.query_content_hash.strip())
        object.__setattr__(self, "exact_matches", tuple(self.exact_matches))
        object.__setattr__(self, "near_duplicate_matches", tuple(self.near_duplicate_matches))
        object.__setattr__(self, "overlap_matches", tuple(self.overlap_matches))

    @property
    def has_any_duplicate(self) -> bool:
        """Return True if an exact, near-duplicate, or overlap finding exists."""
        return (
            self.is_exact_duplicate
            or bool(self.near_duplicate_matches)
            or bool(self.overlap_matches)
        )


class ExactDuplicateDetector:
    """Domain service detecting exact duplicate content hashes across KnowledgeItems."""

    def __init__(self, repository: KnowledgeRepository) -> None:
        self._repository = repository

    def detect(
        self,
        content_hash: str,
        exclude_item_id: str | None = None,
    ) -> DuplicateDetectionResult:
        """Look for KnowledgeItems whose active version hash matches ``content_hash``."""
        if not isinstance(content_hash, str) or not content_hash.strip():
            raise InvalidValueObjectError(
                "ExactDuplicateDetector.detect requires a non-empty content_hash."
            )
        matches: tuple[KnowledgeItem, ...] = tuple(
            self._repository.find_by_active_content_hash(content_hash.strip())
        )
        exact_ids = tuple(
            item.item_id
            for item in matches
            if exclude_item_id is None or item.item_id != exclude_item_id
        )
        return DuplicateDetectionResult(
            query_content_hash=content_hash.strip(),
            is_exact_duplicate=bool(exact_ids),
            exact_matches=exact_ids,
        )


__all__ = [
    "DuplicateDetectionResult",
    "DuplicateKind",
    "DuplicateMatch",
    "ExactDuplicateDetector",
]