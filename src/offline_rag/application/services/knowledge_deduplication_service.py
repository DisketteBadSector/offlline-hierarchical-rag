"""Knowledge deduplication application service (Phase 5).

Detects exact duplicates (via ``ExactDuplicateDetector``), near-duplicates and
partial overlaps (via embedding cosine similarity) between a candidate
KnowledgeItem version and the existing KnowledgeItems of the same Node.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from offline_rag.application.ports import EmbeddingModelPort, UnitOfWork
from offline_rag.domain.knowledge.deduplication import (
    DuplicateDetectionResult,
    DuplicateKind,
    DuplicateMatch,
    ExactDuplicateDetector,
)
from offline_rag.domain.knowledge.entities import KnowledgeChunk
from offline_rag.domain.shared.value_objects import NodeIdentifier, SimilarityScore


@dataclass(frozen=True)
class DeduplicationPolicy:
    """Configurable thresholds and bounds for near-duplicate/overlap detection."""

    near_duplicate_threshold: float = 0.92
    overlap_threshold: float = 0.80
    max_candidates_per_node: int = 200

    def __post_init__(self) -> None:
        if not (0.0 <= self.overlap_threshold < self.near_duplicate_threshold <= 1.0):
            raise ValueError(
                "DeduplicationPolicy requires 0 <= overlap_threshold < "
                "near_duplicate_threshold <= 1."
            )
        if self.max_candidates_per_node <= 0:
            raise ValueError("DeduplicationPolicy.max_candidates_per_node must be > 0.")


class KnowledgeDeduplicationService:
    """Detect exact/near duplicates and overlaps against a Node's active KnowledgeChunks."""

    def __init__(
        self,
        uow: UnitOfWork,
        embedding_port: EmbeddingModelPort,
        policy: DeduplicationPolicy | None = None,
    ) -> None:
        self._uow = uow
        self._embedding = embedding_port
        self._policy = policy or DeduplicationPolicy()
        self._exact_detector = ExactDuplicateDetector(uow.knowledge)

    @property
    def policy(self) -> DeduplicationPolicy:
        return self._policy

    @staticmethod
    def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return max(0.0, min(1.0, dot / (na * nb)))

    def detect_for_chunks(
        self,
        node_id: NodeIdentifier,
        candidate_chunks: tuple[KnowledgeChunk, ...],
        content_hash: str,
        exclude_item_id: str | None = None,
    ) -> DuplicateDetectionResult:
        """Run exact + near-duplicate + overlap detection for a candidate version."""
        if not isinstance(node_id, NodeIdentifier):
            raise ValueError("node_id must be a NodeIdentifier.")

        # 1) Exact duplicate short-circuit
        exact_result = self._exact_detector.detect(
            content_hash=content_hash, exclude_item_id=exclude_item_id
        )
        if exact_result.is_exact_duplicate:
            return exact_result

        if not candidate_chunks:
            return exact_result

        # 2) Compute candidate vectors once
        candidate_vecs = [self._embedding.embed_text(chk.text) for chk in candidate_chunks]

        # 3) Compare against existing active chunks of the same Node
        existing_chunks = self._uow.knowledge.list_active_chunks_by_node(node_id)[
            : self._policy.max_candidates_per_node
        ]

        near_matches: list[DuplicateMatch] = []
        overlap_matches: list[DuplicateMatch] = []
        for existing in existing_chunks:
            if exclude_item_id and existing.source_item_id == exclude_item_id:
                continue
            existing_vec = self._embedding.embed_text(existing.text)
            best_similarity = max(
                (self._cosine(cv, existing_vec) for cv in candidate_vecs),
                default=0.0,
            )
            similarity = SimilarityScore(best_similarity)

            if best_similarity >= self._policy.near_duplicate_threshold:
                near_matches.append(
                    DuplicateMatch(
                        kind=DuplicateKind.NEAR,
                        existing_item_id=existing.source_item_id,
                        existing_version_id=existing.version_tag.tag,
                        similarity=similarity,
                        matched_chunk_ids=(existing.chunk_id,),
                    )
                )
            elif best_similarity >= self._policy.overlap_threshold:
                overlap_matches.append(
                    DuplicateMatch(
                        kind=DuplicateKind.OVERLAP,
                        existing_item_id=existing.source_item_id,
                        existing_version_id=existing.version_tag.tag,
                        similarity=similarity,
                        matched_chunk_ids=(existing.chunk_id,),
                    )
                )

        return DuplicateDetectionResult(
            query_content_hash=content_hash,
            is_exact_duplicate=False,
            exact_matches=(),
            near_duplicate_matches=self._collapse_by_item(near_matches),
            overlap_matches=self._collapse_by_item(overlap_matches),
        )

    @staticmethod
    def _collapse_by_item(matches: list[DuplicateMatch]) -> tuple[DuplicateMatch, ...]:
        """Keep the strongest match per existing item, and sort deterministically."""
        best_by_item: dict[str, DuplicateMatch] = {}
        for m in matches:
            prev = best_by_item.get(m.existing_item_id)
            if prev is None or m.similarity.value > prev.similarity.value:
                best_by_item[m.existing_item_id] = m
        return tuple(
            sorted(best_by_item.values(), key=lambda m: m.similarity.value, reverse=True)
        )


__all__ = ["DeduplicationPolicy", "KnowledgeDeduplicationService"]