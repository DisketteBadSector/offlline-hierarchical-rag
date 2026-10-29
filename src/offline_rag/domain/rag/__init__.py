"""RAG Core Bounded Context: Retrieval, Context Assembly, and Grounded Response Models."""

from __future__ import annotations

from dataclasses import dataclass, field

from offline_rag.domain.knowledge.entities import KnowledgeChunk
from offline_rag.domain.shared.errors import DomainValidationError, InvariantViolationError
from offline_rag.domain.shared.value_objects import (
    NodeIdentifier,
    RetrievalQuery,
    SimilarityScore,
)


@dataclass(frozen=True)
class RetrievedChunk:
    """A KnowledgeChunk paired with its retrieval SimilarityScore."""

    chunk: KnowledgeChunk
    similarity: SimilarityScore

    def __post_init__(self) -> None:
        if not isinstance(self.chunk, KnowledgeChunk):
            raise DomainValidationError("RetrievedChunk.chunk must be a KnowledgeChunk.")
        if not isinstance(self.similarity, SimilarityScore):
            raise DomainValidationError("RetrievedChunk.similarity must be a SimilarityScore.")


@dataclass(frozen=True)
class RAGContextPackage:
    """Assembled RAG context enforcing node boundary and confidentiality rules (Section 8.1)."""

    query: RetrievalQuery
    retrieved_chunks: tuple[RetrievedChunk, ...]
    max_context_chars: int = 12000

    def __post_init__(self) -> None:
        if not isinstance(self.query, RetrievalQuery):
            raise DomainValidationError("RAGContextPackage.query must be a RetrievalQuery.")
        if self.max_context_chars <= 0:
            raise DomainValidationError("RAGContextPackage.max_context_chars must be > 0.")

        chunks = tuple(self.retrieved_chunks)
        for rc in chunks:
            chunk_node = rc.chunk.inherited_metadata.node_id
            if chunk_node != self.query.requesting_node_id:
                raise InvariantViolationError(
                    f"Node boundary violation: chunk '{rc.chunk.chunk_id}' belongs to node "
                    f"'{chunk_node}', but query is scoped to '{self.query.requesting_node_id}'."
                )
        object.__setattr__(self, "retrieved_chunks", chunks)

    @property
    def is_empty(self) -> bool:
        return len(self.retrieved_chunks) == 0

    @property
    def source_references(self) -> tuple[str, ...]:
        refs: list[str] = []
        for rc in self.retrieved_chunks:
            ref = rc.chunk.citation_reference
            if ref not in refs:
                refs.append(ref)
        return tuple(refs)

    def format_context_text(self) -> str:
        """Assemble context blocks with explicit source citations up to max_context_chars."""
        blocks: list[str] = []
        current_len = 0
        for rc in self.retrieved_chunks:
            block = f"[{rc.chunk.citation_reference}] {rc.chunk.text}"
            if current_len + len(block) > self.max_context_chars:
                break
            blocks.append(block)
            current_len += len(block) + 1
        return "\n".join(blocks)


@dataclass(frozen=True)
class RAGResponse:
    """Grounded question-answering or analysis response produced by the RAG pipeline (Section 8.4)."""

    query: RetrievalQuery
    answer_text: str
    source_references: tuple[str, ...] = ()
    uncertainty_declared: bool = False
    uncertainty_reason: str | None = None
    suggested_issue_ids: tuple[str, ...] = field(default_factory=tuple)
    suggested_executor_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.query, RetrievalQuery):
            raise DomainValidationError("RAGResponse.query must be a RetrievalQuery.")
        if not isinstance(self.answer_text, str) or not self.answer_text.strip():
            raise DomainValidationError("RAGResponse.answer_text cannot be empty.")
        refs = tuple(r.strip() for r in self.source_references if isinstance(r, str) and r.strip())
        if not refs and not self.uncertainty_declared:
            raise InvariantViolationError(
                "RAGResponse without source_references must explicitly declare uncertainty."
            )
        object.__setattr__(self, "answer_text", self.answer_text.strip())
        object.__setattr__(self, "source_references", refs)


__all__ = [
    "NodeIdentifier",
    "RAGContextPackage",
    "RAGResponse",
    "RetrievalQuery",
    "RetrievedChunk",
    "SimilarityScore",
]
