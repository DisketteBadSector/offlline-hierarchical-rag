"""Knowledge Management Bounded Context."""

from offline_rag.domain.knowledge.chunking import (
    ChunkingPolicy,
    ExtractedBlock,
    ExtractedDocumentContent,
    ExtractedTable,
    KnowledgeChunker,
)
from offline_rag.domain.knowledge.entities import (
    DocumentFormat,
    KnowledgeChunk,
    KnowledgeItem,
    KnowledgeVersion,
    KnowledgeVersionCreatedEvent,
    SourceReference,
)
from offline_rag.domain.knowledge.repositories import KnowledgeRepository
from offline_rag.domain.shared.value_objects import KnowledgeScope, MetadataRecord, VersionTag

__all__ = [
    "ChunkingPolicy",
    "DocumentFormat",
    "ExtractedBlock",
    "ExtractedDocumentContent",
    "ExtractedTable",
    "KnowledgeChunk",
    "KnowledgeChunker",
    "KnowledgeItem",
    "KnowledgeRepository",
    "KnowledgeScope",
    "KnowledgeVersion",
    "KnowledgeVersionCreatedEvent",
    "MetadataRecord",
    "SourceReference",
    "VersionTag",
]
