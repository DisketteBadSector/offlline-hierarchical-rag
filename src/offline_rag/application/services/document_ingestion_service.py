"""Document Ingestion Application Service (Phase 4 - Section 7.2, 7.3, 7.4, 7.5)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from offline_rag.application.ports import (
    DocumentStoragePort,
    EmbeddingModelPort,
    LvmLayoutAssistantPort,
    UnitOfWork,
    VectorIndexPort,
)
from offline_rag.domain.knowledge import (
    ChunkingPolicy,
    DocumentFormat,
    ExtractedDocumentContent,
    KnowledgeChunk,
    KnowledgeChunker,
    KnowledgeItem,
    KnowledgeScope,
    KnowledgeVersion,
    MetadataRecord,
)
from offline_rag.domain.shared.errors import (
    DomainValidationError,
    EntityNotFoundError,
    InvariantViolationError,
)
from offline_rag.domain.shared.value_objects import (
    CONFIDENTIALITY_RANK,
    NodeIdentifier,
)
from offline_rag.infrastructure.ingestion import DocumentExtractorRegistry


@dataclass(frozen=True)
class DocumentIngestionResult:
    """Structured result returned by DocumentIngestionService after ingesting a document."""

    knowledge_item: KnowledgeItem
    active_version: KnowledgeVersion
    chunks: tuple[KnowledgeChunk, ...]
    extracted_content: ExtractedDocumentContent
    lvm_assisted: bool
    storage_uri: str | None = None


class DocumentIngestionService:
    """Application service orchestrating end-to-end offline document ingestion (DOC, DOCX, PDF)."""

    def __init__(
        self,
        uow: UnitOfWork,
        extractor_registry: DocumentExtractorRegistry | None = None,
        chunker: KnowledgeChunker | None = None,
        lvm_assistant: LvmLayoutAssistantPort | None = None,
        document_storage: DocumentStoragePort | None = None,
        vector_index: VectorIndexPort | None = None,
        embedding_port: EmbeddingModelPort | None = None,
    ) -> None:
        self._uow = uow
        self._extractor_registry = extractor_registry or DocumentExtractorRegistry()
        self._chunker = chunker or KnowledgeChunker(ChunkingPolicy())
        self._lvm_assistant = lvm_assistant
        self._document_storage = document_storage
        self._vector_index = vector_index
        self._embedding_port = embedding_port

    def ingest_file(
        self,
        file_path: Path | str,
        owner_node_id: NodeIdentifier,
        metadata: MetadataRecord,
        document_id: str | None = None,
        title: str | None = None,
        force_lvm_assistance: bool = False,
        enforce_strict_node_confidentiality: bool = False,
    ) -> DocumentIngestionResult:
        """Ingest a local DOC, DOCX, or PDF file into a controlled KnowledgeItem, Version, and Chunks."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {path}")

        raw_bytes = path.read_bytes()
        resolved_doc_id = document_id.strip() if document_id and document_id.strip() else path.stem
        resolved_title = title.strip() if title and title.strip() else path.stem

        return self.ingest_bytes(
            raw_bytes=raw_bytes,
            file_extension_or_format=path.suffix,
            document_id=resolved_doc_id,
            title=resolved_title,
            owner_node_id=owner_node_id,
            metadata=metadata,
            source_path=str(path),
            force_lvm_assistance=force_lvm_assistance,
            enforce_strict_node_confidentiality=enforce_strict_node_confidentiality,
        )

    def ingest_bytes(
        self,
        raw_bytes: bytes,
        file_extension_or_format: DocumentFormat | str,
        document_id: str,
        title: str,
        owner_node_id: NodeIdentifier,
        metadata: MetadataRecord,
        source_path: str | None = None,
        force_lvm_assistance: bool = False,
        enforce_strict_node_confidentiality: bool = False,
    ) -> DocumentIngestionResult:
        """Ingest raw document bytes (DOC, DOCX, or PDF) and persist KnowledgeItem + KnowledgeChunks."""
        if not isinstance(owner_node_id, NodeIdentifier):
            raise DomainValidationError("owner_node_id must be a NodeIdentifier.")
        if not isinstance(metadata, MetadataRecord):
            raise DomainValidationError("metadata must be a MetadataRecord.")

        # 1. Validate owner Node existence and confidentiality alignment
        node = self._uow.nodes.get_by_id(owner_node_id)
        if node is None:
            raise EntityNotFoundError(
                f"Owner Node '{owner_node_id}' is not registered in NodeRepository."
            )
        if metadata.node_id != owner_node_id:
            raise InvariantViolationError(
                f"Metadata node_id ('{metadata.node_id}') does not match owner_node_id ('{owner_node_id}')."
            )
        if enforce_strict_node_confidentiality and (
            CONFIDENTIALITY_RANK[metadata.confidentiality]
            > CONFIDENTIALITY_RANK[node.confidentiality]
        ):
            raise InvariantViolationError(
                f"Document confidentiality ('{metadata.confidentiality}') exceeds "
                f"Node '{owner_node_id}' ceiling ('{node.confidentiality}')."
            )

        # 2. Select extractor adapter by file extension / format and extract content
        extractor = self._extractor_registry.get_extractor(file_extension_or_format)
        extracted = extractor.extract_from_bytes(raw_bytes=raw_bytes, document_id=document_id)

        if force_lvm_assistance and not extracted.requires_lvm_assistance:
            extracted = replace(
                extracted,
                requires_lvm_assistance=True,
                flagged_lvm_pages=extracted.flagged_lvm_pages or (1,),
            )

        # 3. Run optional LVM layout assistant if document is flagged
        v1_tag = metadata.active_version_tag
        assert v1_tag is not None
        lvm_used = False
        if extracted.requires_lvm_assistance and self._lvm_assistant is not None:
            extracted = self._lvm_assistant.enrich_extracted_content(
                extracted=extracted, version_id=v1_tag.tag
            )
            lvm_used = True

        # 4. Chunk extracted content preserving SourceReference and inherited MetadataRecord
        chunks = self._chunker.chunk_document(
            extracted=extracted,
            version_tag=v1_tag,
            inherited_metadata=metadata,
        )

        # 5. Create KnowledgeItem and initial KnowledgeVersion (v1)
        scope = KnowledgeScope(
            owner_node_id=owner_node_id,
            confidentiality=metadata.confidentiality,
            usage_domain=metadata.domain,
        )
        item = KnowledgeItem(
            item_id=document_id,
            title=title,
            scope=scope,
            metadata=metadata,
            initial_content_hash=extracted.raw_content_hash or "hash-v1",
            document_format=extracted.file_format,
            initial_chunks=chunks,
            initial_change_reason="Initial document ingestion (v1)",
            source_path=source_path,
            requires_lvm_interpretation=extracted.requires_lvm_assistance,
        )
        if lvm_used and extracted.lvm_annotations:
            item.attach_lvm_interpretation("\n".join(extracted.lvm_annotations))

        # 6. Persist in KnowledgeRepository (and optional DocumentStorage + VectorIndex)
        storage_uri: str | None = None
        with self._uow:
            if self._document_storage is not None:
                storage_key = f"{owner_node_id}/{document_id}@{v1_tag.tag}.{extracted.file_format.value.lower()}"
                storage_uri = self._document_storage.save_document(storage_key, raw_bytes)

            self._uow.knowledge.save(item)

            if self._vector_index is not None and self._embedding_port is not None:
                self._vector_index.remove_item_chunks(item.item_id)
                for chk in chunks:
                    vec = self._embedding_port.embed_text(chk.text)
                    self._vector_index.index_chunk(chk, vec)

            self._uow.commit()

        return DocumentIngestionResult(
            knowledge_item=item,
            active_version=item.active_version,
            chunks=chunks,
            extracted_content=extracted,
            lvm_assisted=lvm_used,
            storage_uri=storage_uri,
        )


__all__ = ["DocumentIngestionResult", "DocumentIngestionService"]
