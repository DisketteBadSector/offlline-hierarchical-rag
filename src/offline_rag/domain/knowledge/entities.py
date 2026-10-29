"""Knowledge Management Bounded Context: SourceReference, KnowledgeItem, KnowledgeVersion, KnowledgeChunk."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Iterable, Sequence

from offline_rag.domain.shared.base import AggregateRoot, DomainEvent, Entity, utc_now
from offline_rag.domain.shared.errors import (
    DomainValidationError,
    InvalidValueObjectError,
    InvariantViolationError,
)
from offline_rag.domain.shared.value_objects import (
    KnowledgeScope,
    MetadataRecord,
    NodeIdentifier,
    VersionTag,
)


class DocumentFormat(str, Enum):
    """Supported document and knowledge source formats."""

    DOC = "DOC"
    DOCX = "DOCX"
    PDF = "PDF"
    STRUCTURED_MODEL = "STRUCTURED_MODEL"


@dataclass(frozen=True)
class SourceReference:
    """Immutable Value Object representing fine-grained source attribution (Section 7.4 & Phase 4)."""

    document_id: str
    version_id: str
    file_format: DocumentFormat | str = DocumentFormat.DOCX
    page_number: int | None = None
    section_reference: str | None = None
    heading_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise InvalidValueObjectError("SourceReference.document_id cannot be empty.")
        if isinstance(self.version_id, VersionTag):
            object.__setattr__(self, "version_id", self.version_id.tag)
        elif not isinstance(self.version_id, str) or not self.version_id.strip():
            raise InvalidValueObjectError("SourceReference.version_id cannot be empty.")
        else:
            object.__setattr__(self, "version_id", self.version_id.strip())

        try:
            fmt = (
                self.file_format
                if isinstance(self.file_format, DocumentFormat)
                else DocumentFormat(str(self.file_format).strip().upper().lstrip("."))
            )
        except ValueError as exc:
            raise InvalidValueObjectError(
                f"Invalid SourceReference.file_format '{self.file_format}'."
            ) from exc

        if self.page_number is not None and (
            isinstance(self.page_number, bool)
            or not isinstance(self.page_number, int)
            or self.page_number < 1
        ):
            raise InvalidValueObjectError(
                "SourceReference.page_number must be an integer >= 1 or None."
            )

        sec_ref = (
            self.section_reference.strip()
            if isinstance(self.section_reference, str) and self.section_reference.strip()
            else None
        )
        headings = tuple(
            h.strip() for h in self.heading_path if isinstance(h, str) and h.strip()
        )

        object.__setattr__(self, "document_id", self.document_id.strip())
        object.__setattr__(self, "file_format", fmt)
        object.__setattr__(self, "section_reference", sec_ref)
        object.__setattr__(self, "heading_path", headings)

    def to_citation_string(self, sequence_index: int | None = None) -> str:
        """Format canonical citation string."""
        page_part = f"#p{self.page_number}" if self.page_number is not None else ""
        sec_part = f"§{self.section_reference}" if self.section_reference else ""
        chunk_part = f":c{sequence_index}" if sequence_index is not None else ""
        return f"{self.document_id}@{self.version_id}{page_part}{sec_part}{chunk_part}"


@dataclass(frozen=True)
class KnowledgeVersionCreatedEvent(DomainEvent):
    """Emitted when a new KnowledgeVersion is added to a KnowledgeItem."""

    knowledge_item_id: str
    version_tag: VersionTag
    previous_version_tag: VersionTag | None
    node_id: NodeIdentifier


class KnowledgeChunk(Entity):
    """Retrievable, fine-grained chunk of a KnowledgeVersion (Section 7.4)."""

    def __init__(
        self,
        chunk_id: str,
        text: str,
        inherited_metadata: MetadataRecord,
        source_item_id: str | None = None,
        version_tag: VersionTag | None = None,
        chunk_index: int | None = None,
        sequence_index: int | None = None,
        source_reference: SourceReference | None = None,
        page_number: int | None = None,
        section_reference: str | None = None,
        section_heading_path: Sequence[str] = (),
        start_char_offset: int | None = None,
        end_char_offset: int | None = None,
        context_boundary_start: int | None = None,
        context_boundary_end: int | None = None,
        file_format: DocumentFormat | str = DocumentFormat.DOCX,
    ) -> None:
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise DomainValidationError("KnowledgeChunk.chunk_id cannot be empty.")
        if not isinstance(text, str) or not text.strip():
            raise DomainValidationError("KnowledgeChunk.text cannot be empty.")
        if not isinstance(inherited_metadata, MetadataRecord):
            raise DomainValidationError(
                "KnowledgeChunk.inherited_metadata must be a MetadataRecord."
            )

        # Resolve sequence_index / chunk_index
        resolved_idx = sequence_index if sequence_index is not None else chunk_index
        if (
            resolved_idx is None
            or isinstance(resolved_idx, bool)
            or not isinstance(resolved_idx, int)
            or resolved_idx < 0
        ):
            raise DomainValidationError(
                "KnowledgeChunk requires a non-negative integer sequence_index / chunk_index."
            )

        # Resolve SourceReference, source_item_id, and version_tag
        if source_reference is not None:
            if not isinstance(source_reference, SourceReference):
                raise DomainValidationError(
                    "KnowledgeChunk.source_reference must be a SourceReference."
                )
            resolved_doc_id = (
                source_item_id.strip()
                if isinstance(source_item_id, str) and source_item_id.strip()
                else source_reference.document_id
            )
            if resolved_doc_id != source_reference.document_id:
                raise InvariantViolationError(
                    f"source_item_id ('{resolved_doc_id}') does not match "
                    f"source_reference.document_id ('{source_reference.document_id}')."
                )
            resolved_vtag = (
                version_tag
                if version_tag is not None
                else VersionTag.from_version_id(source_reference.version_id)
            )
            if resolved_vtag.tag != source_reference.version_id:
                raise InvariantViolationError(
                    f"version_tag ('{resolved_vtag.tag}') does not match "
                    f"source_reference.version_id ('{source_reference.version_id}')."
                )
            resolved_src_ref = source_reference
        else:
            if not isinstance(source_item_id, str) or not source_item_id.strip():
                raise DomainValidationError("KnowledgeChunk.source_item_id cannot be empty.")
            if not isinstance(version_tag, VersionTag):
                raise DomainValidationError("KnowledgeChunk.version_tag must be a VersionTag.")
            resolved_doc_id = source_item_id.strip()
            resolved_vtag = version_tag
            resolved_src_ref = SourceReference(
                document_id=resolved_doc_id,
                version_id=resolved_vtag.tag,
                file_format=file_format,
                page_number=page_number,
                section_reference=section_reference,
                heading_path=tuple(section_heading_path),
            )

        # Resolve character boundary offsets
        resolved_start = (
            context_boundary_start
            if context_boundary_start is not None
            else (start_char_offset if start_char_offset is not None else 0)
        )
        resolved_end = (
            context_boundary_end
            if context_boundary_end is not None
            else (end_char_offset if end_char_offset is not None else 0)
        )
        if resolved_start < 0 or resolved_end < resolved_start:
            raise DomainValidationError("Invalid KnowledgeChunk character boundary offsets.")

        cleaned_text = text.strip()
        if resolved_end == 0:
            resolved_end = resolved_start + len(cleaned_text)

        super().__init__(chunk_id.strip())
        self._chunk_id = chunk_id.strip()
        self._source_item_id = resolved_doc_id
        self._version_tag = resolved_vtag
        self._sequence_index = resolved_idx
        self._text = cleaned_text
        self._inherited_metadata = inherited_metadata
        self._source_reference = resolved_src_ref
        self._context_boundary_start = resolved_start
        self._context_boundary_end = resolved_end

    @property
    def chunk_id(self) -> str:
        return self._chunk_id

    @property
    def source_item_id(self) -> str:
        return self._source_item_id

    @property
    def version_tag(self) -> VersionTag:
        return self._version_tag

    @property
    def sequence_index(self) -> int:
        return self._sequence_index

    @property
    def chunk_index(self) -> int:
        return self._sequence_index

    @property
    def text(self) -> str:
        return self._text

    @property
    def inherited_metadata(self) -> MetadataRecord:
        return self._inherited_metadata

    @property
    def source_reference(self) -> SourceReference:
        return self._source_reference

    @property
    def page_number(self) -> int | None:
        return self._source_reference.page_number

    @property
    def section_reference(self) -> str | None:
        return self._source_reference.section_reference

    @property
    def heading_path(self) -> tuple[str, ...]:
        return self._source_reference.heading_path

    @property
    def section_heading_path(self) -> tuple[str, ...]:
        return self._source_reference.heading_path

    @property
    def context_boundary_start(self) -> int:
        return self._context_boundary_start

    @property
    def context_boundary_end(self) -> int:
        return self._context_boundary_end

    @property
    def start_char_offset(self) -> int:
        return self._context_boundary_start

    @property
    def end_char_offset(self) -> int:
        return self._context_boundary_end

    @property
    def citation_reference(self) -> str:
        """Return canonical source reference string (item + version + page/chunk)."""
        return self._source_reference.to_citation_string(sequence_index=self._sequence_index)


class KnowledgeVersion(Entity):
    """Specific version of a KnowledgeItem preserving immutable audit history (Section 7.6)."""

    def __init__(
        self,
        knowledge_item_id: str,
        version_tag: VersionTag,
        content_hash: str,
        change_reason: str,
        metadata_snapshot: MetadataRecord,
        previous_version_tag: VersionTag | None = None,
        is_active: bool = True,
        chunks: Iterable[KnowledgeChunk] = (),
        diff_summary: str | None = None,
        created_at: datetime | None = None,
    ) -> None:
        if not isinstance(knowledge_item_id, str) or not knowledge_item_id.strip():
            raise DomainValidationError("KnowledgeVersion.knowledge_item_id cannot be empty.")
        if not isinstance(version_tag, VersionTag):
            raise DomainValidationError("KnowledgeVersion.version_tag must be a VersionTag.")
        if not isinstance(content_hash, str) or not content_hash.strip():
            raise DomainValidationError("KnowledgeVersion.content_hash cannot be empty.")
        if not isinstance(change_reason, str) or not change_reason.strip():
            raise DomainValidationError("KnowledgeVersion.change_reason cannot be empty.")
        if not isinstance(metadata_snapshot, MetadataRecord):
            raise DomainValidationError(
                "KnowledgeVersion.metadata_snapshot must be a MetadataRecord."
            )
        if previous_version_tag is not None:
            if not isinstance(previous_version_tag, VersionTag):
                raise DomainValidationError("previous_version_tag must be a VersionTag.")
            if previous_version_tag.sequence_number >= version_tag.sequence_number:
                raise InvariantViolationError(
                    "previous_version_tag must have a smaller sequence_number than version_tag."
                )

        chunk_list = tuple(chunks)
        for ch in chunk_list:
            if ch.source_item_id != knowledge_item_id.strip():
                raise InvariantViolationError(
                    f"Chunk '{ch.chunk_id}' source_item_id '{ch.source_item_id}' "
                    f"does not match KnowledgeItem '{knowledge_item_id}'."
                )
            if ch.version_tag != version_tag:
                raise InvariantViolationError(
                    f"Chunk '{ch.chunk_id}' version_tag '{ch.version_tag}' "
                    f"does not match KnowledgeVersion '{version_tag}'."
                )

        version_id = f"{knowledge_item_id.strip()}:{version_tag.tag}"
        super().__init__(version_id)
        self._knowledge_item_id = knowledge_item_id.strip()
        self._version_tag = version_tag
        self._content_hash = content_hash.strip()
        self._change_reason = change_reason.strip()
        self._metadata_snapshot = metadata_snapshot
        self._previous_version_tag = previous_version_tag
        self._is_active = bool(is_active)
        self._chunks = chunk_list
        self._diff_summary = diff_summary.strip() if diff_summary else None
        self._created_at = created_at or utc_now()

    @property
    def knowledge_item_id(self) -> str:
        return self._knowledge_item_id

    @property
    def version_tag(self) -> VersionTag:
        return self._version_tag

    @property
    def content_hash(self) -> str:
        return self._content_hash

    @property
    def change_reason(self) -> str:
        return self._change_reason

    @property
    def metadata_snapshot(self) -> MetadataRecord:
        return self._metadata_snapshot

    @property
    def previous_version_tag(self) -> VersionTag | None:
        return self._previous_version_tag

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def chunks(self) -> tuple[KnowledgeChunk, ...]:
        return self._chunks

    @property
    def diff_summary(self) -> str | None:
        return self._diff_summary

    @property
    def created_at(self) -> datetime:
        return self._created_at

    def _deactivate(self) -> None:
        """Internal aggregate method to mark this version inactive when superseded."""
        self._is_active = False

    def replace_chunks(self, chunks: Sequence[KnowledgeChunk]) -> None:
        """Attach or replace chunks belonging to this version."""
        for ch in chunks:
            if ch.source_item_id != self._knowledge_item_id or ch.version_tag != self._version_tag:
                raise InvariantViolationError("Chunk does not match item ID or version tag.")
        self._chunks = tuple(chunks)


class KnowledgeItem(AggregateRoot):
    """Aggregate Root representing a controlled organizational knowledge asset (Section 7)."""

    def __init__(
        self,
        item_id: str,
        title: str,
        scope: KnowledgeScope,
        metadata: MetadataRecord,
        initial_content_hash: str,
        document_format: DocumentFormat | str = DocumentFormat.DOCX,
        initial_chunks: Iterable[KnowledgeChunk] = (),
        initial_change_reason: str = "Initial registration",
        source_path: str | None = None,
        requires_lvm_interpretation: bool = False,
    ) -> None:
        if not isinstance(item_id, str) or not item_id.strip():
            raise DomainValidationError("KnowledgeItem.item_id cannot be empty.")
        if not isinstance(title, str) or not title.strip():
            raise DomainValidationError("KnowledgeItem.title cannot be empty.")
        if not isinstance(scope, KnowledgeScope):
            raise DomainValidationError("KnowledgeItem.scope must be a KnowledgeScope.")
        if not isinstance(metadata, MetadataRecord):
            raise DomainValidationError("KnowledgeItem.metadata must be a MetadataRecord.")
        if scope.owner_node_id != metadata.node_id:
            raise InvariantViolationError(
                f"KnowledgeScope owner_node_id ({scope.owner_node_id}) must match "
                f"MetadataRecord node_id ({metadata.node_id})."
            )

        try:
            doc_fmt = (
                document_format
                if isinstance(document_format, DocumentFormat)
                else DocumentFormat(str(document_format).upper().lstrip("."))
            )
        except ValueError as exc:
            raise DomainValidationError(f"Unsupported document_format: {document_format}") from exc

        super().__init__(item_id.strip())
        self._item_id = item_id.strip()
        self._title = title.strip()
        self._scope = scope
        self._metadata = metadata
        self._document_format = doc_fmt
        self._source_path = source_path
        self._requires_lvm_interpretation = bool(requires_lvm_interpretation)
        self._lvm_extracted_notes: str | None = None

        active_vtag = metadata.active_version_tag
        assert active_vtag is not None
        initial_version = KnowledgeVersion(
            knowledge_item_id=self._item_id,
            version_tag=active_vtag,
            content_hash=initial_content_hash,
            change_reason=initial_change_reason,
            metadata_snapshot=metadata,
            previous_version_tag=None,
            is_active=True,
            chunks=initial_chunks,
        )
        self._versions: list[KnowledgeVersion] = [initial_version]
        self.record_event(
            KnowledgeVersionCreatedEvent(
                knowledge_item_id=self._item_id,
                version_tag=initial_version.version_tag,
                previous_version_tag=None,
                node_id=self._scope.owner_node_id,
            )
        )

    @property
    def item_id(self) -> str:
        return self._item_id

    @property
    def title(self) -> str:
        return self._title

    @property
    def scope(self) -> KnowledgeScope:
        return self._scope

    @property
    def metadata(self) -> MetadataRecord:
        return self._metadata

    @property
    def document_format(self) -> DocumentFormat:
        return self._document_format

    @property
    def source_path(self) -> str | None:
        return self._source_path

    @property
    def requires_lvm_interpretation(self) -> bool:
        return self._requires_lvm_interpretation

    @property
    def lvm_extracted_notes(self) -> str | None:
        return self._lvm_extracted_notes

    @property
    def versions(self) -> tuple[KnowledgeVersion, ...]:
        """Return all versions (historical and active); old versions are never deleted."""
        return tuple(self._versions)

    @property
    def active_version(self) -> KnowledgeVersion:
        """Return the single active version of this KnowledgeItem."""
        active = [v for v in self._versions if v.is_active]
        if len(active) != 1:
            raise InvariantViolationError(
                f"KnowledgeItem '{self._item_id}' must have exactly 1 active version, found {len(active)}."
            )
        return active[0]

    @property
    def active_version_tag(self) -> VersionTag:
        return self.active_version.version_tag

    @property
    def active_chunks(self) -> tuple[KnowledgeChunk, ...]:
        return self.active_version.chunks

    def get_version(self, version_tag: VersionTag | str) -> KnowledgeVersion | None:
        target_tag = version_tag.tag if isinstance(version_tag, VersionTag) else str(version_tag)
        for v in self._versions:
            if v.version_tag.tag == target_tag:
                return v
        return None

    def add_version(
        self,
        content_hash: str,
        change_reason: str,
        chunks: Iterable[KnowledgeChunk] = (),
        diff_summary: str | None = None,
        custom_tag: str | None = None,
    ) -> KnowledgeVersion:
        """Create a new active version while preserving all previous versions as inactive."""
        current_active = self.active_version
        next_tag = current_active.version_tag.next_version(tag=custom_tag)
        updated_metadata = self._metadata.with_active_version(next_tag)

        new_version = KnowledgeVersion(
            knowledge_item_id=self._item_id,
            version_tag=next_tag,
            content_hash=content_hash,
            change_reason=change_reason,
            metadata_snapshot=updated_metadata,
            previous_version_tag=current_active.version_tag,
            is_active=True,
            chunks=chunks,
            diff_summary=diff_summary,
        )

        for v in self._versions:
            v._deactivate()

        self._versions.append(new_version)
        self._metadata = updated_metadata
        self.record_event(
            KnowledgeVersionCreatedEvent(
                knowledge_item_id=self._item_id,
                version_tag=new_version.version_tag,
                previous_version_tag=current_active.version_tag,
                node_id=self._scope.owner_node_id,
            )
        )
        return new_version

    def attach_lvm_interpretation(self, extracted_notes: str) -> None:
        """Attach layout or visual interpretation produced by the local LVM adapter."""
        if not isinstance(extracted_notes, str) or not extracted_notes.strip():
            raise DomainValidationError("LVM extracted_notes cannot be empty.")
        self._lvm_extracted_notes = extracted_notes.strip()
        self._requires_lvm_interpretation = True
