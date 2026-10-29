"""Chunking Engine and Extracted Document Content Models (Section 7.3 & 7.4)."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

from offline_rag.domain.knowledge.entities import (
    DocumentFormat,
    KnowledgeChunk,
    SourceReference,
)
from offline_rag.domain.shared.errors import DomainValidationError, InvalidValueObjectError
from offline_rag.domain.shared.value_objects import MetadataRecord, VersionTag

# Sentence boundary regex supporting Persian and English punctuation
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!؟?؛;\n])\s+")


@dataclass(frozen=True)
class ExtractedTable:
    """Structured table extracted from a DOC, DOCX, or PDF document."""

    rows: tuple[tuple[str, ...], ...]
    page_number: int | None = None
    heading_path: tuple[str, ...] = ()
    caption: str | None = None

    def __post_init__(self) -> None:
        normalized_rows = tuple(
            tuple(str(cell).strip() for cell in row)
            for row in self.rows
            if any(str(c).strip() for c in row)
        )
        headings = tuple(h.strip() for h in self.heading_path if isinstance(h, str) and h.strip())
        object.__setattr__(self, "rows", normalized_rows)
        object.__setattr__(self, "heading_path", headings)

    def to_text(self) -> str:
        """Render table rows in pipe-delimited text suitable for chunking."""
        lines: list[str] = []
        if self.caption:
            lines.append(f"[جدول: {self.caption}]")
        for row in self.rows:
            lines.append(" | ".join(row))
        return "\n".join(lines)


@dataclass(frozen=True)
class ExtractedBlock:
    """Single structural text block extracted from a document."""

    text: str
    heading_path: tuple[str, ...] = ()
    page_number: int | None = None
    section_reference: str | None = None
    block_type: str = "paragraph"
    heading_level: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise DomainValidationError("ExtractedBlock.text cannot be empty.")
        if self.page_number is not None and (
            isinstance(self.page_number, bool)
            or not isinstance(self.page_number, int)
            or self.page_number < 1
        ):
            raise DomainValidationError("ExtractedBlock.page_number must be an integer >= 1.")
        headings = tuple(h.strip() for h in self.heading_path if isinstance(h, str) and h.strip())
        sec_ref = (
            self.section_reference.strip()
            if isinstance(self.section_reference, str) and self.section_reference.strip()
            else (headings[-1] if headings else None)
        )
        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(self, "heading_path", headings)
        object.__setattr__(self, "section_reference", sec_ref)


@dataclass(frozen=True)
class ExtractedDocumentContent:
    """Structured output returned by DocumentExtractorPort adapters."""

    document_id: str
    file_format: DocumentFormat
    blocks: tuple[ExtractedBlock, ...]
    tables: tuple[ExtractedTable, ...] = ()
    page_count: int = 1
    requires_lvm_assistance: bool = False
    flagged_lvm_pages: tuple[int, ...] = ()
    lvm_annotations: tuple[str, ...] = ()
    raw_content_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise DomainValidationError("ExtractedDocumentContent.document_id cannot be empty.")
        fmt = (
            self.file_format
            if isinstance(self.file_format, DocumentFormat)
            else DocumentFormat(str(self.file_format).strip().upper().lstrip("."))
        )
        object.__setattr__(self, "document_id", self.document_id.strip())
        object.__setattr__(self, "file_format", fmt)
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "tables", tuple(self.tables))
        object.__setattr__(self, "flagged_lvm_pages", tuple(self.flagged_lvm_pages))
        object.__setattr__(self, "lvm_annotations", tuple(self.lvm_annotations))

    @property
    def full_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)

    def with_lvm_enrichment(
        self,
        lvm_blocks: Sequence[ExtractedBlock],
        lvm_notes: Sequence[str],
    ) -> ExtractedDocumentContent:
        """Return a new ExtractedDocumentContent enriched with LVM layout descriptions."""
        return replace(
            self,
            blocks=tuple([*self.blocks, *lvm_blocks]),
            lvm_annotations=tuple([*self.lvm_annotations, *lvm_notes]),
        )


@dataclass(frozen=True)
class ChunkingPolicy:
    """Configuration policy for KnowledgeChunker."""

    max_chunk_chars: int = 800
    chunk_overlap_chars: int = 100
    respect_heading_boundaries: bool = True
    respect_page_boundaries: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_chunk_chars, bool)
            or not isinstance(self.max_chunk_chars, int)
            or self.max_chunk_chars < 40
        ):
            raise InvalidValueObjectError("ChunkingPolicy.max_chunk_chars must be an integer >= 40.")
        if (
            isinstance(self.chunk_overlap_chars, bool)
            or not isinstance(self.chunk_overlap_chars, int)
            or self.chunk_overlap_chars < 0
            or self.chunk_overlap_chars >= self.max_chunk_chars
        ):
            raise InvalidValueObjectError(
                "ChunkingPolicy.chunk_overlap_chars must be >= 0 and < max_chunk_chars."
            )


class KnowledgeChunker:
    """Domain service splitting ExtractedDocumentContent into fine-grained KnowledgeChunks."""

    def __init__(self, policy: ChunkingPolicy | None = None) -> None:
        self._policy = policy or ChunkingPolicy()

    @property
    def policy(self) -> ChunkingPolicy:
        return self._policy

    @staticmethod
    def _split_preserving_sentences_and_words(text: str, max_chars: int) -> list[str]:
        """Split text into pieces <= max_chars without breaking sentences or words in the middle."""
        cleaned = text.strip()
        if not cleaned:
            return []
        if len(cleaned) <= max_chars:
            return [cleaned]

        raw_sentences = [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(cleaned) if s.strip()]
        units: list[str] = []
        for sent in raw_sentences:
            if len(sent) <= max_chars:
                units.append(sent)
            else:
                # Sentence exceeds max_chars -> split along word boundaries, never mid-word
                words = sent.split()
                current_words: list[str] = []
                current_len = 0
                for word in words:
                    add_len = len(word) + (1 if current_words else 0)
                    if current_words and current_len + add_len > max_chars:
                        units.append(" ".join(current_words))
                        current_words = [word]
                        current_len = len(word)
                    else:
                        current_words.append(word)
                        current_len += add_len
                if current_words:
                    units.append(" ".join(current_words))

        pieces: list[str] = []
        buf: list[str] = []
        buf_len = 0
        for unit in units:
            sep_len = 1 if buf else 0
            if buf and buf_len + sep_len + len(unit) > max_chars:
                pieces.append(" ".join(buf))
                buf = [unit]
                buf_len = len(unit)
            else:
                buf.append(unit)
                buf_len += sep_len + len(unit)
        if buf:
            pieces.append(" ".join(buf))
        return pieces

    @staticmethod
    def _compute_word_safe_overlap(previous_text: str, overlap_chars: int) -> str:
        """Extract trailing words from previous_text up to overlap_chars without slicing any word."""
        if overlap_chars <= 0 or not previous_text:
            return ""
        words = previous_text.split()
        collected: list[str] = []
        total = 0
        for word in reversed(words):
            add_len = len(word) + (1 if collected else 0)
            if total + add_len > overlap_chars:
                break
            collected.append(word)
            total += add_len
        return " ".join(reversed(collected))

    def chunk_document(
        self,
        extracted: ExtractedDocumentContent,
        version_tag: VersionTag,
        inherited_metadata: MetadataRecord,
    ) -> tuple[KnowledgeChunk, ...]:
        """Chunk an ExtractedDocumentContent according to policy while preserving metadata and SourceReference."""
        if not extracted.blocks:
            return ()

        max_chars = self._policy.max_chunk_chars
        overlap_chars = self._policy.chunk_overlap_chars

        # Group contiguous blocks that share compatible boundary attributes
        grouped_segments: list[
            tuple[tuple[str, ...], int | None, str | None, list[str]]
        ] = []

        for block in extracted.blocks:
            h_path = block.heading_path
            p_num = block.page_number
            sec_ref = block.section_reference

            if not grouped_segments:
                grouped_segments.append((h_path, p_num, sec_ref, [block.text]))
                continue

            prev_h, prev_p, prev_sec, prev_texts = grouped_segments[-1]
            heading_changed = self._policy.respect_heading_boundaries and (h_path != prev_h)
            page_changed = self._policy.respect_page_boundaries and (p_num != prev_p)

            if heading_changed or page_changed:
                grouped_segments.append((h_path, p_num, sec_ref or (h_path[-1] if h_path else prev_sec), [block.text]))
            else:
                prev_texts.append(block.text)

        chunks: list[KnowledgeChunk] = []
        cursor_offset = 0
        seq_idx = 0

        for h_path, p_num, sec_ref, block_texts in grouped_segments:
            segment_text = "\n".join(block_texts).strip()
            if not segment_text:
                continue

            pieces = self._split_preserving_sentences_and_words(segment_text, max_chars)
            prev_piece_text = ""

            for piece in pieces:
                overlap_prefix = ""
                if prev_piece_text and overlap_chars > 0:
                    candidate_overlap = self._compute_word_safe_overlap(
                        prev_piece_text, overlap_chars
                    )
                    if candidate_overlap and len(candidate_overlap) + 1 + len(piece) <= max_chars:
                        overlap_prefix = candidate_overlap

                final_text = f"{overlap_prefix} {piece}".strip() if overlap_prefix else piece
                start_offset = max(0, cursor_offset - (len(overlap_prefix) + 1 if overlap_prefix else 0))
                end_offset = start_offset + len(final_text)
                cursor_offset += len(piece) + 1

                src_ref = SourceReference(
                    document_id=extracted.document_id,
                    version_id=version_tag.tag,
                    file_format=extracted.file_format,
                    page_number=p_num,
                    section_reference=sec_ref,
                    heading_path=h_path,
                )
                chunk_id = f"{extracted.document_id}:{version_tag.tag}:chk-{seq_idx:04d}"
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=chunk_id,
                        source_item_id=extracted.document_id,
                        version_tag=version_tag,
                        sequence_index=seq_idx,
                        text=final_text,
                        inherited_metadata=inherited_metadata,
                        source_reference=src_ref,
                        context_boundary_start=start_offset,
                        context_boundary_end=end_offset,
                    )
                )
                prev_piece_text = piece
                seq_idx += 1

        return tuple(chunks)


__all__ = [
    "ChunkingPolicy",
    "ExtractedBlock",
    "ExtractedDocumentContent",
    "ExtractedTable",
    "KnowledgeChunker",
]
