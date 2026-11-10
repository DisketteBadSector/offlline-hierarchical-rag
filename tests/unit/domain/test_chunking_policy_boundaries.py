"""Phase 5 supplementary tests for ChunkingPolicy boundary and overlap correctness."""

from __future__ import annotations

from offline_rag.domain.knowledge import (
    ChunkingPolicy,
    DocumentFormat,
    ExtractedBlock,
    ExtractedDocumentContent,
    KnowledgeChunker,
)
from offline_rag.domain.shared import MetadataRecord, NodeIdentifier, VersionTag


def _meta() -> MetadataRecord:
    return MetadataRecord(
        domain="مهندسی",
        node_id=NodeIdentifier("N-300"),
        document_type="report",
        initial_version_id="v1",
        active_version_id="v1",
    )


def _extract(doc_id: str, blocks: tuple[ExtractedBlock, ...]) -> ExtractedDocumentContent:
    return ExtractedDocumentContent(
        document_id=doc_id,
        file_format=DocumentFormat.PDF,
        blocks=blocks,
        page_count=max((b.page_number or 1) for b in blocks) if blocks else 1,
    )


def test_chunk_context_boundary_offsets_are_monotonic() -> None:
    long_text = (
        "جمله اول درباره معماری. جمله دوم درباره خردسازی. "
        "جمله سوم درباره تشخیص تکرار. جمله چهارم درباره انتقال دانش. "
        "جمله پنجم درباره ارزیابی داور. جمله ششم درباره RFP."
    )
    chunker = KnowledgeChunker(
        ChunkingPolicy(max_chunk_chars=70, chunk_overlap_chars=15)
    )
    chunks = chunker.chunk_document(
        extracted=_extract(
            "DOC-OFF",
            (ExtractedBlock(text=long_text, heading_path=("بخش",), page_number=1),),
        ),
        version_tag=VersionTag.initial(),
        inherited_metadata=_meta(),
    )
    assert len(chunks) >= 2
    for chk in chunks:
        assert chk.context_boundary_end > chk.context_boundary_start
        assert chk.context_boundary_end - chk.context_boundary_start == len(chk.text)


def test_chunker_never_exceeds_max_chars_after_overlap_added() -> None:
    text = "کلمه " * 300
    max_chars = 90
    chunker = KnowledgeChunker(
        ChunkingPolicy(max_chunk_chars=max_chars, chunk_overlap_chars=20)
    )
    chunks = chunker.chunk_document(
        extracted=_extract(
            "DOC-LIM",
            (ExtractedBlock(text=text, heading_path=("بخش",), page_number=1),),
        ),
        version_tag=VersionTag.initial(),
        inherited_metadata=_meta(),
    )
    for chk in chunks:
        assert len(chk.text) <= max_chars


def test_heading_change_starts_new_chunk() -> None:
    chunker = KnowledgeChunker(
        ChunkingPolicy(max_chunk_chars=200, chunk_overlap_chars=0)
    )
    chunks = chunker.chunk_document(
        extracted=_extract(
            "DOC-HEAD",
            (
                ExtractedBlock(
                    text="متن زیر تیتر اول.",
                    heading_path=("تیتر اول",),
                    page_number=1,
                    block_type="paragraph",
                ),
                ExtractedBlock(
                    text="متن زیر تیتر دوم.",
                    heading_path=("تیتر دوم",),
                    page_number=1,
                    block_type="paragraph",
                ),
            ),
        ),
        version_tag=VersionTag.initial(),
        inherited_metadata=_meta(),
    )
    assert len(chunks) == 2
    assert chunks[0].source_reference.heading_path == ("تیتر اول",)
    assert chunks[1].source_reference.heading_path == ("تیتر دوم",)


def test_page_change_starts_new_chunk() -> None:
    chunker = KnowledgeChunker(
        ChunkingPolicy(max_chunk_chars=200, chunk_overlap_chars=0)
    )
    chunks = chunker.chunk_document(
        extracted=_extract(
            "DOC-PAGE",
            (
                ExtractedBlock(
                    text="متن صفحه اول.",
                    heading_path=("بخش",),
                    page_number=1,
                    block_type="paragraph",
                ),
                ExtractedBlock(
                    text="متن صفحه دوم.",
                    heading_path=("بخش",),
                    page_number=2,
                    block_type="paragraph",
                ),
            ),
        ),
        version_tag=VersionTag.initial(),
        inherited_metadata=_meta(),
    )
    assert len(chunks) == 2
    assert chunks[0].source_reference.page_number == 1
    assert chunks[1].source_reference.page_number == 2