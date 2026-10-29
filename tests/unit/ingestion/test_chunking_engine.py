"""Unit tests for KnowledgeChunker boundary rules, word/sentence integrity, and SourceReference preservation."""

from __future__ import annotations

from offline_rag.domain.knowledge import (
    ChunkingPolicy,
    DocumentFormat,
    ExtractedBlock,
    ExtractedDocumentContent,
    KnowledgeChunker,
)
from offline_rag.domain.shared import (
    MetadataRecord,
    NodeIdentifier,
    VersionTag,
)


def _sample_metadata() -> MetadataRecord:
    return MetadataRecord(
        domain="مهندسی دانش",
        node_id=NodeIdentifier("N-300"),
        document_type="technical_report",
        confidentiality="restricted",
        initial_version_id="v1",
        active_version_id="v1",
    )


def test_chunker_respects_heading_and_page_boundaries_and_preserves_source_reference() -> None:
    blocks = (
        ExtractedBlock(
            text="۱. چکیده پروژه",
            heading_path=("۱. چکیده پروژه",),
            page_number=1,
            section_reference="۱. چکیده پروژه",
            block_type="heading",
            heading_level=1,
        ),
        ExtractedBlock(
            text="این پروژه هسته RAG سلسله‌مراتبی آفلاین را بر پایه طراحی دامنه‌محور پیاده‌سازی می‌کند.",
            heading_path=("۱. چکیده پروژه",),
            page_number=1,
            section_reference="۱. چکیده پروژه",
            block_type="paragraph",
        ),
        ExtractedBlock(
            text="۱.۱ اهداف فرعی",
            heading_path=("۱. چکیده پروژه", "۱.۱ اهداف فرعی"),
            page_number=1,
            section_reference="۱.۱ اهداف فرعی",
            block_type="heading",
            heading_level=2,
        ),
        ExtractedBlock(
            text="پشتیبانی از اسناد DOC، DOCX و PDF در محیط کاملاً آفلاین بدون اینترنت.",
            heading_path=("۱. چکیده پروژه", "۱.۱ اهداف فرعی"),
            page_number=1,
            section_reference="۱.۱ اهداف فرعی",
            block_type="paragraph",
        ),
        ExtractedBlock(
            text="ادامه مستندات فنی در صفحه دوم برای بررسی تفکیک مرز صفحات در موتور خردسازی.",
            heading_path=("۱. چکیده پروژه", "۱.۱ اهداف فرعی"),
            page_number=2,
            section_reference="۱.۱ اهداف فرعی",
            block_type="paragraph",
        ),
    )

    extracted = ExtractedDocumentContent(
        document_id="DOC-CHUNK-TEST",
        file_format=DocumentFormat.PDF,
        blocks=blocks,
        page_count=2,
    )
    policy = ChunkingPolicy(
        max_chunk_chars=300,
        chunk_overlap_chars=30,
        respect_heading_boundaries=True,
        respect_page_boundaries=True,
    )
    chunker = KnowledgeChunker(policy=policy)
    meta = _sample_metadata()
    v1 = VersionTag.initial()

    chunks = chunker.chunk_document(extracted=extracted, version_tag=v1, inherited_metadata=meta)

    # Should produce 3 chunks:
    # Chunk 0: heading ("۱. چکیده پروژه",) on page 1
    # Chunk 1: heading ("۱. چکیده پروژه", "۱.۱ اهداف فرعی") on page 1
    # Chunk 2: heading ("۱. چکیده پروژه", "۱.۱ اهداف فرعی") on page 2
    assert len(chunks) == 3
    assert [c.sequence_index for c in chunks] == [0, 1, 2]

    assert chunks[0].source_reference.document_id == "DOC-CHUNK-TEST"
    assert chunks[0].source_reference.version_id == "v1"
    assert chunks[0].source_reference.file_format == DocumentFormat.PDF
    assert chunks[0].source_reference.page_number == 1
    assert chunks[0].source_reference.heading_path == ("۱. چکیده پروژه",)
    assert chunks[0].inherited_metadata == meta

    assert chunks[1].source_reference.page_number == 1
    assert chunks[1].source_reference.heading_path == ("۱. چکیده پروژه", "۱.۱ اهداف فرعی")

    assert chunks[2].source_reference.page_number == 2
    assert chunks[2].source_reference.heading_path == ("۱. چکیده پروژه", "۱.۱ اهداف فرعی")
    assert chunks[2].context_boundary_end > chunks[2].context_boundary_start


def test_chunker_never_splits_persian_or_english_words_in_middle() -> None:
    sentences = [
        "جمله اول درباره معماری دامنه‌محور و جداسازی لایه‌های کاربردی از زیرساخت است.",
        "Sentence two verifies that English terminology like KnowledgeItem and AggregateRoot stays intact.",
        "جمله سوم بررسی می‌کند که موتور خردسازی هرگز کلمات فارسی یا انگلیسی را از وسط نمی‌شکند.",
        "جمله چهارم برای آزمایش هم‌پوشانی کلمات کامل بین دو قطعه متوالی اضافه شده است.",
    ]
    long_paragraph = " ".join(sentences)
    all_original_words = set(long_paragraph.split())

    extracted = ExtractedDocumentContent(
        document_id="DOC-WORDS",
        file_format=DocumentFormat.DOCX,
        blocks=(
            ExtractedBlock(
                text=long_paragraph,
                heading_path=("بخش تست",),
                page_number=1,
            ),
        ),
    )
    chunker = KnowledgeChunker(
        policy=ChunkingPolicy(
            max_chunk_chars=120,
            chunk_overlap_chars=25,
            respect_heading_boundaries=True,
            respect_page_boundaries=True,
        )
    )
    chunks = chunker.chunk_document(
        extracted=extracted,
        version_tag=VersionTag.initial(),
        inherited_metadata=_sample_metadata(),
    )
    assert len(chunks) >= 3
    for idx, chk in enumerate(chunks):
        assert chk.sequence_index == idx
        assert len(chk.text) <= 120
        # Every token in the chunk must be a complete word from the original text
        for token in chk.text.split():
            assert token in all_original_words, f"Broken word detected: {token}"
