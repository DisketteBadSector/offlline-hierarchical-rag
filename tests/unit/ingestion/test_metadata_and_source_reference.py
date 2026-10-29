"""Unit tests for SourceReference and MetadataRecord schema validation and retrieval filtering."""

from __future__ import annotations

import pytest

from offline_rag.domain.knowledge import (
    DocumentFormat,
    KnowledgeChunk,
    SourceReference,
)
from offline_rag.domain.shared import (
    InvalidValueObjectError,
    MetadataRecord,
    NodeIdentifier,
    RetrievalQuery,
    VersionTag,
)
from offline_rag.infrastructure import (
    DeterministicMockEmbeddingAdapter,
    InMemoryVectorIndex,
)


def test_source_reference_valid_and_invalid() -> None:
    ref = SourceReference(
        document_id=" DOC-200-01 ",
        version_id="v1",
        file_format="docx",
        page_number=3,
        section_reference="بخش ۲.۱",
        heading_path=("۱. چکیده پروژه", "اهداف"),
    )
    assert ref.document_id == "DOC-200-01"
    assert ref.version_id == "v1"
    assert ref.file_format == DocumentFormat.DOCX
    assert ref.page_number == 3
    assert ref.section_reference == "بخش ۲.۱"
    assert ref.heading_path == ("۱. چکیده پروژه", "اهداف")
    assert ref.to_citation_string(sequence_index=2) == "DOC-200-01@v1#p3§بخش ۲.۱:c2"

    for bad_kwargs in (
        {"document_id": "", "version_id": "v1", "file_format": "PDF"},
        {"document_id": "D1", "version_id": "  ", "file_format": "PDF"},
        {"document_id": "D1", "version_id": "v1", "file_format": "EXE"},
        {"document_id": "D1", "version_id": "v1", "file_format": "PDF", "page_number": 0},
    ):
        with pytest.raises(InvalidValueObjectError):
            SourceReference(**bad_kwargs)  # type: ignore[arg-type]


def test_metadata_record_mandatory_fields_and_filtering() -> None:
    meta = MetadataRecord(
        domain="مدیریت دانش",
        sub_domain="خردسازی",
        node_id=NodeIdentifier("N-200"),
        document_type="technical_report",
        confidentiality="internal",
        initial_version_id="v1",
        active_version_id="v1",
        project_id="PRJ-RAG",
        tags=("rag", "docx"),
    )
    assert meta.initial_version_id == "v1"
    assert meta.active_version_id == "v1"
    assert meta.initial_version_tag == VersionTag(1, "v1")
    assert meta.active_version_tag == VersionTag(1, "v1")

    assert meta.matches_filters({"domain": "مدیریت دانش", "node_id": "N-200", "tags": "rag"}) is True
    assert meta.matches_filters({"document_type": "policy"}) is False

    # Missing mandatory initial_version_id / active_version_id raises InvalidValueObjectError
    with pytest.raises(InvalidValueObjectError):
        MetadataRecord(
            domain="فنی",
            node_id=NodeIdentifier("N-200"),
            document_type="report",
            confidentiality="internal",
        )

    # Missing domain or document_type or invalid confidentiality raises InvalidValueObjectError
    with pytest.raises(InvalidValueObjectError):
        MetadataRecord(
            domain="   ",
            node_id=NodeIdentifier("N-200"),
            document_type="report",
            confidentiality="internal",
            initial_version_id="v1",
            active_version_id="v1",
        )
    with pytest.raises(InvalidValueObjectError):
        MetadataRecord(
            domain="فنی",
            node_id=NodeIdentifier("N-200"),
            document_type="",
            confidentiality="internal",
            initial_version_id="v1",
            active_version_id="v1",
        )
    with pytest.raises(InvalidValueObjectError):
        MetadataRecord(
            domain="فنی",
            node_id=NodeIdentifier("N-200"),
            document_type="report",
            confidentiality="invalid_level",
            initial_version_id="v1",
            active_version_id="v1",
        )


def test_vector_index_filters_by_metadata_record_fields() -> None:
    vec_index = InMemoryVectorIndex()
    emb = DeterministicMockEmbeddingAdapter(dimension=32)

    meta1 = MetadataRecord(
        domain="هوش مصنوعی",
        node_id=NodeIdentifier("N-200"),
        document_type="research_notes",
        confidentiality="internal",
        initial_version_id="v1",
        active_version_id="v1",
        project_id="PRJ-A",
    )
    meta2 = MetadataRecord(
        domain="هوش مصنوعی",
        node_id=NodeIdentifier("N-200"),
        document_type="policy",
        confidentiality="internal",
        initial_version_id="v1",
        active_version_id="v1",
        project_id="PRJ-B",
    )

    c1 = KnowledgeChunk(
        chunk_id="C1",
        source_item_id="DOC-1",
        version_tag=VersionTag.initial(),
        sequence_index=0,
        text="یادداشت پژوهشی مدل کوچک آفلاین",
        inherited_metadata=meta1,
    )
    c2 = KnowledgeChunk(
        chunk_id="C2",
        source_item_id="DOC-2",
        version_tag=VersionTag.initial(),
        sequence_index=0,
        text="سیاست نسخه‌بندی مدل کوچک آفلاین",
        inherited_metadata=meta2,
    )
    vec_index.index_chunk(c1, emb.embed_text(c1.text))
    vec_index.index_chunk(c2, emb.embed_text(c2.text))

    query = RetrievalQuery(
        raw_text="مدل کوچک آفلاین",
        requesting_node_id=NodeIdentifier("N-200"),
        metadata_filters=(("document_type", "policy"), ("project_id", "PRJ-B")),
    )
    results = vec_index.search(query, emb.embed_text(query.normalized_text))
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "C2"
