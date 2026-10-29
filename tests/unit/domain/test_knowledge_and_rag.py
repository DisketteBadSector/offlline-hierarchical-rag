"""Unit tests for KnowledgeItem versioning, KnowledgeChunk, and RAG Core domain models."""

from __future__ import annotations

import pytest

from offline_rag.domain.knowledge import (
    DocumentFormat,
    KnowledgeChunk,
    KnowledgeItem,
)
from offline_rag.domain.rag import (
    RAGContextPackage,
    RAGResponse,
    RetrievedChunk,
)
from offline_rag.domain.shared import (
    InvariantViolationError,
    KnowledgeScope,
    MetadataRecord,
    NodeIdentifier,
    RetrievalQuery,
    SimilarityScore,
    VersionTag,
)


def _make_sample_metadata(node_str: str = "N-300") -> MetadataRecord:
    v1 = VersionTag.initial()
    return MetadataRecord(
        domain="پژوهش هوش مصنوعی",
        node_id=NodeIdentifier(node_str),
        author_or_owner="تیم پروژه پ",
        document_type="technical_spec",
        initial_version_tag=v1,
        active_version_tag=v1,
        confidentiality="internal",
    )


def test_knowledge_item_versioning_preserves_history_and_switches_active_version() -> None:
    node_id = NodeIdentifier("N-300")
    meta = _make_sample_metadata("N-300")
    scope = KnowledgeScope(owner_node_id=node_id, confidentiality="internal")
    v1 = VersionTag.initial()

    chunk_v1 = KnowledgeChunk(
        chunk_id="CHK-001-v1",
        source_item_id="DOC-001",
        version_tag=v1,
        chunk_index=0,
        text="متن نسخه اول سند معماری.",
        inherited_metadata=meta,
        page_number=1,
    )
    item = KnowledgeItem(
        item_id="DOC-001",
        title="سند معماری هسته RAG",
        scope=scope,
        metadata=meta,
        initial_content_hash="hash_v1_abc",
        document_format=DocumentFormat.PDF,
        initial_chunks=(chunk_v1,),
    )

    assert len(item.versions) == 1
    assert item.active_version.version_tag == v1
    assert item.active_version.is_active is True
    assert item.active_chunks[0].citation_reference == "DOC-001@v1#p1:c0"

    # Add Version 2
    v2_expected = VersionTag(sequence_number=2, tag="v2")
    chunk_v2 = KnowledgeChunk(
        chunk_id="CHK-001-v2",
        source_item_id="DOC-001",
        version_tag=v2_expected,
        chunk_index=0,
        text="متن به‌روزرسانی‌شده در نسخه دوم سند معماری.",
        inherited_metadata=meta.with_active_version(v2_expected),
        page_number=2,
    )
    v2 = item.add_version(
        content_hash="hash_v2_def",
        change_reason="افزودن بخش داوری LLM",
        chunks=(chunk_v2,),
        diff_summary="افزوده شدن فصل ۱۶",
    )

    # Verify history preservation and active pointer switch
    assert len(item.versions) == 2
    assert item.get_version("v1") is not None
    assert item.get_version("v1").is_active is False  # type: ignore[union-attr]
    assert v2.is_active is True
    assert item.active_version == v2
    assert item.active_version.previous_version_tag == v1
    assert item.metadata.active_version_tag == v2_expected
    assert item.metadata.initial_version_tag == v1
    assert len(item.pull_domain_events()) == 2


def test_rag_context_package_enforces_node_boundary_and_formats_citations() -> None:
    meta_n300 = _make_sample_metadata("N-300")
    meta_n200 = _make_sample_metadata("N-200")
    v1 = VersionTag.initial()

    chunk_n300 = KnowledgeChunk(
        chunk_id="CHK-300",
        source_item_id="DOC-300",
        version_tag=v1,
        chunk_index=0,
        text="دانش اختصاصی گره N-300.",
        inherited_metadata=meta_n300,
        page_number=3,
    )
    chunk_n200 = KnowledgeChunk(
        chunk_id="CHK-200",
        source_item_id="DOC-200",
        version_tag=v1,
        chunk_index=0,
        text="دانش گره N-200.",
        inherited_metadata=meta_n200,
    )

    query = RetrievalQuery(raw_text="وضعیت پروژه", requesting_node_id=NodeIdentifier("N-300"))
    ctx = RAGContextPackage(
        query=query,
        retrieved_chunks=(RetrievedChunk(chunk=chunk_n300, similarity=SimilarityScore(0.92)),),
    )
    assert ctx.source_references == ("DOC-300@v1#p3:c0",)
    assert "[DOC-300@v1#p3:c0]" in ctx.format_context_text()

    # Cross-node chunk injection must raise InvariantViolationError
    with pytest.raises(InvariantViolationError):
        RAGContextPackage(
            query=query,
            retrieved_chunks=(RetrievedChunk(chunk=chunk_n200, similarity=SimilarityScore(0.88)),),
        )

    # RAGResponse without citations must declare uncertainty
    with pytest.raises(InvariantViolationError):
        RAGResponse(
            query=query,
            answer_text="پاسخ بدون منبع و بدون اعلام عدم قطعیت",
            source_references=(),
            uncertainty_declared=False,
        )

    valid_uncertain_resp = RAGResponse(
        query=query,
        answer_text="دانش کافی در گره N-300 برای پاسخ یافت نشد.",
        source_references=(),
        uncertainty_declared=True,
        uncertainty_reason="عدم وجود سند مرتبط در مخزن گره",
    )
    assert valid_uncertain_resp.uncertainty_declared is True
