"""Unit tests for DocumentIngestionService across real DOC, DOCX, and PDF sample documents and LVM hook."""

from __future__ import annotations

from pathlib import Path

import pytest

from offline_rag.application.services import DocumentIngestionService
from offline_rag.domain.knowledge import (
    ChunkingPolicy,
    DocumentFormat,
    KnowledgeChunker,
)
from offline_rag.domain.shared import (
    EntityNotFoundError,
    InvariantViolationError,
    MetadataRecord,
    NodeIdentifier,
)
from offline_rag.infrastructure import (
    DeterministicMockEmbeddingAdapter,
    FixtureConfigLoader,
    InMemoryDocumentStorage,
    InMemoryVectorIndex,
    MockLvmLayoutAdapter,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO_ROOT / "data" / "fixtures"
SAMPLE_DOCS_DIR = REPO_ROOT / "data" / "sample-documents"


def _make_metadata(node_str: str, doc_type: str, confidentiality: str = "internal") -> MetadataRecord:
    return MetadataRecord(
        domain="مدیریت دانش سازمانی",
        node_id=NodeIdentifier(node_str),
        document_type=doc_type,
        confidentiality=confidentiality,
        initial_version_id="v1",
        active_version_id="v1",
        author_or_owner="تیم پروژه",
    )


def test_ingest_real_docx_sample_documents_preserves_headings_and_tables() -> None:
    uow = FixtureConfigLoader.bootstrap_in_memory_uow(FIXTURES_DIR)
    lvm_mock = MockLvmLayoutAdapter()
    storage = InMemoryDocumentStorage()
    vec_index = InMemoryVectorIndex()
    emb_mock = DeterministicMockEmbeddingAdapter(dimension=64)

    service = DocumentIngestionService(
        uow=uow,
        chunker=KnowledgeChunker(ChunkingPolicy(max_chunk_chars=600, chunk_overlap_chars=60)),
        lvm_assistant=lvm_mock,
        document_storage=storage,
        vector_index=vec_index,
        embedding_port=emb_mock,
    )

    docx_files = sorted((SAMPLE_DOCS_DIR / "docx").glob("*.docx"))
    assert len(docx_files) >= 3

    for idx, docx_path in enumerate(docx_files, start=1):
        res = service.ingest_file(
            file_path=docx_path,
            owner_node_id=NodeIdentifier("N-200"),
            metadata=_make_metadata("N-200", "docx_report"),
            document_id=f"SAMPLE-DOCX-{idx:02d}",
        )
        assert res.knowledge_item.document_format == DocumentFormat.DOCX
        assert res.active_version.version_tag.tag == "v1"
        assert len(res.chunks) >= 2
        assert res.storage_uri is not None and res.storage_uri.startswith("memory://")
        assert any(len(c.source_reference.heading_path) >= 1 for c in res.chunks)

    # Specifically verify managerial-report-parent-org-a.docx extracted its table
    mgr_res = service.ingest_file(
        file_path=SAMPLE_DOCS_DIR / "docx" / "managerial-report-parent-org-a.docx",
        owner_node_id=NodeIdentifier("N-100"),
        metadata=_make_metadata("N-100", "managerial_report"),
        document_id="DOCX-MGR-A",
    )
    assert len(mgr_res.extracted_content.tables) >= 1
    assert any("پروژه" in c.text for c in mgr_res.chunks)


def test_ingest_real_doc_sample_documents_extracts_persian_content_and_chunks() -> None:
    uow = FixtureConfigLoader.bootstrap_in_memory_uow(FIXTURES_DIR)
    service = DocumentIngestionService(
        uow=uow,
        chunker=KnowledgeChunker(ChunkingPolicy(max_chunk_chars=500, chunk_overlap_chars=50)),
    )

    doc_files = sorted((SAMPLE_DOCS_DIR / "doc").glob("*.doc"))
    assert len(doc_files) == 2

    for idx, doc_path in enumerate(doc_files, start=1):
        res = service.ingest_file(
            file_path=doc_path,
            owner_node_id=NodeIdentifier("N-300"),
            metadata=_make_metadata("N-300", "technical_doc", confidentiality="restricted"),
            document_id=f"SAMPLE-DOC-{idx:02d}",
        )
        assert res.knowledge_item.document_format == DocumentFormat.DOC
        assert len(res.chunks) >= 1
        assert all(c.source_reference.file_format == DocumentFormat.DOC for c in res.chunks)
        assert all(c.inherited_metadata.node_id == NodeIdentifier("N-300") for c in res.chunks)


def test_ingest_real_pdf_sample_documents_preserves_page_numbers_and_invokes_lvm_hook() -> None:
    uow = FixtureConfigLoader.bootstrap_in_memory_uow(FIXTURES_DIR)
    lvm_mock = MockLvmLayoutAdapter()
    service = DocumentIngestionService(
        uow=uow,
        chunker=KnowledgeChunker(ChunkingPolicy(max_chunk_chars=500, chunk_overlap_chars=50)),
        lvm_assistant=lvm_mock,
    )

    # 1. Multi-page PDF: versioning-and-confidentiality-policy.pdf (2 pages)
    policy_pdf = SAMPLE_DOCS_DIR / "pdf" / "versioning-and-confidentiality-policy.pdf"
    res_policy = service.ingest_file(
        file_path=policy_pdf,
        owner_node_id=NodeIdentifier("N-400"),
        metadata=_make_metadata("N-400", "policy_pdf"),
        document_id="PDF-POLICY-01",
    )
    assert res_policy.knowledge_item.document_format == DocumentFormat.PDF
    assert res_policy.extracted_content.page_count == 2
    chunk_pages = {c.source_reference.page_number for c in res_policy.chunks}
    assert 1 in chunk_pages and 2 in chunk_pages

    # 2. Research notes PDF: research-notes-small-model-team-t.pdf
    notes_pdf = SAMPLE_DOCS_DIR / "pdf" / "research-notes-small-model-team-t.pdf"
    res_notes = service.ingest_file(
        file_path=notes_pdf,
        owner_node_id=NodeIdentifier("N-301"),
        metadata=_make_metadata("N-301", "research_notes", confidentiality="restricted"),
        document_id="PDF-NOTES-01",
    )
    assert res_notes.lvm_assisted is True
    assert res_notes.knowledge_item.requires_lvm_interpretation is True
    assert res_notes.knowledge_item.lvm_extracted_notes is not None
    assert "PDF-NOTES-01" in res_notes.knowledge_item.lvm_extracted_notes
    assert len(lvm_mock.processed_requests) >= 1


def test_scanned_or_low_text_pdf_is_flagged_and_enriched_by_mock_lvm_adapter() -> None:
    uow = FixtureConfigLoader.bootstrap_in_memory_uow(FIXTURES_DIR)
    lvm_mock = MockLvmLayoutAdapter()
    service = DocumentIngestionService(
        uow=uow,
        lvm_assistant=lvm_mock,
    )

    # Minimal valid PDF with an image XObject and zero text (simulating scanned S-09 PDF)
    scanned_pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /Resources << /XObject << /Im1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
        b"4 0 obj << /Type /XObject /Subtype /Image /Width 100 /Height 100 >> endobj\n"
        b"5 0 obj << /Length 12 >>\nstream\nq /Im1 Do Q\nendstream\nendobj\n%%EOF"
    )
    res = service.ingest_bytes(
        raw_bytes=scanned_pdf_bytes,
        file_extension_or_format="pdf",
        document_id="S-09-SCANNED",
        title="سند اسکن‌شده نمونه S-09",
        owner_node_id=NodeIdentifier("N-200"),
        metadata=_make_metadata("N-200", "scanned_pdf"),
    )
    assert res.extracted_content.requires_lvm_assistance is True
    assert res.lvm_assisted is True
    assert len(res.chunks) == 1
    assert "تفسیر بصری LVM" in res.chunks[0].text
    assert res.chunks[0].source_reference.page_number == 1


def test_ingestion_service_rejects_unknown_node_or_confidentiality_violation() -> None:
    uow = FixtureConfigLoader.bootstrap_in_memory_uow(FIXTURES_DIR)
    service = DocumentIngestionService(uow=uow)

    with pytest.raises(EntityNotFoundError):
        service.ingest_bytes(
            raw_bytes=b"sample doc content",
            file_extension_or_format="doc",
            document_id="DOC-ERR",
            title="خطا",
            owner_node_id=NodeIdentifier("N-999"),
            metadata=_make_metadata("N-999", "doc"),
        )

    # N-200 has confidentiality='internal'; ingesting 'confidential' with strict check fails
    with pytest.raises(InvariantViolationError):
        service.ingest_bytes(
            raw_bytes="گزارش محرمانه تست".encode("utf-8"),
            file_extension_or_format="doc",
            document_id="DOC-CONF",
            title="سند محرمانه",
            owner_node_id=NodeIdentifier("N-200"),
            metadata=_make_metadata("N-200", "doc", confidentiality="confidential"),
            enforce_strict_node_confidentiality=True,
        )
