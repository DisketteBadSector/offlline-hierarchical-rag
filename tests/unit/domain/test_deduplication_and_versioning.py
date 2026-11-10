"""Unit tests for Phase 5: versioning, exact/near-duplicate, and overlap detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from offline_rag.application.services.knowledge_deduplication_service import (
    DeduplicationPolicy,
    KnowledgeDeduplicationService,
)
from offline_rag.domain.knowledge import (
    KnowledgeChunk,
    KnowledgeItem,
    KnowledgeScope,
    KnowledgeVersion,
)
from offline_rag.domain.knowledge.deduplication import (
    DuplicateDetectionResult,
    DuplicateKind,
    DuplicateMatch,
    ExactDuplicateDetector,
)
from offline_rag.domain.shared import (
    MetadataRecord,
    NodeIdentifier,
    SimilarityScore,
    VersionTag,
)
from offline_rag.infrastructure import (
    DeterministicMockEmbeddingAdapter,
    FixtureConfigLoader,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO_ROOT / "data" / "fixtures"


def _meta(node: str = "N-300", v: str = "v1") -> MetadataRecord:
    return MetadataRecord(
        domain="فنی",
        node_id=NodeIdentifier(node),
        document_type="report",
        initial_version_id=v,
        active_version_id=v,
    )


def _make_chunk(
    chunk_id: str,
    item_id: str,
    text: str,
    version_tag: VersionTag | None = None,
    node: str = "N-300",
) -> KnowledgeChunk:
    v = version_tag or VersionTag.initial()
    return KnowledgeChunk(
        chunk_id=chunk_id,
        source_item_id=item_id,
        version_tag=v,
        sequence_index=0,
        text=text,
        inherited_metadata=_meta(node=node, v=v.tag),
    )


def _require_version(item: KnowledgeItem, tag: str) -> KnowledgeVersion:
    """Type-safe accessor: fail loudly if a version is unexpectedly missing."""
    v = item.get_version(tag)
    assert v is not None, f"version '{tag}' not found on {item.item_id}"
    return v


def test_duplicate_detection_result_rejects_empty_hash() -> None:
    with pytest.raises(Exception):
        DuplicateDetectionResult(query_content_hash="", is_exact_duplicate=False)


def test_duplicate_match_requires_valid_kind_and_similarity() -> None:
    with pytest.raises(Exception):
        DuplicateMatch(
            kind="exact",  # type: ignore[arg-type]
            existing_item_id="DOC-1",
            existing_version_id="v1",
            similarity=SimilarityScore(0.95),
        )

    with pytest.raises(Exception):
        DuplicateMatch(
            kind=DuplicateKind.NEAR,
            existing_item_id="",
            existing_version_id="v1",
            similarity=SimilarityScore(0.95),
        )


def test_exact_duplicate_is_detected_by_content_hash() -> None:
    uow = FixtureConfigLoader.bootstrap_in_memory_uow(FIXTURES_DIR)
    node = NodeIdentifier("N-300")

    existing = KnowledgeItem(
        item_id="DOC-A",
        title="سند الف",
        scope=KnowledgeScope(owner_node_id=node),
        metadata=_meta(),
        initial_content_hash="HASH-SAME",
        initial_chunks=(_make_chunk("C-A1", "DOC-A", "محتوای یکسان برای تست تکرار."),),
    )
    uow.knowledge.save(existing)

    detector = ExactDuplicateDetector(uow.knowledge)
    result = detector.detect(content_hash="HASH-SAME")
    assert result.is_exact_duplicate is True
    assert result.exact_matches == ("DOC-A",)
    assert result.has_any_duplicate is True


def test_exact_duplicate_detector_can_exclude_self_on_update() -> None:
    uow = FixtureConfigLoader.bootstrap_in_memory_uow(FIXTURES_DIR)
    node = NodeIdentifier("N-300")

    item = KnowledgeItem(
        item_id="DOC-SELF",
        title="سند خود",
        scope=KnowledgeScope(owner_node_id=node),
        metadata=_meta(),
        initial_content_hash="HASH-SELF",
        initial_chunks=(_make_chunk("C-S1", "DOC-SELF", "محتوای خودی."),),
    )
    uow.knowledge.save(item)

    detector = ExactDuplicateDetector(uow.knowledge)
    same = detector.detect(content_hash="HASH-SELF", exclude_item_id="DOC-SELF")
    assert same.is_exact_duplicate is False
    assert same.exact_matches == ()


def test_near_duplicate_detected_by_embedding_similarity() -> None:
    uow = FixtureConfigLoader.bootstrap_in_memory_uow(FIXTURES_DIR)
    emb = DeterministicMockEmbeddingAdapter(dimension=64)
    node = NodeIdentifier("N-300")

    existing_chunk = _make_chunk(
        "C-EX",
        "DOC-EX",
        "هسته RAG سلسله‌مراتبی آفلاین برای مدیریت دانش سازمانی و انتقال دانش",
    )
    uow.knowledge.save(
        KnowledgeItem(
            item_id="DOC-EX",
            title="سند موجود",
            scope=KnowledgeScope(owner_node_id=node),
            metadata=_meta(),
            initial_content_hash="H1",
            initial_chunks=(existing_chunk,),
        )
    )

    svc = KnowledgeDeduplicationService(
        uow,
        emb,
        policy=DeduplicationPolicy(
            near_duplicate_threshold=0.60, overlap_threshold=0.30
        ),
    )

    candidate = _make_chunk(
        "C-NEW",
        "DOC-NEW",
        "هسته RAG سلسله‌مراتبی آفلاین برای مدیریت دانش سازمانی و انتقال دانش مدیریتی",
    )
    result = svc.detect_for_chunks(
        node_id=node,
        candidate_chunks=(candidate,),
        content_hash="H-NEW",
    )
    assert result.is_exact_duplicate is False
    assert result.has_any_duplicate is True
    assert result.near_duplicate_matches[0].existing_item_id == "DOC-EX"
    assert result.near_duplicate_matches[0].kind == DuplicateKind.NEAR


def test_overlap_detected_when_below_near_threshold() -> None:
    uow = FixtureConfigLoader.bootstrap_in_memory_uow(FIXTURES_DIR)
    emb = DeterministicMockEmbeddingAdapter(dimension=64)
    node = NodeIdentifier("N-300")

    uow.knowledge.save(
        KnowledgeItem(
            item_id="DOC-A",
            title="سند الف",
            scope=KnowledgeScope(owner_node_id=node),
            metadata=_meta(),
            initial_content_hash="H1",
            initial_chunks=(
                _make_chunk("C-A1", "DOC-A", "معماری دامنه‌محور و مدیریت دانش سازمانی آفلاین"),
            ),
        )
    )

    svc = KnowledgeDeduplicationService(
        uow,
        emb,
        policy=DeduplicationPolicy(
            near_duplicate_threshold=0.99, overlap_threshold=0.20
        ),
    )
    candidate = _make_chunk(
        "C-N1", "DOC-N", "معماری دامنه‌محور و مدیریت دانش سازمانی آفلاین و امنیت"
    )
    result = svc.detect_for_chunks(
        node_id=node, candidate_chunks=(candidate,), content_hash="H-N"
    )
    assert result.has_any_duplicate is True
    assert len(result.overlap_matches) >= 1
    assert result.overlap_matches[0].existing_item_id == "DOC-A"


def test_deduplication_ignores_self_when_updating() -> None:
    uow = FixtureConfigLoader.bootstrap_in_memory_uow(FIXTURES_DIR)
    emb = DeterministicMockEmbeddingAdapter(dimension=64)
    node = NodeIdentifier("N-300")

    uow.knowledge.save(
        KnowledgeItem(
            item_id="DOC-SELF",
            title="سند خود",
            scope=KnowledgeScope(owner_node_id=node),
            metadata=_meta(),
            initial_content_hash="H1",
            initial_chunks=(_make_chunk("C-S1", "DOC-SELF", "متن خودی برای به‌روزرسانی"),),
        )
    )

    svc = KnowledgeDeduplicationService(
        uow, emb, policy=DeduplicationPolicy(near_duplicate_threshold=0.60, overlap_threshold=0.20)
    )
    candidate = _make_chunk(
        "C-S1b", "DOC-SELF", "متن خودی برای به‌روزرسانی", VersionTag(2, "v2")
    )
    result = svc.detect_for_chunks(
        node_id=node,
        candidate_chunks=(candidate,),
        content_hash="H2",
        exclude_item_id="DOC-SELF",
    )
    assert result.is_exact_duplicate is False
    assert all(m.existing_item_id != "DOC-SELF" for m in result.near_duplicate_matches)
    assert all(m.existing_item_id != "DOC-SELF" for m in result.overlap_matches)


def test_add_version_preserves_history_and_switches_active() -> None:
    node = NodeIdentifier("N-300")
    v1 = VersionTag.initial()
    item = KnowledgeItem(
        item_id="DOC-V",
        title="سند نسخه‌دار",
        scope=KnowledgeScope(owner_node_id=node),
        metadata=_meta(),
        initial_content_hash="H1",
        initial_chunks=(
            KnowledgeChunk(
                chunk_id="C-V1",
                source_item_id="DOC-V",
                version_tag=v1,
                sequence_index=0,
                text="متن نسخه اول.",
                inherited_metadata=_meta(v="v1"),
            ),
        ),
    )

    v2 = item.add_version(content_hash="H2", change_reason="به‌روزرسانی محتوا")
    assert v2.version_tag.tag == "v2"
    assert len(item.versions) == 2

    old_v1 = _require_version(item, "v1")
    assert old_v1.is_active is False
    assert old_v1.version_tag.tag == "v1"

    assert item.active_version.version_tag.tag == "v2"
    assert item.metadata.active_version_tag == VersionTag(2, "v2")
    assert item.metadata.initial_version_tag == VersionTag(1, "v1")