"""Comprehensive unit tests for all 11 immutable Value Objects."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from offline_rag.domain.shared import (
    InvalidValueObjectError,
    JudgeRubric,
    JudgeThreshold,
    KnowledgeScope,
    LayerIndex,
    MetadataRecord,
    NodeIdentifier,
    ProgressPercentage,
    RetrievalQuery,
    RubricCriterion,
    SimilarityScore,
    TransferDetailLevel,
    VersionTag,
)


def test_node_identifier_valid_and_invalid() -> None:
    node_id = NodeIdentifier(" N-301 ")
    assert str(node_id) == "N-301"
    assert node_id == NodeIdentifier("N-301")

    with pytest.raises(FrozenInstanceError):
        node_id.value = "N-999"  # type: ignore[misc]

    for invalid in ("", "   ", "301", "NODE-100", "N-1"):
        with pytest.raises(InvalidValueObjectError):
            NodeIdentifier(invalid)


def test_layer_index_valid_and_invalid() -> None:
    l0 = LayerIndex(0)
    l3 = LayerIndex(3)
    l4 = LayerIndex(4)
    assert l0.is_strategic_root is True
    assert l4.is_knowledge_source is True
    assert l3.expected_parent_rule_layer() == LayerIndex(2)
    assert int(l3) == 3

    with pytest.raises(InvalidValueObjectError):
        l0.expected_parent_rule_layer()

    for invalid in (-1, 5, 99, True, False, 1.5, "2"):  # type: ignore[list-item]
        with pytest.raises(InvalidValueObjectError):
            LayerIndex(invalid)  # type: ignore[arg-type]


def test_knowledge_scope_valid_and_invalid() -> None:
    scope = KnowledgeScope(
        owner_node_id=NodeIdentifier("N-200"),
        confidentiality="internal",
        usage_domain="پژوهش نرم‌افزار",
    )
    assert scope.is_accessible_at_confidentiality("internal") is True
    assert scope.is_accessible_at_confidentiality("restricted") is True
    assert scope.is_accessible_at_confidentiality("public") is False

    with pytest.raises(InvalidValueObjectError):
        KnowledgeScope(owner_node_id=NodeIdentifier("N-200"), confidentiality="top_secret")

    with pytest.raises(InvalidValueObjectError):
        KnowledgeScope(owner_node_id=NodeIdentifier("N-200"), usage_domain="  ")


def test_version_tag_valid_and_invalid() -> None:
    v1 = VersionTag.initial()
    assert v1.sequence_number == 1
    assert v1.tag == "v1"

    v2 = v1.next_version()
    assert v2.sequence_number == 2
    assert v2.tag == "v2"
    assert v2 > v1

    for invalid_seq in (0, -5, True, "1"):  # type: ignore[list-item]
        with pytest.raises(InvalidValueObjectError):
            VersionTag(sequence_number=invalid_seq)  # type: ignore[arg-type]


def test_progress_percentage_valid_and_invalid() -> None:
    p0 = ProgressPercentage(0)
    p50 = ProgressPercentage(50.556)
    p100 = ProgressPercentage(100.0)
    assert p0.is_not_started is True
    assert p50.value == 50.56
    assert p100.is_completed is True

    for invalid in (-0.1, 100.1, float("nan"), float("inf"), True, "50"):  # type: ignore[list-item]
        with pytest.raises(InvalidValueObjectError):
            ProgressPercentage(invalid)  # type: ignore[arg-type]


def test_judge_rubric_and_threshold_valid_and_invalid() -> None:
    rubric = JudgeRubric()
    assert rubric.max_score == 20.0
    assert len(rubric.criteria) == 7

    scores = {
        "source_faithfulness": 5.0,
        "completeness": 3.5,
        "detail_level_fit": 3.0,
        "uncertainty_handling": 2.5,
        "clarity_and_structure": 2.0,
        "source_citability": 2.0,
        "no_unsupported_claims": 1.0,
    }
    total = rubric.validate_and_sum_scores(scores)
    assert total == 19.0

    threshold = JudgeThreshold()
    assert threshold.min_acceptable_score == 18.0
    assert threshold.is_satisfied_by(total) is True
    assert threshold.is_satisfied_by(17.5) is False

    # Exceeding a criterion max_points raises InvalidValueObjectError
    bad_scores = dict(scores, source_faithfulness=5.5)
    with pytest.raises(InvalidValueObjectError):
        rubric.validate_and_sum_scores(bad_scores)

    # Sum of criteria not matching expected_total_ceiling raises InvalidValueObjectError
    with pytest.raises(InvalidValueObjectError):
        JudgeRubric(
            criteria=(RubricCriterion("c1", "معیار ۱", 10.0),),
            expected_total_ceiling=20.0,
        )

    with pytest.raises(InvalidValueObjectError):
        JudgeThreshold(min_acceptable_score=21.0, max_possible_score=20.0)


def test_retrieval_query_and_similarity_score_valid_and_invalid() -> None:
    sim = SimilarityScore(0.85)
    assert sim.meets_threshold(0.80) is True
    assert sim.meets_threshold(SimilarityScore(0.90)) is False

    with pytest.raises(InvalidValueObjectError):
        SimilarityScore(1.01)
    with pytest.raises(InvalidValueObjectError):
        SimilarityScore(-0.01)

    query = RetrievalQuery(
        raw_text="  بررسي   وضعيت پروژه\u200cها در لایه ۲  ",
        requesting_node_id=NodeIdentifier("N-200"),
        top_k=5,
        min_similarity=sim,
    )
    assert query.normalized_text == "بررسی وضعیت پروژه ها در لایه ۲"

    with pytest.raises(InvalidValueObjectError):
        RetrievalQuery(raw_text="   ", requesting_node_id=NodeIdentifier("N-200"))
    with pytest.raises(InvalidValueObjectError):
        RetrievalQuery(raw_text="پرسش", requesting_node_id=NodeIdentifier("N-200"), top_k=0)


def test_transfer_detail_level_valid_and_invalid() -> None:
    lvl = TransferDetailLevel.for_target_layer(LayerIndex(0))
    assert str(lvl) == "executive_high"
    assert str(TransferDetailLevel.for_target_layer(LayerIndex(3))) == "technical_detailed"

    with pytest.raises(InvalidValueObjectError):
        TransferDetailLevel("unknown_level")
    with pytest.raises(InvalidValueObjectError):
        TransferDetailLevel.for_target_layer(LayerIndex(4))


def test_metadata_record_valid_and_invalid() -> None:
    v1 = VersionTag.initial()
    v2 = v1.next_version()
    meta = MetadataRecord(
        domain="هوش مصنوعی",
        sub_domain="RAG آفلاین",
        node_id=NodeIdentifier("N-300"),
        project_id="PRJ-01",
        author_or_owner="تیم پروژه پ",
        document_type="technical_report",
        confidentiality="restricted",
        initial_version_tag=v1,
        active_version_tag=v1,
        status="active",
        priority="high",
        tags=("rag", "ddd"),
        source_reliability=0.95,
        target_transfer_layer=LayerIndex(2),
        suggested_detail_level=TransferDetailLevel("technical_summary"),
    )
    updated_meta = meta.with_active_version(v2)
    assert updated_meta.active_version_tag == v2
    assert meta.active_version_tag == v1

    with pytest.raises(InvalidValueObjectError):
        MetadataRecord(
            domain="",
            node_id=NodeIdentifier("N-300"),
            author_or_owner="نویسنده",
            document_type="report",
            initial_version_tag=v2,
            active_version_tag=v1,
        )
    with pytest.raises(InvalidValueObjectError):
        MetadataRecord(
            domain="فنی",
            node_id=NodeIdentifier("N-300"),
            author_or_owner="نویسنده",
            document_type="report",
            initial_version_tag=v1,
            active_version_tag=v1,
            source_reliability=1.5,
        )
