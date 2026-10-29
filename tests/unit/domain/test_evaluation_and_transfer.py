"""Unit tests for JudgeScore, TransferPackage, and upward delivery fallback."""

from __future__ import annotations

from pathlib import Path

import pytest

from offline_rag.application.services import TransferApplicationService
from offline_rag.domain.evaluation import JudgeRubric, JudgeScore, JudgeThreshold
from offline_rag.domain.shared import (
    InvariantViolationError,
    LayerIndex,
    NodeIdentifier,
    TransferDetailLevel,
    VersionTag,
)
from offline_rag.domain.transfer import TransferPackage, TransferPackageStatus
from offline_rag.infrastructure import (
    FixtureConfigLoader,
    InMemoryCSVTransferExporter,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO_ROOT / "data" / "fixtures"


def _make_criterion_scores(total_19: bool = True) -> dict[str, float]:
    if total_19:
        return {
            "source_faithfulness": 5.0,
            "completeness": 4.0,
            "detail_level_fit": 3.0,
            "uncertainty_handling": 2.0,
            "clarity_and_structure": 2.0,
            "source_citability": 2.0,
            "no_unsupported_claims": 1.0,
        }
    return {
        "source_faithfulness": 3.0,
        "completeness": 2.5,
        "detail_level_fit": 2.0,
        "uncertainty_handling": 2.0,
        "clarity_and_structure": 1.5,
        "source_citability": 1.0,
        "no_unsupported_claims": 0.0,
    }


def test_judge_score_computes_total_and_pass_fail_against_threshold() -> None:
    rubric = JudgeRubric()
    threshold = JudgeThreshold(18.0, 20.0)

    passing_judge = JudgeScore(
        score_id="J-01",
        target_artifact_id="PKG-01",
        criterion_scores=_make_criterion_scores(total_19=True),
        explanation="وفاداری کامل به منبع و پوشش تمام فیلدهای الزامی لایه ۲.",
        rubric=rubric,
        threshold=threshold,
    )
    assert passing_judge.total_score == 19.0
    assert passing_judge.passed is True

    failing_judge = JudgeScore(
        score_id="J-02",
        target_artifact_id="PKG-02",
        criterion_scores=_make_criterion_scores(total_19=False),
        explanation="ادعای بدون پشتوانه و نقص در نتایج پژوهشی.",
        rubric=rubric,
        threshold=threshold,
        missing_information=("research_results",),
        unsupported_claims=("ادعای کاهش ۹۰ درصدی مصرف حافظه بدون مدرک",),
    )
    assert failing_judge.total_score == 12.0
    assert failing_judge.passed is False
    assert len(failing_judge.unsupported_claims) == 1


def test_transfer_package_fallback_to_previous_accepted_when_rejected_by_judge() -> None:
    uow = FixtureConfigLoader.bootstrap_in_memory_uow(FIXTURES_DIR)
    csv_exporter = InMemoryCSVTransferExporter()
    service = TransferApplicationService(uow=uow, csv_exporter=csv_exporter)

    payload_n300 = {
        "methods": "خردسازی معنایی با مدل BGE-M3",
        "research_results": "Recall@5 برابر ۰.۸۴ روی اسناد نمونه",
        "limitations": "سقف ۶ گیگابایت VRAM و اجرای ترتیبی",
        "executor_capabilities": "تیم پروژه پ مسلط به DDD و RAG آفلاین",
        "test_notes": "تمامی آزمون‌های واحد پاس شدند",
    }

    # 1. First cycle package scores 19/20 -> ACCEPTED
    pkg_v1 = service.prepare_and_judge_package(
        package_id="PKG-N300-V1",
        source_node_id=NodeIdentifier("N-300"),
        reporting_phase="phase-11-cycle-1",
        payload_sections=payload_n300,
        source_references=("DOC-300@v1#p1:c0",),
        criterion_scores=_make_criterion_scores(total_19=True),
        judge_explanation="بسته اول کاملاً منطبق با قواعد لایه ۲ است.",
    )
    assert pkg_v1.status == TransferPackageStatus.ACCEPTED
    assert pkg_v1.version_tag == VersionTag(1, "v1")

    # 2. Second cycle package scores 12/20 -> REJECTED_BY_JUDGE
    pkg_v2 = service.prepare_and_judge_package(
        package_id="PKG-N300-V2",
        source_node_id=NodeIdentifier("N-300"),
        reporting_phase="phase-11-cycle-2",
        payload_sections=payload_n300,
        source_references=("DOC-300@v2#p1:c0",),
        criterion_scores=_make_criterion_scores(total_19=False),
        judge_explanation="امتیاز کمتر از حد نصاب ۱۸.",
    )
    assert pkg_v2.status == TransferPackageStatus.REJECTED_BY_JUDGE
    assert pkg_v2.version_tag == VersionTag(2, "v2")
    assert pkg_v2.previous_accepted_package_id == "PKG-N300-V1"

    # 3. Parent node requests latest deliverable package -> receives V1 with fallback warning!
    resolution = service.request_latest_deliverable_package(NodeIdentifier("N-300"))
    assert resolution.used_previous_accepted_fallback is True
    assert resolution.delivered_package.package_id == "PKG-N300-V1"
    assert resolution.rejected_latest_package_id == "PKG-N300-V2"
    assert resolution.warning_message is not None and "PKG-N300-V1" in resolution.warning_message

    # 4. Verify CSV export formatting
    csv_text = service.export_package_csv(resolution.delivered_package)
    assert "PKG-N300-V1" in csv_text
    assert "N-300" in csv_text
    assert "N-200" in csv_text


def test_transfer_package_rejects_missing_required_fields_or_excessive_length() -> None:
    with pytest.raises(InvariantViolationError):
        TransferPackage(
            package_id="PKG-BAD",
            source_node_id=NodeIdentifier("N-300"),
            target_parent_node_id=NodeIdentifier("N-200"),
            target_layer_index=LayerIndex(2),
            detail_level=TransferDetailLevel("technical_summary"),
            version_tag=VersionTag.initial(),
            reporting_phase="phase-1",
            payload_sections={"methods": "فقط روش‌ها"},
            source_references=("DOC-01",),
            required_fields=("methods", "research_results"),
        )
