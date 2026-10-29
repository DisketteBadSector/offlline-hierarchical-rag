"""Unit tests for Issue, SubIssue, Executor, IKERelationship (12 types), RFP/Proposal/Feedback, Model/Prompt, and In-Memory Repositories."""

from __future__ import annotations

from pathlib import Path

import pytest

from offline_rag.application.services import KnowledgeApplicationService
from offline_rag.domain.executor import Executor
from offline_rag.domain.ike_model import IKERelationship, IKERelationshipType
from offline_rag.domain.issue import Issue, IssueStatus, SubIssue
from offline_rag.domain.knowledge import KnowledgeChunk, KnowledgeItem
from offline_rag.domain.model_prompt import (
    ModelRole,
    PromptTaskCategory,
    PromptTemplate,
)
from offline_rag.domain.rfp import (
    FeedbackReport,
    Proposal,
    ProposalRecommendation,
    RFP,
)
from offline_rag.domain.shared import (
    InvariantViolationError,
    KnowledgeScope,
    MetadataRecord,
    NodeIdentifier,
    ProgressPercentage,
    RetrievalQuery,
    SimilarityScore,
    VersionTag,
)
from offline_rag.infrastructure import (
    DeterministicMockEmbeddingAdapter,
    DeterministicMockLLMAdapter,
    DeterministicMockLVMAdapter,
    FixtureConfigLoader,
    InMemoryDocumentStorage,
    InMemoryModelRuntime,
    InMemoryUnitOfWork,
    InMemoryVectorIndex,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO_ROOT / "data" / "fixtures"


def test_issue_decomposition_and_deterministic_progress_calculation() -> None:
    node_id = NodeIdentifier("N-300")
    v1 = VersionTag.initial()
    meta = MetadataRecord(
        domain="مهندسی نرم‌افزار",
        node_id=node_id,
        author_or_owner="تیم پروژه پ",
        document_type="issue_record",
        initial_version_tag=v1,
        active_version_tag=v1,
    )
    issue = Issue(
        issue_id="ISS-100",
        node_id=node_id,
        title="بهینه‌سازی بازیابی ترکیبی فارسی",
        description="نیاز به ترکیب جست‌وجوی برداری و فیلتر فراداده در گره N-300.",
        metadata=meta,
        priority="high",
    )
    sub1 = SubIssue(
        sub_issue_id="SUB-101",
        parent_issue_id="ISS-100",
        title="نرمال‌سازی نویسه‌های فارسی",
        description="تبدیل ی و ک عربی به فارسی و حذف نیم‌فاصله اضافی.",
        relative_order=1,
        status=IssueStatus.RESOLVED,
    )
    sub2 = SubIssue(
        sub_issue_id="SUB-102",
        parent_issue_id="ISS-100",
        title="اعمال فیلتر محرمانگی در نمایه برداری",
        description="کنترل سطح دسترسی پیش از بازگرداندن قطعات Top-K.",
        relative_order=2,
        progress=ProgressPercentage(50.0),
    )
    issue.add_sub_issue(sub2)
    issue.add_sub_issue(sub1)

    # Ordered by relative_order
    assert [s.sub_issue_id for s in issue.sub_issues] == ["SUB-101", "SUB-102"]

    # Duplicate relative_order must fail
    with pytest.raises(InvariantViolationError):
        issue.add_sub_issue(
            SubIssue(
                sub_issue_id="SUB-103",
                parent_issue_id="ISS-100",
                title="تکراری",
                description="تست ترتیب تکراری",
                relative_order=1,
            )
        )

    issue.propose_solution("استفاده از BGE-M3 به همراه فیلتر MetadataRecord در لایه بازیابی.")
    progress = issue.recalculate_progress(
        has_supporting_knowledge=True,
        has_executor_mapping=True,
    )
    # 15 (subissues exist) + 0.35 * 75.0 (26.25) + 20 (knowledge) + 15 (executor) + 15 (solution) = 91.25
    assert progress.value == 91.25


def test_executor_and_all_12_ike_relationship_types_and_rfp_lifecycle() -> None:
    uow = InMemoryUnitOfWork()
    node_id = NodeIdentifier("N-300")

    executor = Executor(
        executor_id="EX-01",
        name="دکتر علوی - متخصص بازیابی اطلاعات",
        node_id=node_id,
        organizational_role="پژوهشگر ارشد",
        expertise_domains=("بازیابی اطلاعات", "مدل‌های زبانی کوچک", "طراحی دامنه‌محور"),
        past_project_ids=("PRJ-RAG-01",),
        authored_knowledge_ids=("DOC-001",),
    )
    uow.executors.save(executor)
    assert len(uow.executors.find_by_expertise("مدل های زبانی", node_id=node_id)) == 1

    # Verify all 12 IKERelationshipTypes can be instantiated and stored
    assert len(IKERelationshipType) == 12
    for idx, rel_type in enumerate(IKERelationshipType, start=1):
        rel = IKERelationship(
            relationship_id=f"IKE-{idx:02d}",
            node_id=node_id,
            relationship_type=rel_type,
            source_entity_id="ISS-100",
            target_entity_id="DOC-001",
            confidence=SimilarityScore(0.9),
            created_in_version=VersionTag.initial(),
            evidence_reference=f"DOC-001@v1:c{idx}",
        )
        uow.ike.save(rel)

    assert len(uow.ike.list_by_node(node_id)) == 12
    assert len(uow.ike.list_by_source("ISS-100")) == 12

    # RFP, Proposal, FeedbackReport
    rfp = RFP(
        rfp_id="RFP-2026-01",
        node_id=node_id,
        origin_issue_id="ISS-100",
        title="توسعه ماژول تشخیص هم‌پوشانی معنایی آفلاین",
        problem_statement="نیاز به تشخیص هم‌پوشانی قطعات دانش فارسی در محیط بدون اینترنت.",
        scope="پیاده‌سازی الگوریتم ترکیبی هش و برداری با آزمون واحد.",
        technical_requirements=("پشتیبانی از BGE-M3", "اجرا در کمتر از ۶ گیگابایت VRAM"),
        expected_deliverables=("کد منبع ماژول", "مجموعه آزمون واحد"),
        evaluation_criteria=("دقت تشخیص هم‌پوشانی", "پوشش آزمون واحد"),
        constraints=("ممنوعیت تماس شبکه‌ای",),
        required_expertise=("پردازش زبان طبیعی فارسی", "DDD"),
        research_gaps=("تنظیم آستانه بهینه شباهت برای متون فنی فارسی",),
    )
    uow.rfps.save_rfp(rfp)

    proposal = Proposal(
        proposal_id="PROP-01",
        rfp_id="RFP-2026-01",
        proposer_name="مرکز پژوهشی ب",
        proposed_methodology="استفاده از شباهت کسینوسی قطعات ۵۱۲ توکنی و داور محلی.",
        technical_summary="طراحی تمیز با پورت‌های انتزاعی و Test Double.",
    )
    proposal.record_evaluation(score=88.5)
    uow.rfps.save_proposal(proposal)

    feedback = FeedbackReport(
        feedback_id="FB-01",
        proposal_id="PROP-01",
        rfp_id="RFP-2026-01",
        strengths=("رعایت کامل قید آفلاین", "معماری ماژولار"),
        weaknesses=("عدم اشاره به زمان دقیق بارگذاری ترتیبی"),
        missing_requirements=(),
        technical_risks=("محدودیت حافظه در اسناد بالای ۱۰۰ صفحه"),
        suggested_score=88.5,
        recommendation=ProposalRecommendation.SELECT,
        criterion_references=("RFP-2026-01#criteria-1", "PROP-01#section-2"),
    )
    uow.rfps.save_feedback_report(feedback)

    assert uow.rfps.get_rfp_by_id("RFP-2026-01") == rfp
    assert len(uow.rfps.list_proposals_by_rfp("RFP-2026-01")) == 1
    assert uow.rfps.get_feedback_by_proposal("PROP-01") == feedback


def test_model_and_prompt_registry_and_in_memory_adapters() -> None:
    uow = FixtureConfigLoader.bootstrap_in_memory_uow(FIXTURES_DIR)

    emb_spec = uow.models.get_by_role(ModelRole.EMBEDDING)
    llm_spec = uow.models.get_by_role(ModelRole.LLM)
    lvm_spec = uow.models.get_by_role(ModelRole.LVM)
    assert emb_spec is not None and emb_spec.model_id == "bge-m3"
    assert llm_spec is not None and llm_spec.model_id == "diba-base"
    assert lvm_spec is not None and lvm_spec.model_id == "glm-ocr"
    assert len(PromptTaskCategory) == 12
    for task in PromptTaskCategory:
        assert llm_spec.supports_task(task.value)

    template = PromptTemplate(
        template_id="qa-v1",
        task_category=PromptTaskCategory.QUESTION_ANSWERING,
        version_tag=VersionTag.initial(),
        target_model_id="diba-base",
        system_prompt="تو دستیار پاسخ‌گویی مبتنی بر منبع در گره سازمانی هستی.",
        user_template="زمینه:\n{{context}}\n\nپرسش کاربر در گره {{node_id}}:\n{{query}}",
        expected_output_keys=("answer", "source_references", "uncertainty_declared"),
    )
    uow.prompts.save(template)
    assert template.required_placeholders == ("context", "node_id", "query")

    messages = template.assemble_messages(
        {
            "context": "[DOC-01@v1:c0] متن سند نمونه",
            "node_id": "N-300",
            "query": "هدف پروژه چیست؟",
        }
    )
    assert "N-300" in messages["user"]

    with pytest.raises(InvariantViolationError):
        template.assemble_messages({"context": "ناقص"})

    # Verify KnowledgeApplicationService + InMemoryVectorIndex + Mock Adapters
    vec_index = InMemoryVectorIndex()
    emb_adapter = DeterministicMockEmbeddingAdapter(dimension=64)
    know_service = KnowledgeApplicationService(
        uow=uow, vector_index=vec_index, embedding_port=emb_adapter
    )
    v1 = VersionTag.initial()
    meta = MetadataRecord(
        domain="فنی",
        node_id=NodeIdentifier("N-300"),
        author_or_owner="تیم پ",
        document_type="doc",
        initial_version_tag=v1,
        active_version_tag=v1,
    )
    chunk = KnowledgeChunk(
        chunk_id="C-1",
        source_item_id="K-1",
        version_tag=v1,
        chunk_index=0,
        text="معماری دامنه‌محور هسته RAG آفلاین",
        inherited_metadata=meta,
    )
    item = KnowledgeItem(
        item_id="K-1",
        title="سند ۱",
        scope=KnowledgeScope(owner_node_id=NodeIdentifier("N-300")),
        metadata=meta,
        initial_content_hash="h1",
        initial_chunks=(chunk,),
    )
    know_service.register_knowledge_item(item)

    q = RetrievalQuery(
        raw_text="معماری دامنه‌محور هسته RAG",
        requesting_node_id=NodeIdentifier("N-300"),
    )
    hits = vec_index.search(q, emb_adapter.embed_text(q.normalized_text))
    assert len(hits) == 1
    assert hits[0].chunk.chunk_id == "C-1"

    # Verify DocumentStorage, LLM, LVM, and ModelRuntime test doubles
    doc_store = InMemoryDocumentStorage()
    uri = doc_store.save_document("sample.pdf", b"%PDF-1.4")
    assert uri == "memory://sample.pdf"
    assert doc_store.exists("sample.pdf") is True
    assert doc_store.load_document("sample.pdf") == b"%PDF-1.4"

    llm_mock = DeterministicMockLLMAdapter()
    resp_str = llm_mock.generate(messages["system"], messages["user"])
    assert "source_references" in resp_str
    assert len(llm_mock.call_history) == 1

    lvm_mock = DeterministicMockLVMAdapter()
    layout = lvm_mock.analyze_page_layout("S-09.pdf", 1)
    assert layout["tables_count"] == 1

    runtime = InMemoryModelRuntime(max_concurrent_models=1)
    runtime.load_model("glm-ocr")
    runtime.load_model("bge-m3")
    runtime.load_model("diba-base")
    assert runtime.get_loaded_models() == ("diba-base",)
    assert ("unload", "glm-ocr") in runtime.operations_log
    assert ("unload", "bge-m3") in runtime.operations_log
