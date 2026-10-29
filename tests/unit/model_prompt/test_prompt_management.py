"""Unit tests for Prompt Decoupling, Registry, Versioning, and PromptAssembler (Section 8.3 & 18.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from offline_rag.domain.model_prompt import (
    CitationPolicy,
    ExpectedOutputFormat,
    PromptAssembler,
    PromptTaskCategory,
    PromptTemplate,
)
from offline_rag.domain.shared.errors import (
    DomainValidationError,
    EntityNotFoundError,
    MissingPromptVariableError,
)
from offline_rag.domain.shared.value_objects import VersionTag
from offline_rag.infrastructure.config import (
    FixtureConfigLoader,
    YamlPromptTemplateRepository,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = REPO_ROOT / "data" / "prompts"
PROMPT_REGISTRY_YAML = PROMPTS_DIR / "prompt_registry.yaml"

ALL_12_TASK_TYPES = (
    "qa",
    "technical_summary",
    "executive_managerial_summary",
    "issue_decomposition",
    "sub_issue_comparison",
    "executor_expertise_analysis",
    "research_gap_identification",
    "knowledge_transfer_prep",
    "rfp_generation",
    "proposal_evaluation",
    "feedback_generation",
    "llm_as_judge",
)


def test_load_all_12_prompt_categories_from_yaml_registry() -> None:
    """Verify all 12 standard prompt categories are loaded from data/prompts/prompt_registry.yaml."""
    assert PROMPT_REGISTRY_YAML.exists()
    repo = YamlPromptTemplateRepository(PROMPTS_DIR)

    active_templates = repo.list_all_active()
    assert len(active_templates) == 12

    loaded_task_types = {tpl.task_type.value for tpl in active_templates}
    assert loaded_task_types == set(ALL_12_TASK_TYPES)

    for task_str in ALL_12_TASK_TYPES:
        tpl = repo.get_latest_by_task(task_str)
        assert tpl is not None
        assert tpl.task_type == PromptTaskCategory(task_str)
        assert str(tpl.version) == "1.0.0"
        assert tpl.prompt_id.startswith("PRM-")
        assert tpl.last_updated != ""
        assert tpl.system_template != ""
        assert tpl.user_template != ""
        assert len(tpl.required_variables) >= 1


def test_retrieve_active_and_historical_versions_without_losing_history() -> None:
    """Verify retrieving active version (1.0.0), historical version (0.9.0), and creating new versions."""
    repo = YamlPromptTemplateRepository(PROMPT_REGISTRY_YAML)

    # 1. Active version is 1.0.0
    active_qa = repo.get_active_by_id("PRM-QA")
    assert active_qa is not None
    assert str(active_qa.version) == "1.0.0"
    assert len(active_qa.history) >= 1
    assert [str(v) for v in active_qa.list_available_versions()] == ["0.9.0", "1.0.0"]

    # 2. Retrieve older historical version 0.9.0 directly from repository
    old_qa = repo.get_by_id_and_version("PRM-QA", "0.9.0")
    assert old_qa is not None
    assert str(old_qa.version) == "0.9.0"
    assert old_qa.prompt_id == "PRM-QA"
    assert old_qa.required_variables == ("context", "query")
    assert old_qa.citation_policy == CitationPolicy.OPTIONAL
    assert old_qa.expected_output_format.required_keys == ("answer", "source_references")

    # Active version has stricter schema and citation policy than 0.9.0
    assert active_qa.required_variables == ("node_scope", "context", "query")
    assert active_qa.citation_policy == CitationPolicy.STRICT_SOURCE_ID
    assert "uncertainty_declared" in active_qa.expected_output_format.required_keys

    # 3. Creating a new version 1.1.0 preserves both 0.9.0 and 1.0.0 in history
    v1_1_qa = active_qa.create_new_version(
        new_version=VersionTag.from_version_id("1.1.0"),
        last_updated="2026-10-30",
        change_note="افزودن کلید confidence_level به خروجی پرسش و پاسخ",
    )
    assert str(v1_1_qa.version) == "1.1.0"
    assert len(v1_1_qa.history) == 2
    assert [str(v) for v in v1_1_qa.list_available_versions()] == ["0.9.0", "1.0.0", "1.1.0"]

    repo.save(v1_1_qa)
    assert str(repo.get_active_by_id("PRM-QA").version) == "1.1.0"  # type: ignore[union-attr]
    assert repo.get_by_id_and_version("PRM-QA", "0.9.0") is not None
    assert repo.get_by_id_and_version("PRM-QA", "1.0.0") is not None


def test_prompt_assembler_variable_validation_and_policy_attachment() -> None:
    """Verify PromptAssembler checks requiredVariables, raises MissingPromptVariableError, and appends instructions."""
    repo = YamlPromptTemplateRepository(PROMPT_REGISTRY_YAML)
    assembler = PromptAssembler(repo)

    # 1. Missing required variables raises MissingPromptVariableError
    with pytest.raises(MissingPromptVariableError) as exc_info:
        assembler.assemble(
            prompt_id="PRM-QA",
            variables={"context": "[DOC-01#p1] متن مستند"},
        )
    assert exc_info.value.prompt_id == "PRM-QA"
    assert set(exc_info.value.missing_variables) == {"node_scope", "query"}

    # Empty/whitespace variable value also triggers MissingPromptVariableError
    with pytest.raises(MissingPromptVariableError):
        assembler.assemble(
            prompt_id="PRM-QA",
            variables={"node_scope": "N-300", "context": "متن", "query": "   "},
        )

    # 2. Successful assembly of active version (1.0.0)
    assembled = assembler.assemble(
        prompt_id="PRM-QA",
        variables={
            "node_scope": "N-301",
            "context": "[DOC-301#sec2] مدل BGE-M3 برای بازیابی چندزبانه انتخاب شد.",
            "query": "کدام مدل امبدینگ در گره N-301 انتخاب شده است؟",
        },
    )
    assert assembled.prompt_id == "PRM-QA"
    assert str(assembled.version) == "1.0.0"
    assert "N-301" in assembled.user_message
    assert "[DOC-301#sec2]" in assembled.user_message
    # Automatic attachment of citation policy and expected output format
    assert "strict_source_id" in assembled.user_message
    assert "answer, source_references, uncertainty_declared" in assembled.user_message
    assert len(assembled.to_messages()) == 2

    # 3. Assembling older version (0.9.0) explicitly
    assembled_old = assembler.assemble(
        prompt_id="PRM-QA",
        version="0.9.0",
        variables={
            "context": "[DOC-301#sec2] متن نمونه",
            "query": "پرسش نمونه",
        },
    )
    assert str(assembled_old.version) == "0.9.0"
    assert "optional" in assembled_old.user_message

    # 4. Unknown prompt or version raises EntityNotFoundError
    with pytest.raises(EntityNotFoundError):
        assembler.assemble(prompt_id="PRM-QA", version="9.9.9", variables={})


def test_judge_rubric_id_and_expected_output_format_linkage() -> None:
    """Validate judgeRubricId, citationPolicy, and expectedOutputFormat across all 12 prompts."""
    repo = YamlPromptTemplateRepository(PROMPT_REGISTRY_YAML)
    active_templates = repo.list_all_active()

    for tpl in active_templates:
        assert tpl.judge_rubric_id.startswith("RUBRIC-")
        assert tpl.judge_rubric_id.endswith("-20")
        assert tpl.expected_output_format.format in ("json", "markdown")
        assert len(tpl.expected_output_format.required_keys) >= 3
        assert isinstance(tpl.citation_policy, CitationPolicy)
        # All historical entries also preserve valid judgeRubricId and expectedOutputFormat
        for hist in tpl.history:
            assert hist.judge_rubric_id.startswith("RUBRIC-")
            assert hist.expected_output_format.format in ("json", "markdown")
            assert len(hist.expected_output_format.required_keys) >= 1

    # Verify invalid ExpectedOutputFormat is rejected
    with pytest.raises(DomainValidationError):
        ExpectedOutputFormat(format="xml", required_keys=("a",))


def test_bootstrap_uow_loads_prompt_registry_and_test_llm_uses_assembler() -> None:
    """Verify FixtureConfigLoader bootstraps prompts into UnitOfWork and test_llm uses PromptAssembler."""
    uow = FixtureConfigLoader.bootstrap_in_memory_uow(REPO_ROOT / "data" / "fixtures")
    qa_tpl = uow.prompts.get_active_by_id("PRM-QA")
    assert qa_tpl is not None
    assert str(qa_tpl.version) == "1.0.0"

    judge_tpl = uow.prompts.get_latest_by_task(PromptTaskCategory.LLM_AS_JUDGE)
    assert judge_tpl is not None
    assert judge_tpl.prompt_id == "PRM-JUDGE"
