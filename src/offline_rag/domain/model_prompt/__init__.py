"""Model and Prompt Management Bounded Context: OfflineModelSpec, PromptTemplate, and Repositories (Section 8.2 & 8.3)."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from offline_rag.domain.shared.base import AggregateRoot
from offline_rag.domain.shared.errors import DomainValidationError, InvariantViolationError
from offline_rag.domain.shared.value_objects import VersionTag

PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


class ModelRole(str, Enum):
    """Role of a local offline AI model."""

    EMBEDDING = "embedding"
    LLM = "llm"
    LVM = "lvm"


class PromptTaskCategory(str, Enum):
    """The 12 standard prompt task categories defined in Section 8.3 of the proposal."""

    QUESTION_ANSWERING = "question_answering"
    TECHNICAL_SUMMARIZATION = "technical_summarization"
    MANAGERIAL_EXECUTIVE_SUMMARIZATION = "managerial_executive_summarization"
    ISSUE_DECOMPOSITION = "issue_decomposition"
    SUBISSUE_COMPARISON = "subissue_comparison"
    EXECUTOR_EXPERTISE_ANALYSIS = "executor_expertise_analysis"
    RESEARCH_GAP_IDENTIFICATION = "research_gap_identification"
    KNOWLEDGE_TRANSFER_PREPARATION = "knowledge_transfer_preparation"
    RFP_GENERATION = "rfp_generation"
    PROPOSAL_EVALUATION = "proposal_evaluation"
    FEEDBACK_GENERATION = "feedback_generation"
    LLM_AS_JUDGE_SCORING = "llm_as_judge_scoring"


@dataclass(frozen=True)
class HardwareConstraints:
    """Hardware profile constraints for a local offline model."""

    min_vram_gb: float
    min_ram_gb: float
    preferred_device: str = "gpu"
    cpu_fallback_supported: bool = True
    execution_mode: str = "sequential_load_unload"
    extra_constraints: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if self.min_vram_gb < 0.0 or self.min_ram_gb <= 0.0:
            raise DomainValidationError("HardwareConstraints VRAM/RAM values must be valid.")
        if self.preferred_device not in ("gpu", "cpu"):
            raise DomainValidationError("HardwareConstraints.preferred_device must be 'gpu' or 'cpu'.")
        object.__setattr__(self, "extra_constraints", MappingProxyType(dict(self.extra_constraints)))


class OfflineModelSpec(AggregateRoot):
    """Registered local offline model specification corresponding to data/fixtures/models.yaml."""

    def __init__(
        self,
        model_id: str,
        role: ModelRole | str,
        provider_endpoint: str,
        context_window: int,
        max_output_tokens: int,
        temperature_default: float,
        supported_tasks: Iterable[str],
        hardware_constraints: HardwareConstraints,
        source_name: str = "",
        weights_format: str = "",
        local_weights_path: str = "",
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise DomainValidationError("OfflineModelSpec.model_id cannot be empty.")
        try:
            m_role = role if isinstance(role, ModelRole) else ModelRole(str(role).lower())
        except ValueError as exc:
            raise DomainValidationError(f"Invalid ModelRole: {role}") from exc

        if not isinstance(provider_endpoint, str) or not provider_endpoint.strip():
            raise DomainValidationError("OfflineModelSpec.provider_endpoint cannot be empty.")
        if isinstance(context_window, bool) or not isinstance(context_window, int) or context_window <= 0:
            raise DomainValidationError("OfflineModelSpec.context_window must be > 0.")
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
        ):
            raise DomainValidationError("OfflineModelSpec.max_output_tokens must be > 0.")
        if temperature_default < 0.0 or temperature_default > 2.0:
            raise DomainValidationError("OfflineModelSpec.temperature_default must be in [0.0, 2.0].")

        tasks = tuple(t.strip() for t in supported_tasks if isinstance(t, str) and t.strip())
        if not tasks:
            raise DomainValidationError("OfflineModelSpec.supported_tasks cannot be empty.")
        if not isinstance(hardware_constraints, HardwareConstraints):
            raise DomainValidationError(
                "OfflineModelSpec.hardware_constraints must be a HardwareConstraints instance."
            )

        super().__init__(model_id.strip())
        self._model_id = model_id.strip()
        self._role = m_role
        self._provider_endpoint = provider_endpoint.strip()
        self._context_window = context_window
        self._max_output_tokens = max_output_tokens
        self._temperature_default = float(temperature_default)
        self._supported_tasks = tasks
        self._hardware_constraints = hardware_constraints
        self._source_name = source_name.strip()
        self._weights_format = weights_format.strip()
        self._local_weights_path = local_weights_path.strip()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def role(self) -> ModelRole:
        return self._role

    @property
    def provider_endpoint(self) -> str:
        return self._provider_endpoint

    @property
    def context_window(self) -> int:
        return self._context_window

    @property
    def max_output_tokens(self) -> int:
        return self._max_output_tokens

    @property
    def temperature_default(self) -> float:
        return self._temperature_default

    @property
    def supported_tasks(self) -> tuple[str, ...]:
        return self._supported_tasks

    @property
    def hardware_constraints(self) -> HardwareConstraints:
        return self._hardware_constraints

    @property
    def source_name(self) -> str:
        return self._source_name

    @property
    def weights_format(self) -> str:
        return self._weights_format

    @property
    def local_weights_path(self) -> str:
        return self._local_weights_path

    def supports_task(self, task_name: str) -> bool:
        return task_name in self._supported_tasks


class PromptTemplate(AggregateRoot):
    """Versioned prompt template bound to a PromptTaskCategory and target model (Section 8.3)."""

    def __init__(
        self,
        template_id: str,
        task_category: PromptTaskCategory | str,
        version_tag: VersionTag,
        target_model_id: str,
        system_prompt: str,
        user_template: str,
        expected_output_keys: Iterable[str] = (),
        source_citation_required: bool = True,
        uncertainty_declaration_required: bool = True,
    ) -> None:
        if not isinstance(template_id, str) or not template_id.strip():
            raise DomainValidationError("PromptTemplate.template_id cannot be empty.")
        try:
            cat = (
                task_category
                if isinstance(task_category, PromptTaskCategory)
                else PromptTaskCategory(str(task_category))
            )
        except ValueError as exc:
            raise DomainValidationError(f"Invalid PromptTaskCategory: {task_category}") from exc

        if not isinstance(version_tag, VersionTag):
            raise DomainValidationError("PromptTemplate.version_tag must be a VersionTag.")
        if not isinstance(target_model_id, str) or not target_model_id.strip():
            raise DomainValidationError("PromptTemplate.target_model_id cannot be empty.")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise DomainValidationError("PromptTemplate.system_prompt cannot be empty.")
        if not isinstance(user_template, str) or not user_template.strip():
            raise DomainValidationError("PromptTemplate.user_template cannot be empty.")

        composite_id = f"{template_id.strip()}:{version_tag.tag}"
        super().__init__(composite_id)
        self._template_id = template_id.strip()
        self._task_category = cat
        self._version_tag = version_tag
        self._target_model_id = target_model_id.strip()
        self._system_prompt = system_prompt.strip()
        self._user_template = user_template.strip()
        self._expected_output_keys = tuple(
            k.strip() for k in expected_output_keys if isinstance(k, str) and k.strip()
        )
        self._source_citation_required = bool(source_citation_required)
        self._uncertainty_declaration_required = bool(uncertainty_declaration_required)

    @property
    def template_id(self) -> str:
        return self._template_id

    @property
    def task_category(self) -> PromptTaskCategory:
        return self._task_category

    @property
    def version_tag(self) -> VersionTag:
        return self._version_tag

    @property
    def target_model_id(self) -> str:
        return self._target_model_id

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def user_template(self) -> str:
        return self._user_template

    @property
    def expected_output_keys(self) -> tuple[str, ...]:
        return self._expected_output_keys

    @property
    def source_citation_required(self) -> bool:
        return self._source_citation_required

    @property
    def uncertainty_declaration_required(self) -> bool:
        return self._uncertainty_declaration_required

    @property
    def required_placeholders(self) -> tuple[str, ...]:
        found = PLACEHOLDER_PATTERN.findall(self._user_template)
        return tuple(dict.fromkeys(found))

    def assemble_messages(self, variables: Mapping[str, str]) -> dict[str, str]:
        """Render the prompt template with required variables and return {'system': ..., 'user': ...}."""
        missing = [p for p in self.required_placeholders if p not in variables]
        if missing:
            raise InvariantViolationError(
                f"Missing required prompt placeholders for '{self._template_id}': {missing}"
            )
        rendered_user = self._user_template
        for key, val in variables.items():
            rendered_user = re.sub(r"\{\{\s*" + re.escape(key) + r"\s*\}\}", str(val), rendered_user)
        return {"system": self._system_prompt, "user": rendered_user}


class ModelRegistryRepository(ABC):
    """Abstract repository for registered OfflineModelSpec entries."""

    @abstractmethod
    def save(self, model_spec: OfflineModelSpec) -> None:
        """Persist or update an OfflineModelSpec."""

    @abstractmethod
    def get_by_id(self, model_id: str) -> OfflineModelSpec | None:
        """Retrieve a model specification by modelId."""

    @abstractmethod
    def get_by_role(self, role: ModelRole | str) -> OfflineModelSpec | None:
        """Retrieve the selected model specification for a given role ('embedding', 'llm', 'lvm')."""

    @abstractmethod
    def list_all(self) -> Sequence[OfflineModelSpec]:
        """Return all registered OfflineModelSpec instances."""


class PromptTemplateRepository(ABC):
    """Abstract repository for versioned PromptTemplate aggregates."""

    @abstractmethod
    def save(self, template: PromptTemplate) -> None:
        """Persist a PromptTemplate version."""

    @abstractmethod
    def get_by_id_and_version(
        self, template_id: str, version_tag: VersionTag | str
    ) -> PromptTemplate | None:
        """Retrieve a specific version of a PromptTemplate."""

    @abstractmethod
    def get_latest_by_task(self, task_category: PromptTaskCategory | str) -> PromptTemplate | None:
        """Retrieve the latest versioned PromptTemplate for a task category."""

    @abstractmethod
    def list_by_task(self, task_category: PromptTaskCategory | str) -> Sequence[PromptTemplate]:
        """List all versions of PromptTemplates registered for a task category."""


__all__ = [
    "HardwareConstraints",
    "ModelRegistryRepository",
    "ModelRole",
    "OfflineModelSpec",
    "PromptTaskCategory",
    "PromptTemplate",
    "PromptTemplateRepository",
]
