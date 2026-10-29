"""Model and Prompt Management Bounded Context: OfflineModelSpec, PromptTemplate, PromptAssembler, and Repositories (Section 8.2 & 8.3)."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from offline_rag.domain.shared.base import AggregateRoot
from offline_rag.domain.shared.errors import (
    DomainValidationError,
    EntityNotFoundError,
    MissingPromptVariableError,
)
from offline_rag.domain.shared.value_objects import VersionTag

PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


class ModelRole(str, Enum):
    """Role of a local offline AI model."""

    EMBEDDING = "embedding"
    LLM = "llm"
    LVM = "lvm"


class PromptTaskCategory(str, Enum):
    """The 12 standard prompt task categories defined in Section 8.3 of the proposal."""

    QA = "qa"
    TECHNICAL_SUMMARY = "technical_summary"
    EXECUTIVE_MANAGERIAL_SUMMARY = "executive_managerial_summary"
    ISSUE_DECOMPOSITION = "issue_decomposition"
    SUB_ISSUE_COMPARISON = "sub_issue_comparison"
    EXECUTOR_EXPERTISE_ANALYSIS = "executor_expertise_analysis"
    RESEARCH_GAP_IDENTIFICATION = "research_gap_identification"
    KNOWLEDGE_TRANSFER_PREP = "knowledge_transfer_prep"
    RFP_GENERATION = "rfp_generation"
    PROPOSAL_EVALUATION = "proposal_evaluation"
    FEEDBACK_GENERATION = "feedback_generation"
    LLM_AS_JUDGE = "llm_as_judge"

    # Backward-compatible enum member aliases
    QUESTION_ANSWERING = "qa"
    TECHNICAL_SUMMARIZATION = "technical_summary"
    MANAGERIAL_EXECUTIVE_SUMMARIZATION = "executive_managerial_summary"
    SUBISSUE_COMPARISON = "sub_issue_comparison"
    KNOWLEDGE_TRANSFER_PREPARATION = "knowledge_transfer_prep"
    LLM_AS_JUDGE_SCORING = "llm_as_judge"

    @classmethod
    def _missing_(cls, value: object) -> PromptTaskCategory | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower()
        legacy_map = {
            "question_answering": cls.QA,
            "technical_summarization": cls.TECHNICAL_SUMMARY,
            "managerial_executive_summarization": cls.EXECUTIVE_MANAGERIAL_SUMMARY,
            "subissue_comparison": cls.SUB_ISSUE_COMPARISON,
            "knowledge_transfer_preparation": cls.KNOWLEDGE_TRANSFER_PREP,
            "llm_as_judge_scoring": cls.LLM_AS_JUDGE,
        }
        return legacy_map.get(normalized)


TASK_LEGACY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "qa": ("qa", "question_answering"),
    "technical_summary": ("technical_summary", "technical_summarization"),
    "executive_managerial_summary": (
        "executive_managerial_summary",
        "managerial_executive_summarization",
    ),
    "issue_decomposition": ("issue_decomposition",),
    "sub_issue_comparison": ("sub_issue_comparison", "subissue_comparison"),
    "executor_expertise_analysis": ("executor_expertise_analysis",),
    "research_gap_identification": ("research_gap_identification",),
    "knowledge_transfer_prep": ("knowledge_transfer_prep", "knowledge_transfer_preparation"),
    "rfp_generation": ("rfp_generation",),
    "proposal_evaluation": ("proposal_evaluation",),
    "feedback_generation": ("feedback_generation",),
    "llm_as_judge": ("llm_as_judge", "llm_as_judge_scoring"),
}


class CitationPolicy(str, Enum):
    """Citation enforcement policy for assembled prompts."""

    STRICT_SOURCE_ID = "strict_source_id"
    OPTIONAL = "optional"
    NONE = "none"


@dataclass(frozen=True)
class ExpectedOutputFormat:
    """Expected output format ('json' or 'markdown') and mandatory schema keys."""

    format: str = "json"
    required_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.format, str) or self.format.strip().lower() not in (
            "json",
            "markdown",
        ):
            raise DomainValidationError(
                f"ExpectedOutputFormat.format must be 'json' or 'markdown', got '{self.format}'."
            )
        fmt = self.format.strip().lower()
        keys = tuple(
            k.strip() for k in self.required_keys if isinstance(k, str) and k.strip()
        )
        object.__setattr__(self, "format", fmt)
        object.__setattr__(self, "required_keys", keys)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | ExpectedOutputFormat) -> ExpectedOutputFormat:
        if isinstance(data, ExpectedOutputFormat):
            return data
        if isinstance(data, str):
            return cls(format=data, required_keys=())
        if not isinstance(data, Mapping):
            raise DomainValidationError("ExpectedOutputFormat data must be a mapping.")
        fmt = str(data.get("format", "json"))
        raw_keys = data.get("requiredKeys") or data.get("required_keys") or ()
        return cls(format=fmt, required_keys=tuple(str(k) for k in raw_keys))


@dataclass(frozen=True)
class PromptHistoryEntry:
    """Immutable historical snapshot of a previous PromptTemplate version."""

    version: VersionTag
    last_updated: str
    system_template: str
    user_template: str
    required_variables: tuple[str, ...]
    expected_output_format: ExpectedOutputFormat
    citation_policy: CitationPolicy
    judge_rubric_id: str
    change_note: str = ""

    def __post_init__(self) -> None:
        v_tag = (
            self.version
            if isinstance(self.version, VersionTag)
            else VersionTag.from_version_id(str(self.version))
        )
        if not isinstance(self.last_updated, str) or not self.last_updated.strip():
            raise DomainValidationError("PromptHistoryEntry.last_updated cannot be empty.")
        if not isinstance(self.system_template, str) or not self.system_template.strip():
            raise DomainValidationError("PromptHistoryEntry.system_template cannot be empty.")
        if not isinstance(self.user_template, str) or not self.user_template.strip():
            raise DomainValidationError("PromptHistoryEntry.user_template cannot be empty.")
        req_vars = tuple(
            v.strip() for v in self.required_variables if isinstance(v, str) and v.strip()
        )
        out_fmt = ExpectedOutputFormat.from_dict(self.expected_output_format)
        cit_pol = (
            self.citation_policy
            if isinstance(self.citation_policy, CitationPolicy)
            else CitationPolicy(str(self.citation_policy).strip().lower())
        )
        if not isinstance(self.judge_rubric_id, str) or not self.judge_rubric_id.strip():
            raise DomainValidationError("PromptHistoryEntry.judge_rubric_id cannot be empty.")

        object.__setattr__(self, "version", v_tag)
        object.__setattr__(self, "last_updated", self.last_updated.strip())
        object.__setattr__(self, "system_template", self.system_template.strip())
        object.__setattr__(self, "user_template", self.user_template.strip())
        object.__setattr__(self, "required_variables", req_vars)
        object.__setattr__(self, "expected_output_format", out_fmt)
        object.__setattr__(self, "citation_policy", cit_pol)
        object.__setattr__(self, "judge_rubric_id", self.judge_rubric_id.strip())
        object.__setattr__(self, "change_note", str(self.change_note or "").strip())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | PromptHistoryEntry) -> PromptHistoryEntry:
        if isinstance(data, PromptHistoryEntry):
            return data
        if not isinstance(data, Mapping):
            raise DomainValidationError("PromptHistoryEntry must be constructed from a mapping.")
        return cls(
            version=VersionTag.from_version_id(str(data["version"])),
            last_updated=str(data.get("lastUpdated") or data.get("last_updated") or "2026-10-01"),
            system_template=str(data.get("systemTemplate") or data.get("system_template") or ""),
            user_template=str(data.get("userTemplate") or data.get("user_template") or ""),
            required_variables=tuple(
                str(v)
                for v in (data.get("requiredVariables") or data.get("required_variables") or ())
            ),
            expected_output_format=ExpectedOutputFormat.from_dict(
                data.get("expectedOutputFormat")
                or data.get("expected_output_format")
                or {"format": "json", "requiredKeys": []}
            ),
            citation_policy=CitationPolicy(
                str(
                    data.get("citationPolicy")
                    or data.get("citation_policy")
                    or "strict_source_id"
                )
            ),
            judge_rubric_id=str(
                data.get("judgeRubricId")
                or data.get("judge_rubric_id")
                or "RUBRIC-JUDGE-20PT-V1"
            ),
            change_note=str(data.get("changeNote") or data.get("change_note") or ""),
        )


@dataclass(frozen=True)
class AssembledPrompt:
    """Immutable assembled prompt ready to be sent to an LLM runtime."""

    prompt_id: str
    version: VersionTag
    task_type: PromptTaskCategory
    system_message: str
    user_message: str
    expected_output_format: ExpectedOutputFormat
    citation_policy: CitationPolicy
    judge_rubric_id: str

    @property
    def system(self) -> str:
        return self.system_message

    @property
    def user(self) -> str:
        return self.user_message

    def to_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.user_message},
        ]

    def __getitem__(self, key: str) -> str:
        if key in ("system", "system_message"):
            return self.system_message
        if key in ("user", "user_message"):
            return self.user_message
        raise KeyError(key)


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
            raise DomainValidationError(
                "HardwareConstraints.preferred_device must be 'gpu' or 'cpu'."
            )
        object.__setattr__(
            self, "extra_constraints", MappingProxyType(dict(self.extra_constraints))
        )


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
        if (
            isinstance(context_window, bool)
            or not isinstance(context_window, int)
            or context_window <= 0
        ):
            raise DomainValidationError("OfflineModelSpec.context_window must be > 0.")
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
        ):
            raise DomainValidationError("OfflineModelSpec.max_output_tokens must be > 0.")
        if temperature_default < 0.0 or temperature_default > 2.0:
            raise DomainValidationError(
                "OfflineModelSpec.temperature_default must be in [0.0, 2.0]."
            )

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
        if task_name in self._supported_tasks:
            return True
        try:
            cat = PromptTaskCategory(task_name)
            aliases = TASK_LEGACY_ALIASES.get(cat.value, (cat.value,))
            return any(alias in self._supported_tasks for alias in aliases)
        except ValueError:
            return False


class PromptTemplate(AggregateRoot):
    """Versioned prompt template aggregate root storing active config and immutable version history (Section 8.3)."""

    def __init__(
        self,
        prompt_id: str | None = None,
        task_type: PromptTaskCategory | str | None = None,
        version: VersionTag | str | None = None,
        system_template: str | None = None,
        user_template: str = "",
        required_variables: Iterable[str] | None = None,
        expected_output_format: ExpectedOutputFormat | Mapping[str, Any] | None = None,
        citation_policy: CitationPolicy | str = CitationPolicy.STRICT_SOURCE_ID,
        judge_rubric_id: str = "RUBRIC-JUDGE-20PT-V1",
        last_updated: str = "2026-10-29",
        history: Iterable[PromptHistoryEntry | Mapping[str, Any]] = (),
        target_model_id: str = "diba-base",
        *,
        # Backward-compatible keyword aliases from Phase 3
        template_id: str | None = None,
        task_category: PromptTaskCategory | str | None = None,
        version_tag: VersionTag | str | None = None,
        system_prompt: str | None = None,
        expected_output_keys: Iterable[str] = (),
        source_citation_required: bool | None = None,
        uncertainty_declaration_required: bool = True,
    ) -> None:
        resolved_id = prompt_id if prompt_id is not None else template_id
        if not isinstance(resolved_id, str) or not resolved_id.strip():
            raise DomainValidationError("PromptTemplate.prompt_id cannot be empty.")

        raw_task = task_type if task_type is not None else task_category
        if raw_task is None:
            raise DomainValidationError("PromptTemplate.task_type cannot be empty.")
        try:
            cat = (
                raw_task
                if isinstance(raw_task, PromptTaskCategory)
                else PromptTaskCategory(str(raw_task))
            )
        except ValueError as exc:
            raise DomainValidationError(f"Invalid PromptTaskCategory: {raw_task}") from exc

        raw_ver = version if version is not None else version_tag
        if raw_ver is None:
            raise DomainValidationError("PromptTemplate.version is required.")
        v_tag = (
            raw_ver
            if isinstance(raw_ver, VersionTag)
            else VersionTag.from_version_id(str(raw_ver))
        )

        resolved_sys = system_template if system_template is not None else system_prompt
        if not isinstance(resolved_sys, str) or not resolved_sys.strip():
            raise DomainValidationError("PromptTemplate.system_template cannot be empty.")
        if not isinstance(user_template, str) or not user_template.strip():
            raise DomainValidationError("PromptTemplate.user_template cannot be empty.")
        if not isinstance(target_model_id, str) or not target_model_id.strip():
            raise DomainValidationError("PromptTemplate.target_model_id cannot be empty.")
        if not isinstance(last_updated, str) or not last_updated.strip():
            raise DomainValidationError("PromptTemplate.last_updated cannot be empty.")
        if not isinstance(judge_rubric_id, str) or not judge_rubric_id.strip():
            raise DomainValidationError("PromptTemplate.judge_rubric_id cannot be empty.")

        # Determine required_variables (explicit or extracted from {{...}} placeholders)
        extracted_vars = tuple(dict.fromkeys(PLACEHOLDER_PATTERN.findall(user_template)))
        if required_variables is not None:
            req_vars = tuple(
                v.strip() for v in required_variables if isinstance(v, str) and v.strip()
            )
        else:
            req_vars = extracted_vars

        # Determine expected_output_format
        if expected_output_format is not None:
            out_fmt = ExpectedOutputFormat.from_dict(expected_output_format)
        else:
            keys_tuple = tuple(
                k.strip() for k in expected_output_keys if isinstance(k, str) and k.strip()
            )
            out_fmt = ExpectedOutputFormat(format="json", required_keys=keys_tuple)

        # Determine citation_policy
        if source_citation_required is False and citation_policy == CitationPolicy.STRICT_SOURCE_ID:
            cit_pol = CitationPolicy.NONE
        else:
            try:
                cit_pol = (
                    citation_policy
                    if isinstance(citation_policy, CitationPolicy)
                    else CitationPolicy(str(citation_policy).strip().lower())
                )
            except ValueError as exc:
                raise DomainValidationError(
                    f"Invalid CitationPolicy: {citation_policy}"
                ) from exc

        parsed_history = tuple(PromptHistoryEntry.from_dict(h) for h in history)
        hist_versions = [h.version.tag for h in parsed_history]
        if len(hist_versions) != len(set(hist_versions)):
            raise DomainValidationError(
                f"Duplicate versions detected in history of prompt '{resolved_id}'."
            )

        composite_id = f"{resolved_id.strip()}:{v_tag.tag}"
        super().__init__(composite_id)
        self._prompt_id = resolved_id.strip()
        self._task_type = cat
        self._version = v_tag
        self._last_updated = last_updated.strip()
        self._target_model_id = target_model_id.strip()
        self._system_template = resolved_sys.strip()
        self._user_template = user_template.strip()
        self._required_variables = req_vars
        self._expected_output_format = out_fmt
        self._citation_policy = cit_pol
        self._judge_rubric_id = judge_rubric_id.strip()
        self._history = parsed_history
        self._source_citation_required = (
            bool(source_citation_required)
            if source_citation_required is not None
            else (cit_pol == CitationPolicy.STRICT_SOURCE_ID)
        )
        self._uncertainty_declaration_required = bool(uncertainty_declaration_required)

    @property
    def prompt_id(self) -> str:
        return self._prompt_id

    @property
    def template_id(self) -> str:
        return self._prompt_id

    @property
    def task_type(self) -> PromptTaskCategory:
        return self._task_type

    @property
    def task_category(self) -> PromptTaskCategory:
        return self._task_type

    @property
    def version(self) -> VersionTag:
        return self._version

    @property
    def version_tag(self) -> VersionTag:
        return self._version

    @property
    def last_updated(self) -> str:
        return self._last_updated

    @property
    def target_model_id(self) -> str:
        return self._target_model_id

    @property
    def system_template(self) -> str:
        return self._system_template

    @property
    def system_prompt(self) -> str:
        return self._system_template

    @property
    def user_template(self) -> str:
        return self._user_template

    @property
    def required_variables(self) -> tuple[str, ...]:
        return self._required_variables

    @property
    def required_placeholders(self) -> tuple[str, ...]:
        return self._required_variables

    @property
    def expected_output_format(self) -> ExpectedOutputFormat:
        return self._expected_output_format

    @property
    def expected_output_keys(self) -> tuple[str, ...]:
        return self._expected_output_format.required_keys

    @property
    def citation_policy(self) -> CitationPolicy:
        return self._citation_policy

    @property
    def judge_rubric_id(self) -> str:
        return self._judge_rubric_id

    @property
    def history(self) -> tuple[PromptHistoryEntry, ...]:
        return self._history

    @property
    def source_citation_required(self) -> bool:
        return self._source_citation_required

    @property
    def uncertainty_declaration_required(self) -> bool:
        return self._uncertainty_declaration_required

    def list_available_versions(self) -> tuple[VersionTag, ...]:
        """Return all historical versions and the active version in sequence order."""
        all_versions = [h.version for h in self._history] + [self._version]
        return tuple(sorted(all_versions, key=lambda v: v.sequence_number))

    def get_version_snapshot(self, version: VersionTag | str) -> PromptTemplate | None:
        """Retrieve a PromptTemplate snapshot for either the active version or a version in history."""
        target_tag = version.tag if isinstance(version, VersionTag) else str(version).strip()
        if self._version.tag == target_tag:
            return self
        for h in self._history:
            if h.version.tag == target_tag:
                return PromptTemplate(
                    prompt_id=self._prompt_id,
                    task_type=self._task_type,
                    version=h.version,
                    last_updated=h.last_updated,
                    system_template=h.system_template,
                    user_template=h.user_template,
                    required_variables=h.required_variables,
                    expected_output_format=h.expected_output_format,
                    citation_policy=h.citation_policy,
                    judge_rubric_id=h.judge_rubric_id,
                    history=(),
                    target_model_id=self._target_model_id,
                )
        return None

    def create_new_version(
        self,
        new_version: VersionTag | str,
        system_template: str | None = None,
        user_template: str | None = None,
        required_variables: Iterable[str] | None = None,
        expected_output_format: ExpectedOutputFormat | Mapping[str, Any] | None = None,
        citation_policy: CitationPolicy | str | None = None,
        judge_rubric_id: str | None = None,
        last_updated: str = "2026-10-29",
        change_note: str = "",
    ) -> PromptTemplate:
        """Create a new PromptTemplate version while archiving the current active version into history."""
        next_v = (
            new_version
            if isinstance(new_version, VersionTag)
            else VersionTag.from_version_id(str(new_version))
        )
        if next_v.sequence_number <= self._version.sequence_number:
            raise DomainValidationError(
                f"New version '{next_v.tag}' must be greater than current version '{self._version.tag}'."
            )
        archived_entry = PromptHistoryEntry(
            version=self._version,
            last_updated=self._last_updated,
            system_template=self._system_template,
            user_template=self._user_template,
            required_variables=self._required_variables,
            expected_output_format=self._expected_output_format,
            citation_policy=self._citation_policy,
            judge_rubric_id=self._judge_rubric_id,
            change_note=change_note,
        )
        return PromptTemplate(
            prompt_id=self._prompt_id,
            task_type=self._task_type,
            version=next_v,
            last_updated=last_updated,
            system_template=system_template or self._system_template,
            user_template=user_template or self._user_template,
            required_variables=(
                required_variables
                if required_variables is not None
                else self._required_variables
            ),
            expected_output_format=(
                expected_output_format
                if expected_output_format is not None
                else self._expected_output_format
            ),
            citation_policy=(
                citation_policy if citation_policy is not None else self._citation_policy
            ),
            judge_rubric_id=judge_rubric_id or self._judge_rubric_id,
            history=self._history + (archived_entry,),
            target_model_id=self._target_model_id,
        )

    def assemble_messages(self, variables: Mapping[str, Any]) -> dict[str, str]:
        """Render the prompt template with required variables and return {'system': ..., 'user': ...}."""
        missing = [
            var
            for var in self._required_variables
            if var not in variables
            or variables[var] is None
            or not str(variables[var]).strip()
        ]
        if missing:
            raise MissingPromptVariableError(
                prompt_id=self._prompt_id,
                missing_variables=missing,
            )
        rendered_sys = self._system_template
        rendered_user = self._user_template
        for key, val in variables.items():
            pattern = r"\{\{\s*" + re.escape(key) + r"\s*\}\}"
            rendered_sys = re.sub(pattern, str(val), rendered_sys)
            rendered_user = re.sub(pattern, str(val), rendered_user)
        return {"system": rendered_sys, "user": rendered_user}


class PromptTemplateRepository(ABC):
    """Abstract repository for versioned PromptTemplate aggregates."""

    @abstractmethod
    def save(self, template: PromptTemplate) -> None:
        """Persist a PromptTemplate version."""

    @abstractmethod
    def get_active_by_id(self, prompt_id: str) -> PromptTemplate | None:
        """Retrieve the active version of a PromptTemplate by promptId."""

    @abstractmethod
    def get_by_id_and_version(
        self, template_id: str, version_tag: VersionTag | str
    ) -> PromptTemplate | None:
        """Retrieve a specific version (active or historical) of a PromptTemplate."""

    @abstractmethod
    def get_latest_by_task(self, task_category: PromptTaskCategory | str) -> PromptTemplate | None:
        """Retrieve the active/latest versioned PromptTemplate for a task category."""

    @abstractmethod
    def list_by_task(self, task_category: PromptTaskCategory | str) -> Sequence[PromptTemplate]:
        """List all versions of PromptTemplates registered for a task category."""

    def list_all_active(self) -> Sequence[PromptTemplate]:
        """Return all active PromptTemplates across the 12 task categories."""
        results: list[PromptTemplate] = []
        for cat in PromptTaskCategory:
            tpl = self.get_latest_by_task(cat)
            if tpl is not None:
                results.append(tpl)
        return tuple(results)


class PromptAssembler:
    """Domain service that validates variables and assembles versioned prompts with citation & schema policies."""

    def __init__(self, repository: PromptTemplateRepository) -> None:
        self._repository = repository

    @staticmethod
    def build_citation_instruction(policy: CitationPolicy) -> str:
        if policy == CitationPolicy.STRICT_SOURCE_ID:
            return (
                "[سیاست استناد الزامی (strict_source_id): هر گزاره باید به شناسه دقیق سند و بخش منبع "
                "(مانند DOC-ID#sec) مستند شود و از هرگونه حدس یا ادعای بدون پشتوانه خودداری گردد.]"
            )
        if policy == CitationPolicy.OPTIONAL:
            return (
                "[سیاست استناد اختیاری (optional): در صورت وجود شناسه منبع در ورودی، به آن ارجاع بده.]"
            )
        return "[سیاست استناد (none): نیازی به درج شناسه استناد درون‌متنی نیست.]"

    @staticmethod
    def build_output_format_instruction(output_format: ExpectedOutputFormat) -> str:
        keys_str = ", ".join(output_format.required_keys)
        if output_format.format == "json":
            return (
                f"[قالب خروجی مورد انتظار (JSON): پاسخ را صرفاً در قالب یک شیء JSON معتبر "
                f"شامل کلیدهای الزامی زیر برگردان: {keys_str}]"
            )
        return (
            f"[قالب خروجی مورد انتظار (Markdown): پاسخ را در قالب Markdown ساختاریافته "
            f"شامل بخش‌های الزامی زیر تنظیم کن: {keys_str}]"
        )

    def resolve_template(
        self,
        prompt_id: str | None = None,
        version: VersionTag | str | None = None,
        task_type: PromptTaskCategory | str | None = None,
    ) -> PromptTemplate:
        """Resolve a PromptTemplate from the registry by promptId or taskType and optional version."""
        template: PromptTemplate | None = None
        if prompt_id is not None:
            if version is not None:
                template = self._repository.get_by_id_and_version(prompt_id, version)
            else:
                template = self._repository.get_active_by_id(prompt_id)
        elif task_type is not None:
            active_tpl = self._repository.get_latest_by_task(task_type)
            if active_tpl is not None:
                if version is not None:
                    template = self._repository.get_by_id_and_version(
                        active_tpl.prompt_id, version
                    )
                else:
                    template = active_tpl
        else:
            raise DomainValidationError(
                "PromptAssembler requires either 'prompt_id' or 'task_type'."
            )

        if template is None:
            target_desc = prompt_id or str(task_type)
            ver_desc = f"@{version}" if version is not None else " (active)"
            raise EntityNotFoundError(
                f"PromptTemplate '{target_desc}{ver_desc}' not found in prompt registry."
            )
        return template

    def assemble(
        self,
        prompt_id: str | None = None,
        variables: Mapping[str, Any] | None = None,
        version: VersionTag | str | None = None,
        task_type: PromptTaskCategory | str | None = None,
    ) -> AssembledPrompt:
        """Validate required variables, render placeholders, and append citation & output schema instructions."""
        template = self.resolve_template(
            prompt_id=prompt_id, version=version, task_type=task_type
        )
        vars_dict = dict(variables or {})
        rendered = template.assemble_messages(vars_dict)

        citation_suffix = self.build_citation_instruction(template.citation_policy)
        schema_suffix = self.build_output_format_instruction(template.expected_output_format)

        final_user_message = (
            f"{rendered['user']}\n\n"
            f"---\n"
            f"{citation_suffix}\n"
            f"{schema_suffix}"
        )

        return AssembledPrompt(
            prompt_id=template.prompt_id,
            version=template.version,
            task_type=template.task_type,
            system_message=rendered["system"],
            user_message=final_user_message,
            expected_output_format=template.expected_output_format,
            citation_policy=template.citation_policy,
            judge_rubric_id=template.judge_rubric_id,
        )


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


__all__ = [
    "AssembledPrompt",
    "CitationPolicy",
    "ExpectedOutputFormat",
    "HardwareConstraints",
    "ModelRegistryRepository",
    "ModelRole",
    "OfflineModelSpec",
    "PromptAssembler",
    "PromptHistoryEntry",
    "PromptTaskCategory",
    "PromptTemplate",
    "PromptTemplateRepository",
]
