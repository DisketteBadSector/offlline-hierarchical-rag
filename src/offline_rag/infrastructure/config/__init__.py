"""Configuration and YAML Fixture Loader for layers.yaml, nodes.yaml, models.yaml, and per-category prompt files in data/prompts/."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import yaml

from offline_rag.domain.model_prompt import (
    CitationPolicy,
    ExpectedOutputFormat,
    HardwareConstraints,
    ModelRole,
    OfflineModelSpec,
    PromptHistoryEntry,
    PromptTaskCategory,
    PromptTemplate,
    PromptTemplateRepository,
)
from offline_rag.domain.organization import (
    BoundaryType,
    Layer,
    LayerTransferRule,
    Node,
    NodeTransferConfig,
    OrganizationHierarchyValidator,
)
from offline_rag.domain.shared.errors import DomainValidationError
from offline_rag.domain.shared.value_objects import (
    JudgeThreshold,
    LayerIndex,
    NodeIdentifier,
    TransferDetailLevel,
    VersionTag,
)
from offline_rag.infrastructure.persistence.in_memory import InMemoryUnitOfWork


def load_yaml_dict(path: Path) -> dict[str, Any]:
    """Load a YAML file into a Python dict using PyYAML."""
    if not path.exists():
        raise FileNotFoundError(f"YAML fixture file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise DomainValidationError(f"Expected mapping root in {path}")
    return data


PROMPT_CATEGORY_FILES: tuple[str, ...] = (
    "qa.yaml",
    "technical_summary.yaml",
    "managerial_summary.yaml",
    "issue_decomposition.yaml",
    "sub_issue_comparison.yaml",
    "executor_analysis.yaml",
    "research_gap.yaml",
    "knowledge_transfer.yaml",
    "rfp_generation.yaml",
    "proposal_evaluation.yaml",
    "feedback_generation.yaml",
    "judge.yaml",
)


class YamlPromptTemplateRepository(PromptTemplateRepository):
    """Repository loading versioned PromptTemplates from per-category YAML files in data/prompts/.

    Each file defines ``promptId``, ``taskType``, ``activeVersion`` and a ``versions`` list.
    The entry matching ``activeVersion`` becomes the active template; all others are kept as history.
    """

    DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[4] / "data" / "prompts"

    def __init__(self, prompts_path: Path | str | None = None) -> None:
        target = Path(prompts_path) if prompts_path is not None else self.DEFAULT_PROMPTS_DIR
        if target.is_dir():
            files = sorted(p for p in target.glob("*.yaml"))
        elif target.exists():
            files = [target]
        else:
            raise FileNotFoundError(f"Prompt path not found: {target}")
        if not files:
            raise DomainValidationError(f"No prompt YAML files found in {target}")

        self._prompts_path = target
        self._templates_by_id_ver: dict[tuple[str, str], PromptTemplate] = {}
        self._active_by_id: dict[str, PromptTemplate] = {}
        self._active_by_task: dict[PromptTaskCategory, PromptTemplate] = {}
        self._source_files: dict[str, Path] = {}
        for f in files:
            self._load_prompt_file(f)

    @property
    def source_files(self) -> dict[str, Path]:
        """Mapping of promptId -> YAML file it was loaded from."""
        return dict(self._source_files)

    def _load_prompt_file(self, prompt_file: Path) -> None:
        raw = load_yaml_dict(prompt_file)
        for key in ("promptId", "taskType", "activeVersion", "versions"):
            if key not in raw:
                raise DomainValidationError(f"Prompt file {prompt_file.name} missing '{key}'.")
        prompt_id = str(raw["promptId"])
        if prompt_id in self._source_files:
            raise DomainValidationError(
                f"Duplicate promptId '{prompt_id}' in {prompt_file.name} and "
                f"{self._source_files[prompt_id].name}."
            )
        task_cat = PromptTaskCategory(str(raw["taskType"]))
        active_ver = str(raw["activeVersion"]).strip()
        versions = raw["versions"]
        if not isinstance(versions, list) or not versions:
            raise DomainValidationError(f"Prompt file {prompt_file.name} has no versions.")

        active_entry = next(
            (v for v in versions if isinstance(v, dict) and str(v.get("version")) == active_ver),
            None,
        )
        if active_entry is None:
            raise DomainValidationError(
                f"activeVersion '{active_ver}' not found in versions of {prompt_file.name}."
            )
        history_entries = tuple(
            PromptHistoryEntry.from_dict(v)
            for v in versions
            if isinstance(v, dict) and str(v.get("version")) != active_ver
        )
        template = PromptTemplate(
            prompt_id=prompt_id,
            task_type=task_cat,
            version=VersionTag.from_version_id(active_ver),
            last_updated=str(active_entry.get("lastUpdated", "2026-10-29")),
            system_template=str(active_entry["systemTemplate"]),
            user_template=str(active_entry["userTemplate"]),
            required_variables=tuple(str(x) for x in active_entry.get("requiredVariables", [])),
            expected_output_format=ExpectedOutputFormat.from_dict(
                active_entry["expectedOutputFormat"]
            ),
            citation_policy=CitationPolicy(str(active_entry["citationPolicy"])),
            judge_rubric_id=str(active_entry["judgeRubricId"]),
            history=history_entries,
            target_model_id=str(raw.get("targetModelId", "diba-base")),
        )
        self._source_files[prompt_id] = prompt_file
        self.save(template)

    def save(self, template: PromptTemplate) -> None:
        self._templates_by_id_ver[(template.prompt_id, template.version.tag)] = template
        for hist in template.history:
            snap = template.get_version_snapshot(hist.version)
            if snap is not None:
                self._templates_by_id_ver[(template.prompt_id, hist.version.tag)] = snap
        self._active_by_id[template.prompt_id] = template
        self._active_by_task[template.task_type] = template

    def get_active_by_id(self, prompt_id: str) -> PromptTemplate | None:
        return self._active_by_id.get(prompt_id)

    def get_by_id_and_version(
        self, template_id: str, version_tag: VersionTag | str
    ) -> PromptTemplate | None:
        tag = version_tag.tag if isinstance(version_tag, VersionTag) else str(version_tag).strip()
        direct = self._templates_by_id_ver.get((template_id, tag))
        if direct is not None:
            return direct
        active = self._active_by_id.get(template_id)
        if active is not None:
            return active.get_version_snapshot(tag)
        return None

    def get_latest_by_task(
        self, task_category: PromptTaskCategory | str
    ) -> PromptTemplate | None:
        cat = (
            task_category
            if isinstance(task_category, PromptTaskCategory)
            else PromptTaskCategory(str(task_category))
        )
        if cat in self._active_by_task:
            return self._active_by_task[cat]
        items = self.list_by_task(cat)
        return items[-1] if items else None

    def list_by_task(
        self, task_category: PromptTaskCategory | str
    ) -> Sequence[PromptTemplate]:
        cat = (
            task_category
            if isinstance(task_category, PromptTaskCategory)
            else PromptTaskCategory(str(task_category))
        )
        matched = [
            t for t in self._templates_by_id_ver.values() if t.task_type == cat
        ]
        return tuple(sorted(matched, key=lambda t: t.version.sequence_number))

    def list_all_active(self) -> Sequence[PromptTemplate]:
        return tuple(self._active_by_id.values())


class FixtureConfigLoader:
    """Loader and validator for layers.yaml, nodes.yaml, models.yaml, and data/prompts/*.yaml."""

    @staticmethod
    def load_transfer_rules(layers_yaml_path: Path) -> tuple[LayerTransferRule, ...]:
        raw = load_yaml_dict(layers_yaml_path)
        rules_data = raw.get("layer_transfer_rules", [])
        if not rules_data:
            raise DomainValidationError(f"No 'layer_transfer_rules' found in {layers_yaml_path}")

        rules: list[LayerTransferRule] = []
        for item in rules_data:
            target_idx = LayerIndex(int(item["targetLayerIndex"]))
            detail = TransferDetailLevel(str(item["detailLevel"]))
            req_fields = tuple(str(f) for f in item.get("requiredFields", []))
            max_len = int(item["maxLength"])
            min_score = float(item.get("minJudgeScore", 18.0))
            usage_scope = str(item.get("usageScope", "upward_transfer"))
            rules.append(
                LayerTransferRule(
                    target_layer_index=target_idx,
                    detail_level=detail,
                    required_fields=req_fields,
                    max_length=max_len,
                    min_judge_score=JudgeThreshold(min_acceptable_score=min_score),
                    usage_scope=usage_scope,
                )
            )
        return tuple(rules)

    @staticmethod
    def load_hierarchy(
        nodes_yaml_path: Path, layers_yaml_path: Path
    ) -> tuple[tuple[Layer, ...], tuple[Node, ...], tuple[LayerTransferRule, ...]]:
        """Load layers.yaml and nodes.yaml and validate R-01 through R-08."""
        rules = FixtureConfigLoader.load_transfer_rules(layers_yaml_path)
        rule_by_layer = {r.target_layer_index: r for r in rules}

        raw_nodes = load_yaml_dict(nodes_yaml_path)
        layers_data = raw_nodes.get("layers", [])
        nodes_data = raw_nodes.get("nodes", [])

        layers: list[Layer] = []
        for l_item in layers_data:
            l_idx = LayerIndex(int(l_item["layerIndex"]))
            layers.append(
                Layer(
                    layer_index=l_idx,
                    name=str(l_item["name"]),
                    description=str(l_item["description"]),
                    required=bool(l_item.get("required", True)),
                    transfer_rule=rule_by_layer.get(l_idx),
                )
            )

        nodes: list[Node] = []
        for n_item in nodes_data:
            n_id = NodeIdentifier(str(n_item["nodeId"]))
            l_idx = LayerIndex(int(n_item["layerIndex"]))
            p_raw = n_item.get("parentNodeId")
            parent_id = NodeIdentifier(str(p_raw)) if p_raw is not None else None
            b_type = BoundaryType(str(n_item["boundaryType"]))
            t_raw = n_item.get("transfer")
            transfer_cfg: NodeTransferConfig | None = None
            if isinstance(t_raw, dict):
                transfer_cfg = NodeTransferConfig(
                    rule_layer_index=LayerIndex(int(t_raw["ruleLayerIndex"])),
                    extra_required_fields=tuple(
                        str(f) for f in t_raw.get("extraRequiredFields", [])
                    ),
                )
            nodes.append(
                Node(
                    node_id=n_id,
                    name=str(n_item["name"]),
                    layer_index=l_idx,
                    parent_node_id=parent_id,
                    activity_domain=str(n_item["activityDomain"]),
                    boundary_type=b_type,
                    confidentiality=str(n_item.get("confidentiality", "internal")),
                    transfer=transfer_cfg,
                )
            )

        OrganizationHierarchyValidator.validate_hierarchy(
            layers=layers, nodes=nodes, transfer_rules=rules
        )
        return tuple(layers), tuple(nodes), rules

    @staticmethod
    def load_models_registry(models_yaml_path: Path) -> tuple[OfflineModelSpec, ...]:
        """Load and validate models.yaml into OfflineModelSpec domain aggregates."""
        raw = load_yaml_dict(models_yaml_path)
        models_data = raw.get("models", [])
        if not models_data:
            raise DomainValidationError(f"No 'models' found in {models_yaml_path}")

        specs: list[OfflineModelSpec] = []
        for m_item in models_data:
            hw_raw = m_item.get("hardwareConstraints")
            if not isinstance(hw_raw, dict):
                raise DomainValidationError(
                    f"Model '{m_item.get('modelId')}' missing hardwareConstraints dict."
                )
            hw = HardwareConstraints(
                min_vram_gb=float(hw_raw.get("minVramGb", 2.0)),
                min_ram_gb=float(hw_raw.get("minRamGb", 4.0)),
                preferred_device=str(hw_raw.get("preferredDevice", "gpu")),
                cpu_fallback_supported=bool(hw_raw.get("cpuFallbackSupported", True)),
                execution_mode=str(hw_raw.get("executionMode", "sequential_load_unload")),
                extra_constraints=hw_raw,
            )
            specs.append(
                OfflineModelSpec(
                    model_id=str(m_item["modelId"]),
                    role=ModelRole(str(m_item["role"]).lower()),
                    provider_endpoint=str(m_item["providerEndpoint"]),
                    context_window=int(m_item["contextWindow"]),
                    max_output_tokens=int(m_item["maxOutputTokens"]),
                    temperature_default=float(m_item["temperatureDefault"]),
                    supported_tasks=tuple(str(t) for t in m_item.get("supportedTasks", [])),
                    hardware_constraints=hw,
                    source_name=str(m_item.get("sourceName", "")),
                    weights_format=str(m_item.get("format", "")),
                    local_weights_path=str(m_item.get("localWeightsPath", "")),
                )
            )
        return tuple(specs)

    @staticmethod
    def load_prompt_registry(
        prompts_path: Path | None = None,
    ) -> YamlPromptTemplateRepository:
        """Load per-category data/prompts/*.yaml files into a YamlPromptTemplateRepository."""
        return YamlPromptTemplateRepository(prompts_path=prompts_path)

    @staticmethod
    def bootstrap_in_memory_uow(
        fixtures_dir: Path, prompts_dir: Path | None = None
    ) -> InMemoryUnitOfWork:
        """Load layers.yaml, nodes.yaml, models.yaml, and data/prompts/*.yaml into an InMemoryUnitOfWork."""
        uow = InMemoryUnitOfWork()
        layers, nodes, rules = FixtureConfigLoader.load_hierarchy(
            nodes_yaml_path=fixtures_dir / "nodes.yaml",
            layers_yaml_path=fixtures_dir / "layers.yaml",
        )
        models = FixtureConfigLoader.load_models_registry(fixtures_dir / "models.yaml")

        resolved_prompts_dir = (
            prompts_dir
            if prompts_dir is not None
            else fixtures_dir.parent / "prompts"
        )
        prompt_templates: Sequence[PromptTemplate] = ()
        if resolved_prompts_dir.is_dir() and any(resolved_prompts_dir.glob("*.yaml")):
            yaml_repo = YamlPromptTemplateRepository(resolved_prompts_dir)
            prompt_templates = yaml_repo.list_all_active()

        with uow:
            for r in rules:
                uow.layers.save_transfer_rule(r)
            for lyr in layers:
                uow.layers.save_layer(lyr)
            for nd in nodes:
                uow.nodes.save(nd)
            for m in models:
                uow.models.save(m)
            for pt in prompt_templates:
                uow.prompts.save(pt)
            uow.commit()
        return uow


__all__ = [
    "FixtureConfigLoader",
    "PROMPT_CATEGORY_FILES",
    "YamlPromptTemplateRepository",
    "load_yaml_dict",
]
