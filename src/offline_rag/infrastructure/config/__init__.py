"""Configuration and YAML Fixture Loader for layers.yaml, nodes.yaml, and models.yaml."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from offline_rag.domain.model_prompt import (
    HardwareConstraints,
    ModelRole,
    OfflineModelSpec,
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
)
from offline_rag.infrastructure.persistence.in_memory import InMemoryUnitOfWork

try:
    import yaml as _yaml  # type: ignore[import-not-found]
except ImportError:
    _yaml = None


def _parse_scalar(val: str) -> Any:
    val = val.strip()
    if not val or val in ("null", "~"):
        return None
    if val == "true":
        return True
    if val == "false":
        return False
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    if val.startswith("[") and val.endswith("]"):
        try:
            return ast.literal_eval(val)
        except Exception:
            return val
    try:
        return float(val) if "." in val else int(val)
    except ValueError:
        return val


def load_yaml_dict(path: Path) -> dict[str, Any]:
    """Load a YAML file into a Python dict using PyYAML or built-in indentation parser."""
    if not path.exists():
        raise FileNotFoundError(f"YAML fixture file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if _yaml is not None:
        data = _yaml.safe_load(text)
        if not isinstance(data, dict):
            raise DomainValidationError(f"Expected mapping root in {path}")
        return data

    root: dict[str, Any] = {}
    current_top_list: list[Any] | None = None
    current_item: dict[str, Any] | None = None
    current_subkey: str | None = None
    current_subcontainer: list[Any] | dict[str, Any] | None = None
    current_sub_subkey: str | None = None

    for raw_line in text.splitlines():
        stripped_comment = raw_line.split("#", 1)[0].rstrip()
        if not stripped_comment.strip():
            continue
        indent = len(stripped_comment) - len(stripped_comment.lstrip(" "))
        line = stripped_comment.strip()

        if indent == 0:
            current_item = None
            current_subkey = None
            current_subcontainer = None
            current_sub_subkey = None
            if ":" in line:
                k, v = line.split(":", 1)
                k, v = k.strip(), v.strip()
                if v:
                    root[k] = _parse_scalar(v)
                    current_top_list = None
                else:
                    root[k] = []
                    current_top_list = root[k]
        elif indent == 2:
            if line.startswith("- ") and isinstance(current_top_list, list):
                current_item = {}
                current_top_list.append(current_item)
                current_subkey = None
                current_subcontainer = None
                current_sub_subkey = None
                rest = line[2:].strip()
                if ":" in rest:
                    k, v = rest.split(":", 1)
                    current_item[k.strip()] = _parse_scalar(v)
            elif ":" in line and isinstance(current_top_list, list) and len(current_top_list) == 0:
                k_top = next(reversed(root.keys()))
                if isinstance(root[k_top], list):
                    root[k_top] = {}
                k, v = line.split(":", 1)
                root[k_top][k.strip()] = _parse_scalar(v)
        elif indent == 4 and current_item is not None:
            current_sub_subkey = None
            if ":" in line:
                k, v = line.split(":", 1)
                k, v = k.strip(), v.strip()
                if v:
                    current_item[k] = _parse_scalar(v)
                    current_subkey = None
                    current_subcontainer = None
                else:
                    current_subkey = k
                    current_subcontainer = None
        elif indent == 6 and current_item is not None and current_subkey is not None:
            if line.startswith("- "):
                if not isinstance(current_subcontainer, list):
                    current_subcontainer = []
                    current_item[current_subkey] = current_subcontainer
                current_subcontainer.append(_parse_scalar(line[2:]))
            elif ":" in line:
                if not isinstance(current_subcontainer, dict):
                    current_subcontainer = {}
                    current_item[current_subkey] = current_subcontainer
                k, v = line.split(":", 1)
                k, v = k.strip(), v.strip()
                if v:
                    current_subcontainer[k] = _parse_scalar(v)
                    current_sub_subkey = None
                else:
                    current_sub_subkey = k
                    current_subcontainer[k] = []
        elif (
            indent == 8
            and isinstance(current_subcontainer, dict)
            and current_sub_subkey is not None
        ):
            if line.startswith("- "):
                current_subcontainer[current_sub_subkey].append(_parse_scalar(line[2:]))

    return root


class FixtureConfigLoader:
    """Loader and validator for layers.yaml, nodes.yaml, and models.yaml."""

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
    def bootstrap_in_memory_uow(fixtures_dir: Path) -> InMemoryUnitOfWork:
        """Load layers.yaml, nodes.yaml, and models.yaml into a ready-to-use InMemoryUnitOfWork."""
        uow = InMemoryUnitOfWork()
        layers, nodes, rules = FixtureConfigLoader.load_hierarchy(
            nodes_yaml_path=fixtures_dir / "nodes.yaml",
            layers_yaml_path=fixtures_dir / "layers.yaml",
        )
        models = FixtureConfigLoader.load_models_registry(fixtures_dir / "models.yaml")

        with uow:
            for r in rules:
                uow.layers.save_transfer_rule(r)
            for lyr in layers:
                uow.layers.save_layer(lyr)
            for nd in nodes:
                uow.nodes.save(nd)
            for m in models:
                uow.models.save(m)
            uow.commit()
        return uow


__all__ = ["FixtureConfigLoader", "load_yaml_dict"]
