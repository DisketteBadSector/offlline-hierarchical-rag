"""Unit tests verifying Phase 1 and Phase 2 fixtures and baseline scripts."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts" / "test") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "test"))

from benchmark_offline_models import (  # noqa: E402
    REQUIRED_MODEL_FIELDS,
    load_and_validate_models_registry,
    run_all_benchmarks,
)
from test_llm import load_yaml_file  # noqa: E402


def test_layers_yaml_resolves_layer3_usage_scope() -> None:
    """Verify targetLayerIndex: 3 has usageScope='intra_layer_or_audit' in layers.yaml."""
    layers_path = REPO_ROOT / "data" / "fixtures" / "layers.yaml"
    data = load_yaml_file(layers_path)
    rules = {rule["targetLayerIndex"]: rule for rule in data["layer_transfer_rules"]}

    assert set(rules.keys()) == {0, 1, 2, 3}
    assert rules[3]["detailLevel"] == "technical_detailed"
    assert rules[3]["usageScope"] == "intra_layer_or_audit"


def test_nodes_yaml_transfer_rules_consistent_with_layers_yaml() -> None:
    """Verify R-07 and R-08 consistency between nodes.yaml and layers.yaml."""
    nodes_data = load_yaml_file(REPO_ROOT / "data" / "fixtures" / "nodes.yaml")
    layers_data = load_yaml_file(REPO_ROOT / "data" / "fixtures" / "layers.yaml")
    valid_rule_indices = {rule["targetLayerIndex"] for rule in layers_data["layer_transfer_rules"]}

    for node in nodes_data["nodes"]:
        boundary = node["boundaryType"]
        transfer = node.get("transfer")
        if boundary in {"ROOT", "KNOWLEDGE_SOURCE"}:
            assert transfer is None
        else:
            assert isinstance(transfer, dict)
            expected_rule_idx = int(node["layerIndex"]) - 1
            assert transfer["ruleLayerIndex"] == expected_rule_idx
            assert transfer["ruleLayerIndex"] in valid_rule_indices


def test_models_yaml_contains_all_roles_and_required_fields() -> None:
    """Verify models.yaml registers embedding, llm, and lvm with all required fields."""
    by_role = load_and_validate_models_registry(REPO_ROOT / "data" / "fixtures" / "models.yaml")
    assert set(by_role.keys()) == {"embedding", "llm", "lvm"}

    for role, model_cfg in by_role.items():
        assert REQUIRED_MODEL_FIELDS.issubset(model_cfg.keys()), f"Missing fields in {role}"
        assert isinstance(model_cfg["supportedTasks"], list) and len(model_cfg["supportedTasks"]) > 0
        assert isinstance(model_cfg["hardwareConstraints"], dict)

    assert len(by_role["llm"]["supportedTasks"]) == 12


def test_benchmark_offline_models_runs_in_fallback_mode() -> None:
    """Verify benchmark_offline_models executes cleanly with offline fallback."""
    report = run_all_benchmarks(
        config_path=REPO_ROOT / "data" / "fixtures" / "models.yaml",
        runs=2,
        force_fallback=True,
    )
    results = report["results"]
    assert results["embedding"]["vectorDimension"] == 1024
    assert results["embedding"]["semanticDiscriminationPassed"] is True
    assert results["llm"]["jsonStabilityRate"] == 1.0
    assert results["lvm"]["layoutSchemaValid"] is True
