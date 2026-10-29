"""Unit tests for Organizational Hierarchy (R-01 to R-08) and YAML Fixture loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from offline_rag.domain.organization import (
    BoundaryType,
    Layer,
    LayerTransferRule,
    Node,
    NodeTransferConfig,
    OrganizationHierarchyValidator,
)
from offline_rag.domain.shared import (
    HierarchyRuleViolationError,
    LayerIndex,
    NodeIdentifier,
    TransferDetailLevel,
)
from offline_rag.infrastructure.config import FixtureConfigLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO_ROOT / "data" / "fixtures"


def test_load_real_nodes_and_layers_fixtures_passes_r01_to_r08() -> None:
    """Directly load data/fixtures/nodes.yaml and layers.yaml and validate R-01..R-08."""
    layers, nodes, rules = FixtureConfigLoader.load_hierarchy(
        nodes_yaml_path=FIXTURES_DIR / "nodes.yaml",
        layers_yaml_path=FIXTURES_DIR / "layers.yaml",
    )
    assert len(layers) == 5
    assert len(nodes) == 6
    assert len(rules) == 4

    rule_by_idx = {r.target_layer_index.value: r for r in rules}
    assert rule_by_idx[3].usage_scope == "intra_layer_or_audit"
    assert rule_by_idx[0].usage_scope == "upward_transfer"

    node_by_id = {str(n.node_id): n for n in nodes}
    n300 = node_by_id["N-300"]
    eff_fields = n300.effective_required_fields(rule_by_idx[2])
    assert "test_notes" in eff_fields
    assert "methods" in eff_fields


def test_rule_r01_node_must_have_valid_registered_layer() -> None:
    layers, nodes, rules = FixtureConfigLoader.load_hierarchy(
        FIXTURES_DIR / "nodes.yaml", FIXTURES_DIR / "layers.yaml"
    )
    # Remove layer 3 from registered layers -> violates R-01 for N-300 and N-301
    incomplete_layers = [lyr for lyr in layers if lyr.layer_index.value != 3]
    with pytest.raises(HierarchyRuleViolationError) as exc_info:
        OrganizationHierarchyValidator.validate_hierarchy(incomplete_layers, nodes, rules)
    assert exc_info.value.rule_id == "R-01"


def test_rule_r02_every_non_root_node_must_have_valid_parent() -> None:
    with pytest.raises(HierarchyRuleViolationError) as exc_info:
        Node(
            node_id=NodeIdentifier("N-100"),
            name="سازمان بدون والد",
            layer_index=LayerIndex(1),
            parent_node_id=None,
            activity_domain="فناوری",
            boundary_type=BoundaryType.INTERMEDIATE,
            transfer=NodeTransferConfig(rule_layer_index=LayerIndex(0)),
        )
    assert exc_info.value.rule_id == "R-02"


def test_rule_r03_parent_layer_index_must_be_smaller_than_child() -> None:
    layers, nodes, rules = FixtureConfigLoader.load_hierarchy(
        FIXTURES_DIR / "nodes.yaml", FIXTURES_DIR / "layers.yaml"
    )
    # Child at layer 1 pointing to parent N-200 at layer 2 -> violates R-03
    bad_child = Node(
        node_id=NodeIdentifier("N-199"),
        name="گره نامعتبر",
        layer_index=LayerIndex(1),
        parent_node_id=NodeIdentifier("N-200"),
        activity_domain="فناوری",
        boundary_type=BoundaryType.INTERMEDIATE,
        transfer=NodeTransferConfig(rule_layer_index=LayerIndex(0)),
    )
    with pytest.raises(HierarchyRuleViolationError) as exc_info:
        OrganizationHierarchyValidator.validate_hierarchy(layers, [*nodes, bad_child], rules)
    assert exc_info.value.rule_id == "R-03"


def test_rule_r04_leaf_node_cannot_have_children() -> None:
    layers, nodes, rules = FixtureConfigLoader.load_hierarchy(
        FIXTURES_DIR / "nodes.yaml", FIXTURES_DIR / "layers.yaml"
    )
    # N-301 is a LEAF node at layer 3; adding a child at layer 4 pointing to N-301 violates R-04
    child_of_leaf = Node(
        node_id=NodeIdentifier("N-401"),
        name="زیرمجموعه گره برگ",
        layer_index=LayerIndex(4),
        parent_node_id=NodeIdentifier("N-301"),
        activity_domain="آرشیو",
        boundary_type=BoundaryType.KNOWLEDGE_SOURCE,
        transfer=None,
    )
    with pytest.raises(HierarchyRuleViolationError) as exc_info:
        OrganizationHierarchyValidator.validate_hierarchy(layers, [*nodes, child_of_leaf], rules)
    assert exc_info.value.rule_id == "R-04"


def test_rule_r05_activity_domain_is_mandatory_and_non_empty() -> None:
    with pytest.raises(HierarchyRuleViolationError) as exc_info:
        Node(
            node_id=NodeIdentifier("N-000"),
            name="ریشه",
            layer_index=LayerIndex(0),
            parent_node_id=None,
            activity_domain="   ",
            boundary_type=BoundaryType.ROOT,
            transfer=None,
        )
    assert exc_info.value.rule_id == "R-05"


def test_rule_r06_every_required_layer_0_to_3_must_have_at_least_one_node() -> None:
    layers, nodes, rules = FixtureConfigLoader.load_hierarchy(
        FIXTURES_DIR / "nodes.yaml", FIXTURES_DIR / "layers.yaml"
    )
    # Remove all layer 3 nodes (so layer 2 has no children and layer 3 is empty) -> violates R-06
    nodes_without_layer3 = [n for n in nodes if n.layer_index.value != 3]
    with pytest.raises(HierarchyRuleViolationError) as exc_info:
        OrganizationHierarchyValidator.validate_hierarchy(layers, nodes_without_layer3, rules)
    assert exc_info.value.rule_id == "R-06"


def test_rule_r07_transfer_rule_layer_index_must_equal_layer_minus_one() -> None:
    with pytest.raises(HierarchyRuleViolationError) as exc_info:
        Node(
            node_id=NodeIdentifier("N-300"),
            name="تیم پروژه",
            layer_index=LayerIndex(3),
            parent_node_id=NodeIdentifier("N-200"),
            activity_domain="RAG",
            boundary_type=BoundaryType.INTERMEDIATE,
            transfer=NodeTransferConfig(rule_layer_index=LayerIndex(1)),  # Should be 2
        )
    assert exc_info.value.rule_id == "R-07"


def test_rule_r08_root_and_knowledge_source_transfer_null_others_mandatory() -> None:
    # ROOT with non-null transfer violates R-08
    with pytest.raises(HierarchyRuleViolationError) as exc_info:
        Node(
            node_id=NodeIdentifier("N-000"),
            name="ریشه",
            layer_index=LayerIndex(0),
            parent_node_id=None,
            activity_domain="راهبری",
            boundary_type=BoundaryType.ROOT,
            transfer=NodeTransferConfig(rule_layer_index=LayerIndex(0)),
        )
    assert exc_info.value.rule_id == "R-08"

    # INTERMEDIATE with transfer=None violates R-08
    with pytest.raises(HierarchyRuleViolationError) as exc_info2:
        Node(
            node_id=NodeIdentifier("N-200"),
            name="مرکز پژوهشی",
            layer_index=LayerIndex(2),
            parent_node_id=NodeIdentifier("N-100"),
            activity_domain="پژوهش",
            boundary_type=BoundaryType.INTERMEDIATE,
            transfer=None,
        )
    assert exc_info2.value.rule_id == "R-08"
