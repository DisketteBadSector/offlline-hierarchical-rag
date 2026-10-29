"""Organizational Hierarchy Bounded Context: Layer, Node, Transfer Rules, and R-01..R-08 Validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

from offline_rag.domain.shared.base import AggregateRoot, DomainEvent, Entity
from offline_rag.domain.shared.errors import (
    DomainValidationError,
    HierarchyRuleViolationError,
    InvalidValueObjectError,
)
from offline_rag.domain.shared.value_objects import (
     VALID_CONFIDENTIALITY_LEVELS,
    JudgeThreshold,
    LayerIndex,
    NodeIdentifier,
    TransferDetailLevel,
)


class BoundaryType(str, Enum):
    """Organizational boundary classification of a Node."""

    ROOT = "ROOT"
    INTERMEDIATE = "INTERMEDIATE"
    LEAF = "LEAF"
    KNOWLEDGE_SOURCE = "KNOWLEDGE_SOURCE"


VALID_USAGE_SCOPES: tuple[str, ...] = ("upward_transfer", "intra_layer_or_audit")


@dataclass(frozen=True)
class NodeTransferConfig:
    """Node-specific reference to its target layer transfer rule and extra required fields."""

    rule_layer_index: LayerIndex
    extra_required_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.rule_layer_index, LayerIndex):
            raise InvalidValueObjectError(
                "NodeTransferConfig.rule_layer_index must be a LayerIndex."
            )
        cleaned = tuple(
            f.strip() for f in self.extra_required_fields if isinstance(f, str) and f.strip()
        )
        object.__setattr__(self, "extra_required_fields", cleaned)


@dataclass(frozen=True)
class LayerTransferRule:
    """Reporting rule defined in layers.yaml for a target layer."""

    target_layer_index: LayerIndex
    detail_level: TransferDetailLevel
    required_fields: tuple[str, ...]
    max_length: int
    min_judge_score: JudgeThreshold = field(default_factory=lambda: JudgeThreshold(18.0, 20.0))
    usage_scope: str = "upward_transfer"

    def __post_init__(self) -> None:
        if not isinstance(self.target_layer_index, LayerIndex):
            raise InvalidValueObjectError(
                "LayerTransferRule.target_layer_index must be a LayerIndex."
            )
        if not isinstance(self.detail_level, TransferDetailLevel):
            raise InvalidValueObjectError(
                "LayerTransferRule.detail_level must be a TransferDetailLevel."
            )
        cleaned_fields = tuple(
            f.strip() for f in self.required_fields if isinstance(f, str) and f.strip()
        )
        if not cleaned_fields:
            raise InvalidValueObjectError("LayerTransferRule.required_fields cannot be empty.")
        if isinstance(self.max_length, bool) or not isinstance(self.max_length, int) or self.max_length <= 0:
            raise InvalidValueObjectError("LayerTransferRule.max_length must be a positive integer.")
        if not isinstance(self.min_judge_score, JudgeThreshold):
            raise InvalidValueObjectError(
                "LayerTransferRule.min_judge_score must be a JudgeThreshold."
            )
        if self.usage_scope not in VALID_USAGE_SCOPES:
            raise InvalidValueObjectError(
                f"Invalid usage_scope '{self.usage_scope}'. Allowed: {VALID_USAGE_SCOPES}"
            )
        object.__setattr__(self, "required_fields", cleaned_fields)


@dataclass(frozen=True)
class NodeRegisteredEvent(DomainEvent):
    """Emitted when a new organizational Node is created or registered."""

    node_id: NodeIdentifier
    layer_index: LayerIndex
    boundary_type: BoundaryType


class Layer(Entity):
    """Organizational hierarchy Layer entity (Layers 0..4)."""

    def __init__(
        self,
        layer_index: LayerIndex,
        name: str,
        description: str,
        required: bool = True,
        transfer_rule: LayerTransferRule | None = None,
    ) -> None:
        if not isinstance(layer_index, LayerIndex):
            raise DomainValidationError("Layer.layer_index must be a LayerIndex.")
        if not isinstance(name, str) or not name.strip():
            raise DomainValidationError("Layer.name cannot be empty.")
        if not isinstance(description, str) or not description.strip():
            raise DomainValidationError("Layer.description cannot be empty.")
        if transfer_rule is not None and transfer_rule.target_layer_index != layer_index:
            raise DomainValidationError(
                f"Transfer rule target_layer_index ({transfer_rule.target_layer_index}) "
                f"does not match Layer ({layer_index})."
            )
        super().__init__(layer_index)
        self._layer_index = layer_index
        self._name = name.strip()
        self._description = description.strip()
        self._required = bool(required)
        self._transfer_rule = transfer_rule

    @property
    def layer_index(self) -> LayerIndex:
        return self._layer_index

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def required(self) -> bool:
        return self._required

    @property
    def transfer_rule(self) -> LayerTransferRule | None:
        return self._transfer_rule


class Node(AggregateRoot):
    """Organizational Node Aggregate Root enforcing R-01, R-02, R-05, R-07, and R-08 at construction."""

    def __init__(
        self,
        node_id: NodeIdentifier,
        name: str,
        layer_index: LayerIndex,
        parent_node_id: NodeIdentifier | None,
        activity_domain: str,
        boundary_type: BoundaryType | str,
        confidentiality: str = "internal",
        transfer: NodeTransferConfig | None = None,
    ) -> None:
        if not isinstance(node_id, NodeIdentifier):
            raise DomainValidationError("Node.node_id must be a NodeIdentifier.")
        if not isinstance(name, str) or not name.strip():
            raise DomainValidationError("Node.name cannot be empty.")
        if not isinstance(layer_index, LayerIndex):
            raise HierarchyRuleViolationError("R-01", "Node must have a valid LayerIndex.")

        # R-05: Activity domain is mandatory and non-empty
        if not isinstance(activity_domain, str) or not activity_domain.strip():
            raise HierarchyRuleViolationError(
                "R-05", f"Node '{node_id}' must have a non-empty activity_domain."
            )

        try:
            b_type = (
                boundary_type
                if isinstance(boundary_type, BoundaryType)
                else BoundaryType(str(boundary_type))
            )
        except ValueError as exc:
            raise DomainValidationError(f"Invalid boundary_type: {boundary_type}") from exc

        if confidentiality not in VALID_CONFIDENTIALITY_LEVELS:
            raise DomainValidationError(f"Invalid confidentiality '{confidentiality}'.")

        # R-02: Every node (except ROOT) has exactly one parent node; ROOT has no parent
        if b_type == BoundaryType.ROOT:
            if parent_node_id is not None:
                raise HierarchyRuleViolationError(
                    "R-02", f"ROOT node '{node_id}' cannot have a parent_node_id."
                )
            if layer_index.value != 0:
                raise HierarchyRuleViolationError(
                    "R-01", f"ROOT node '{node_id}' must reside in layer 0."
                )
        else:
            if parent_node_id is None:
                raise HierarchyRuleViolationError(
                    "R-02", f"Non-ROOT node '{node_id}' must have a parent_node_id."
                )
            if not isinstance(parent_node_id, NodeIdentifier):
                raise HierarchyRuleViolationError(
                    "R-02", "parent_node_id must be a valid NodeIdentifier."
                )
            if parent_node_id == node_id:
                raise HierarchyRuleViolationError(
                    "R-02", f"Node '{node_id}' cannot be its own parent."
                )

        # R-08: ROOT and KNOWLEDGE_SOURCE have transfer == None; others must have valid transfer
        if b_type in (BoundaryType.ROOT, BoundaryType.KNOWLEDGE_SOURCE):
            if transfer is not None:
                raise HierarchyRuleViolationError(
                    "R-08",
                    f"Node '{node_id}' with boundaryType={b_type.value} must have transfer=None.",
                )
        else:
            if transfer is None or not isinstance(transfer, NodeTransferConfig):
                raise HierarchyRuleViolationError(
                    "R-08",
                    f"Node '{node_id}' with boundaryType={b_type.value} must have a valid transfer config.",
                )
            # R-07: If transfer is present, ruleLayerIndex must equal node's LayerIndex - 1
            expected_rule_idx = layer_index.value - 1
            if transfer.rule_layer_index.value != expected_rule_idx:
                raise HierarchyRuleViolationError(
                    "R-07",
                    f"Node '{node_id}' at layer {layer_index.value} must have "
                    f"transfer.ruleLayerIndex == {expected_rule_idx}, "
                    f"got {transfer.rule_layer_index.value}.",
                )

        super().__init__(node_id)
        self._node_id = node_id
        self._name = name.strip()
        self._layer_index = layer_index
        self._parent_node_id = parent_node_id
        self._activity_domain = activity_domain.strip()
        self._boundary_type = b_type
        self._confidentiality = confidentiality
        self._transfer = transfer

        self.record_event(
            NodeRegisteredEvent(
                node_id=self._node_id,
                layer_index=self._layer_index,
                boundary_type=self._boundary_type,
            )
        )

    @property
    def node_id(self) -> NodeIdentifier:
        return self._node_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def layer_index(self) -> LayerIndex:
        return self._layer_index

    @property
    def parent_node_id(self) -> NodeIdentifier | None:
        return self._parent_node_id

    @property
    def activity_domain(self) -> str:
        return self._activity_domain

    @property
    def boundary_type(self) -> BoundaryType:
        return self._boundary_type

    @property
    def confidentiality(self) -> str:
        return self._confidentiality

    @property
    def transfer(self) -> NodeTransferConfig | None:
        return self._transfer

    @property
    def is_leaf(self) -> bool:
        return self._boundary_type == BoundaryType.LEAF

    @property
    def is_knowledge_source(self) -> bool:
        return self._boundary_type == BoundaryType.KNOWLEDGE_SOURCE

    def effective_required_fields(self, base_rule: LayerTransferRule) -> tuple[str, ...]:
        """Combine base LayerTransferRule required_fields with node's extra_required_fields."""
        if self._transfer is None:
            return ()
        if base_rule.target_layer_index != self._transfer.rule_layer_index:
            raise HierarchyRuleViolationError(
                "R-07",
                f"Base rule target layer ({base_rule.target_layer_index.value}) does not match "
                f"node '{self._node_id}' rule_layer_index ({self._transfer.rule_layer_index.value}).",
            )
        combined: list[str] = list(base_rule.required_fields)
        for extra in self._transfer.extra_required_fields:
            if extra not in combined:
                combined.append(extra)
        return tuple(combined)


class OrganizationHierarchyValidator:
    """Domain service validating cross-entity rules R-01 through R-08 across Layers, Rules, and Nodes."""

    @staticmethod
    def validate_hierarchy(
        layers: Sequence[Layer],
        nodes: Sequence[Node],
        transfer_rules: Iterable[LayerTransferRule] = (),
    ) -> None:
        """Validate all R-01..R-08 rules across a collection of layers, nodes, and transfer rules."""
        layer_map = {layer.layer_index: layer for layer in layers}
        rule_map = {rule.target_layer_index: rule for rule in transfer_rules}
        for layer in layers:
            if layer.transfer_rule is not None:
                rule_map[layer.transfer_rule.target_layer_index] = layer.transfer_rule

        node_map: dict[NodeIdentifier, Node] = {}
        for node in nodes:
            if node.node_id in node_map:
                raise DomainValidationError(f"Duplicate node_id '{node.node_id}'.")
            node_map[node.node_id] = node

        # R-01: Every node belongs to a valid registered layer
        for node in nodes:
            if node.layer_index not in layer_map:
                raise HierarchyRuleViolationError(
                    "R-01",
                    f"Node '{node.node_id}' references unregistered layer {node.layer_index.value}.",
                )

        # R-02, R-03, R-04, R-07: Cross-node parent-child and transfer rule checks
        for node in nodes:
            if node.boundary_type != BoundaryType.ROOT:
                if node.parent_node_id is None or node.parent_node_id not in node_map:
                    raise HierarchyRuleViolationError(
                        "R-02",
                        f"Node '{node.node_id}' references non-existent parent '{node.parent_node_id}'.",
                    )
                parent = node_map[node.parent_node_id]

                # R-03: Parent's LayerIndex must be strictly less than child's LayerIndex
                if parent.layer_index.value >= node.layer_index.value:
                    raise HierarchyRuleViolationError(
                        "R-03",
                        f"Parent node '{parent.node_id}' (layer {parent.layer_index.value}) "
                        f"must have a smaller LayerIndex than child '{node.node_id}' "
                        f"(layer {node.layer_index.value}).",
                    )

                # R-04: LEAF node cannot have children
                if parent.boundary_type == BoundaryType.LEAF:
                    raise HierarchyRuleViolationError(
                        "R-04",
                        f"LEAF node '{parent.node_id}' cannot be the parent of '{node.node_id}'.",
                    )
                if parent.boundary_type == BoundaryType.KNOWLEDGE_SOURCE:
                    raise HierarchyRuleViolationError(
                        "R-04",
                        f"KNOWLEDGE_SOURCE node '{parent.node_id}' cannot have child '{node.node_id}'.",
                    )

            # R-07: If node has transfer, corresponding rule must exist in layers.yaml
            if node.transfer is not None and rule_map:
                if node.transfer.rule_layer_index not in rule_map:
                    raise HierarchyRuleViolationError(
                        "R-07",
                        f"Node '{node.node_id}' references missing transfer rule for layer "
                        f"{node.transfer.rule_layer_index.value} in layers.yaml.",
                    )

        # R-06: Every layer (except Layer 4) must have at least one node
        nodes_per_layer: dict[LayerIndex, list[Node]] = {lyr.layer_index: [] for lyr in layers}
        for node in nodes:
            nodes_per_layer[node.layer_index].append(node)

        for layer in layers:
            if layer.layer_index.value != 4 and layer.required:
                if not nodes_per_layer.get(layer.layer_index):
                    raise HierarchyRuleViolationError(
                        "R-06",
                        f"Required layer {layer.layer_index.value} ('{layer.name}') must have at least one node.",
                    )
