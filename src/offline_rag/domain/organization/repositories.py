"""Repository interfaces for the Organizational Hierarchy Bounded Context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from offline_rag.domain.organization.entities import Layer, LayerTransferRule, Node
from offline_rag.domain.shared.value_objects import LayerIndex, NodeIdentifier


class LayerRepository(ABC):
    """Abstract repository for organizational Layers and LayerTransferRules."""

    @abstractmethod
    def save_layer(self, layer: Layer) -> None:
        """Persist or update a Layer."""

    @abstractmethod
    def get_by_index(self, layer_index: LayerIndex) -> Layer | None:
        """Retrieve a Layer by its LayerIndex."""

    @abstractmethod
    def list_layers(self) -> Sequence[Layer]:
        """Return all registered Layers ordered by LayerIndex."""

    @abstractmethod
    def save_transfer_rule(self, rule: LayerTransferRule) -> None:
        """Persist or update a LayerTransferRule."""

    @abstractmethod
    def get_transfer_rule(self, target_layer_index: LayerIndex) -> LayerTransferRule | None:
        """Retrieve the transfer rule for a target layer."""

    @abstractmethod
    def list_transfer_rules(self) -> Sequence[LayerTransferRule]:
        """Return all registered LayerTransferRules."""


class NodeRepository(ABC):
    """Abstract repository for organizational Nodes."""

    @abstractmethod
    def save(self, node: Node) -> None:
        """Persist or update a Node."""

    @abstractmethod
    def get_by_id(self, node_id: NodeIdentifier) -> Node | None:
        """Retrieve a Node by its NodeIdentifier."""

    @abstractmethod
    def list_all(self) -> Sequence[Node]:
        """Return all registered Nodes."""

    @abstractmethod
    def list_by_layer(self, layer_index: LayerIndex) -> Sequence[Node]:
        """Return all Nodes belonging to a given LayerIndex."""

    @abstractmethod
    def list_children(self, parent_node_id: NodeIdentifier) -> Sequence[Node]:
        """Return direct child Nodes of a given parent Node."""
