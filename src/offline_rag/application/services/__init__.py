"""Application Services coordinating domain use cases across Bounded Contexts."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from offline_rag.application.ports import (
    CSVTransferExporterPort,
    EmbeddingModelPort,
    UnitOfWork,
    VectorIndexPort,
)
from offline_rag.application.services.document_ingestion_service import (
    DocumentIngestionResult,
    DocumentIngestionService,
)
from offline_rag.domain.evaluation import JudgeRubric, JudgeScore, JudgeThreshold
from offline_rag.domain.knowledge import (
    KnowledgeChunk,
    KnowledgeItem,
)
from offline_rag.domain.organization import (
    Layer,
    LayerTransferRule,
    Node,
    OrganizationHierarchyValidator,
)
from offline_rag.domain.shared.errors import EntityNotFoundError, HierarchyRuleViolationError
from offline_rag.domain.shared.value_objects import NodeIdentifier, VersionTag
from offline_rag.domain.transfer import TransferDeliveryResolution, TransferPackage


class HierarchyApplicationService:
    """Application service for registering and validating organizational Layers and Nodes."""

    def __init__(self, uow: UnitOfWork) -> None:
        self._uow = uow

    def register_hierarchy(
        self,
        layers: Sequence[Layer],
        nodes: Sequence[Node],
        transfer_rules: Sequence[LayerTransferRule] = (),
    ) -> None:
        """Validate R-01..R-08 and persist all layers, transfer rules, and nodes."""
        OrganizationHierarchyValidator.validate_hierarchy(
            layers=layers, nodes=nodes, transfer_rules=transfer_rules
        )
        with self._uow:
            for rule in transfer_rules:
                self._uow.layers.save_transfer_rule(rule)
            for layer in layers:
                self._uow.layers.save_layer(layer)
            for node in nodes:
                self._uow.nodes.save(node)
            self._uow.commit()


class KnowledgeApplicationService:
    """Application service managing KnowledgeItem registration, versioning, and vector indexing."""

    def __init__(
        self,
        uow: UnitOfWork,
        vector_index: VectorIndexPort | None = None,
        embedding_port: EmbeddingModelPort | None = None,
    ) -> None:
        self._uow = uow
        self._vector_index = vector_index
        self._embedding_port = embedding_port

    def register_knowledge_item(self, item: KnowledgeItem) -> KnowledgeItem:
        """Save a new KnowledgeItem and optionally index its active chunks."""
        with self._uow:
            self._uow.knowledge.save(item)
            self._index_active_chunks(item)
            self._uow.commit()
        return item

    def update_knowledge_item(
        self,
        item_id: str,
        content_hash: str,
        change_reason: str,
        chunks: Iterable[KnowledgeChunk] = (),
        diff_summary: str | None = None,
    ) -> KnowledgeItem:
        """Create a new active version on an existing KnowledgeItem and re-index active chunks."""
        with self._uow:
            item = self._uow.knowledge.get_by_id(item_id)
            if item is None:
                raise EntityNotFoundError(f"KnowledgeItem '{item_id}' not found.")
            item.add_version(
                content_hash=content_hash,
                change_reason=change_reason,
                chunks=chunks,
                diff_summary=diff_summary,
            )
            self._uow.knowledge.save(item)
            self._index_active_chunks(item)
            self._uow.commit()
        return item

    def _index_active_chunks(self, item: KnowledgeItem) -> None:
        if self._vector_index is None or self._embedding_port is None:
            return
        self._vector_index.remove_item_chunks(item.item_id)
        for chunk in item.active_chunks:
            vec = self._embedding_port.embed_text(chunk.text)
            self._vector_index.index_chunk(chunk, vec)


class TransferApplicationService:
    """Application service coordinating cyclic TransferPackage creation, Judge scoring, and upward delivery."""

    def __init__(
        self,
        uow: UnitOfWork,
        csv_exporter: CSVTransferExporterPort | None = None,
    ) -> None:
        self._uow = uow
        self._csv_exporter = csv_exporter

    def prepare_and_judge_package(
        self,
        package_id: str,
        source_node_id: NodeIdentifier,
        reporting_phase: str,
        payload_sections: Mapping[str, Any],
        source_references: Iterable[str],
        criterion_scores: Mapping[str, float],
        judge_explanation: str,
        rubric: JudgeRubric | None = None,
        threshold: JudgeThreshold | None = None,
    ) -> TransferPackage:
        """Assemble a TransferPackage according to Node/Layer rules, score it with Judge, and save."""
        with self._uow:
            node = self._uow.nodes.get_by_id(source_node_id)
            if node is None:
                raise EntityNotFoundError(f"Node '{source_node_id}' not found.")
            if node.transfer is None or node.parent_node_id is None:
                raise HierarchyRuleViolationError(
                    "R-08",
                    f"Node '{source_node_id}' ({node.boundary_type.value}) does not transfer packages upward.",
                )
            rule = self._uow.layers.get_transfer_rule(node.transfer.rule_layer_index)
            if rule is None:
                raise HierarchyRuleViolationError(
                    "R-07",
                    f"Missing transfer rule for target layer {node.transfer.rule_layer_index.value}.",
                )

            prev_accepted = self._uow.transfers.get_latest_accepted_by_node(source_node_id)
            existing = self._uow.transfers.list_by_node(source_node_id)
            next_seq = len(existing) + 1
            version_tag = VersionTag(sequence_number=next_seq)

            effective_required = node.effective_required_fields(rule)
            pkg = TransferPackage(
                package_id=package_id,
                source_node_id=source_node_id,
                target_parent_node_id=node.parent_node_id,
                target_layer_index=rule.target_layer_index,
                detail_level=rule.detail_level,
                version_tag=version_tag,
                reporting_phase=reporting_phase,
                payload_sections=payload_sections,
                source_references=source_references,
                required_fields=effective_required,
                max_length=rule.max_length,
                previous_accepted_package_id=(
                    prev_accepted.package_id if prev_accepted is not None else None
                ),
            )

            judge_score = JudgeScore(
                score_id=f"JUDGE-{package_id}",
                target_artifact_id=package_id,
                criterion_scores=criterion_scores,
                explanation=judge_explanation,
                rubric=rubric or JudgeRubric(),
                threshold=threshold or rule.min_judge_score,
            )
            pkg.apply_judge_score(judge_score)
            self._uow.transfers.save(pkg)
            self._uow.commit()
            return pkg

    def request_latest_deliverable_package(
        self, child_node_id: NodeIdentifier
    ) -> TransferDeliveryResolution:
        """Deliver latest accepted package, or fallback to previous accepted if latest failed Judge (Section 9.2)."""
        latest = self._uow.transfers.get_latest_by_node(child_node_id)
        if latest is None:
            raise EntityNotFoundError(
                f"No TransferPackage found for child node '{child_node_id}'."
            )
        prev_accepted = self._uow.transfers.get_latest_accepted_by_node(child_node_id)
        return latest.resolve_delivery(previous_accepted_package=prev_accepted)

    def export_package_csv(self, package: TransferPackage) -> str:
        """Export a TransferPackage using the configured CSVTransferExporterPort."""
        if self._csv_exporter is None:
            raise RuntimeError("CSVTransferExporterPort is not configured.")
        return self._csv_exporter.export_package(package)


__all__ = [
    "DocumentIngestionResult",
    "DocumentIngestionService",
    "HierarchyApplicationService",
    "KnowledgeApplicationService",
    "TransferApplicationService",
]
