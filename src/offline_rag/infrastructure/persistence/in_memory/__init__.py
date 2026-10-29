"""In-Memory Repository and Port Test Doubles for Unit Testing."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from typing import Any, Sequence

from offline_rag.application.ports import (
    CSVTransferExporterPort,
    DocumentStoragePort,
    EmbeddingModelPort,
    LLMModelPort,
    LVMModelPort,
    ModelRuntimePort,
    UnitOfWork,
    VectorIndexPort,
)
from offline_rag.domain.executor import Executor, ExecutorRepository
from offline_rag.domain.ike_model import IKERelationship, IKERepository
from offline_rag.domain.issue import Issue, IssueRepository
from offline_rag.domain.knowledge import (
    KnowledgeChunk,
    KnowledgeItem,
    KnowledgeRepository,
)
from offline_rag.domain.model_prompt import (
    ModelRegistryRepository,
    ModelRole,
    OfflineModelSpec,
    PromptTaskCategory,
    PromptTemplate,
    PromptTemplateRepository,
)
from offline_rag.domain.organization import (
    Layer,
    LayerRepository,
    LayerTransferRule,
    Node,
    NodeRepository,
)
from offline_rag.domain.rag import RetrievalQuery, RetrievedChunk
from offline_rag.domain.rfp import FeedbackReport, Proposal, RFP, RFPRepository
from offline_rag.domain.shared.errors import EntityNotFoundError
from offline_rag.domain.shared.value_objects import (
    CONFIDENTIALITY_RANK,
    LayerIndex,
    NodeIdentifier,
    SimilarityScore,
    VersionTag,
)
from offline_rag.domain.transfer import (
    TransferPackage,
    TransferPackageRepository,
)


class InMemoryLayerRepository(LayerRepository):
    """In-memory Test Double for LayerRepository."""

    def __init__(self) -> None:
        self._layers: dict[LayerIndex, Layer] = {}
        self._rules: dict[LayerIndex, LayerTransferRule] = {}

    def save_layer(self, layer: Layer) -> None:
        self._layers[layer.layer_index] = layer
        if layer.transfer_rule is not None:
            self._rules[layer.transfer_rule.target_layer_index] = layer.transfer_rule

    def get_by_index(self, layer_index: LayerIndex) -> Layer | None:
        return self._layers.get(layer_index)

    def list_layers(self) -> Sequence[Layer]:
        return tuple(sorted(self._layers.values(), key=lambda lyr: lyr.layer_index.value))

    def save_transfer_rule(self, rule: LayerTransferRule) -> None:
        self._rules[rule.target_layer_index] = rule

    def get_transfer_rule(self, target_layer_index: LayerIndex) -> LayerTransferRule | None:
        return self._rules.get(target_layer_index)

    def list_transfer_rules(self) -> Sequence[LayerTransferRule]:
        return tuple(
            sorted(self._rules.values(), key=lambda r: r.target_layer_index.value)
        )


class InMemoryNodeRepository(NodeRepository):
    """In-memory Test Double for NodeRepository."""

    def __init__(self) -> None:
        self._nodes: dict[NodeIdentifier, Node] = {}

    def save(self, node: Node) -> None:
        self._nodes[node.node_id] = node

    def get_by_id(self, node_id: NodeIdentifier) -> Node | None:
        return self._nodes.get(node_id)

    def list_all(self) -> Sequence[Node]:
        return tuple(self._nodes.values())

    def list_by_layer(self, layer_index: LayerIndex) -> Sequence[Node]:
        return tuple(n for n in self._nodes.values() if n.layer_index == layer_index)

    def list_children(self, parent_node_id: NodeIdentifier) -> Sequence[Node]:
        return tuple(n for n in self._nodes.values() if n.parent_node_id == parent_node_id)


class InMemoryKnowledgeRepository(KnowledgeRepository):
    """In-memory Test Double for KnowledgeRepository."""

    def __init__(self) -> None:
        self._items: dict[str, KnowledgeItem] = {}

    def save(self, item: KnowledgeItem) -> None:
        self._items[item.item_id] = item

    def get_by_id(self, item_id: str) -> KnowledgeItem | None:
        return self._items.get(item_id)

    def list_by_node(self, node_id: NodeIdentifier) -> Sequence[KnowledgeItem]:
        return tuple(
            item for item in self._items.values() if item.scope.owner_node_id == node_id
        )

    def find_by_active_content_hash(self, content_hash: str) -> Sequence[KnowledgeItem]:
        return tuple(
            item
            for item in self._items.values()
            if item.active_version.content_hash == content_hash
        )

    def list_active_chunks_by_node(self, node_id: NodeIdentifier) -> Sequence[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        for item in self.list_by_node(node_id):
            chunks.extend(item.active_chunks)
        return tuple(chunks)


class InMemoryIssueRepository(IssueRepository):
    """In-memory Test Double for IssueRepository."""

    def __init__(self) -> None:
        self._issues: dict[str, Issue] = {}

    def save(self, issue: Issue) -> None:
        self._issues[issue.issue_id] = issue

    def get_by_id(self, issue_id: str) -> Issue | None:
        return self._issues.get(issue_id)

    def list_by_node(self, node_id: NodeIdentifier) -> Sequence[Issue]:
        return tuple(iss for iss in self._issues.values() if iss.node_id == node_id)


class InMemoryExecutorRepository(ExecutorRepository):
    """In-memory Test Double for ExecutorRepository."""

    def __init__(self) -> None:
        self._executors: dict[str, Executor] = {}

    def save(self, executor: Executor) -> None:
        self._executors[executor.executor_id] = executor

    def get_by_id(self, executor_id: str) -> Executor | None:
        return self._executors.get(executor_id)

    def list_by_node(self, node_id: NodeIdentifier) -> Sequence[Executor]:
        return tuple(ex for ex in self._executors.values() if ex.node_id == node_id)

    def find_by_expertise(
        self, domain_keyword: str, node_id: NodeIdentifier | None = None
    ) -> Sequence[Executor]:
        candidates = (
            self.list_by_node(node_id) if node_id is not None else tuple(self._executors.values())
        )
        return tuple(ex for ex in candidates if ex.matches_expertise(domain_keyword))


class InMemoryIKERepository(IKERepository):
    """In-memory Test Double for IKERepository."""

    def __init__(self) -> None:
        self._relationships: dict[str, IKERelationship] = {}

    def save(self, relationship: IKERelationship) -> None:
        self._relationships[relationship.relationship_id] = relationship

    def get_by_id(self, relationship_id: str) -> IKERelationship | None:
        return self._relationships.get(relationship_id)

    def list_by_node(self, node_id: NodeIdentifier) -> Sequence[IKERelationship]:
        return tuple(
            rel for rel in self._relationships.values() if rel.node_id == node_id
        )

    def list_by_source(self, source_entity_id: str) -> Sequence[IKERelationship]:
        return tuple(
            rel
            for rel in self._relationships.values()
            if rel.source_entity_id == source_entity_id
        )


class InMemoryTransferPackageRepository(TransferPackageRepository):
    """In-memory Test Double for TransferPackageRepository."""

    def __init__(self) -> None:
        self._packages: dict[str, TransferPackage] = {}

    def save(self, package: TransferPackage) -> None:
        self._packages[package.package_id] = package

    def get_by_id(self, package_id: str) -> TransferPackage | None:
        return self._packages.get(package_id)

    def list_by_node(self, source_node_id: NodeIdentifier) -> Sequence[TransferPackage]:
        pkgs = [p for p in self._packages.values() if p.source_node_id == source_node_id]
        return tuple(sorted(pkgs, key=lambda p: p.version_tag.sequence_number))

    def get_latest_by_node(self, source_node_id: NodeIdentifier) -> TransferPackage | None:
        pkgs = self.list_by_node(source_node_id)
        return pkgs[-1] if pkgs else None

    def get_latest_accepted_by_node(self, source_node_id: NodeIdentifier) -> TransferPackage | None:
        accepted = [p for p in self.list_by_node(source_node_id) if p.is_accepted]
        return accepted[-1] if accepted else None


class InMemoryRFPRepository(RFPRepository):
    """In-memory Test Double for RFPRepository."""

    def __init__(self) -> None:
        self._rfps: dict[str, RFP] = {}
        self._proposals: dict[str, Proposal] = {}
        self._feedback_by_proposal: dict[str, FeedbackReport] = {}

    def save_rfp(self, rfp: RFP) -> None:
        self._rfps[rfp.rfp_id] = rfp

    def get_rfp_by_id(self, rfp_id: str) -> RFP | None:
        return self._rfps.get(rfp_id)

    def list_rfps_by_node(self, node_id: NodeIdentifier) -> Sequence[RFP]:
        return tuple(r for r in self._rfps.values() if r.node_id == node_id)

    def save_proposal(self, proposal: Proposal) -> None:
        self._proposals[proposal.proposal_id] = proposal

    def get_proposal_by_id(self, proposal_id: str) -> Proposal | None:
        return self._proposals.get(proposal_id)

    def list_proposals_by_rfp(self, rfp_id: str) -> Sequence[Proposal]:
        return tuple(p for p in self._proposals.values() if p.rfp_id == rfp_id)

    def save_feedback_report(self, report: FeedbackReport) -> None:
        self._feedback_by_proposal[report.proposal_id] = report

    def get_feedback_by_proposal(self, proposal_id: str) -> FeedbackReport | None:
        return self._feedback_by_proposal.get(proposal_id)


class InMemoryModelRegistryRepository(ModelRegistryRepository):
    """In-memory Test Double for ModelRegistryRepository."""

    def __init__(self) -> None:
        self._models: dict[str, OfflineModelSpec] = {}

    def save(self, model_spec: OfflineModelSpec) -> None:
        self._models[model_spec.model_id] = model_spec

    def get_by_id(self, model_id: str) -> OfflineModelSpec | None:
        return self._models.get(model_id)

    def get_by_role(self, role: ModelRole | str) -> OfflineModelSpec | None:
        target = role if isinstance(role, ModelRole) else ModelRole(str(role).lower())
        for m in self._models.values():
            if m.role == target:
                return m
        return None

    def list_all(self) -> Sequence[OfflineModelSpec]:
        return tuple(self._models.values())


class InMemoryPromptTemplateRepository(PromptTemplateRepository):
    """In-memory Test Double for PromptTemplateRepository."""

    def __init__(self) -> None:
        self._templates: dict[tuple[str, str], PromptTemplate] = {}
        self._active_by_id: dict[str, PromptTemplate] = {}

    def save(self, template: PromptTemplate) -> None:
        self._templates[(template.template_id, template.version_tag.tag)] = template
        for hist in template.history:
            snap = template.get_version_snapshot(hist.version)
            if snap is not None:
                self._templates[(template.template_id, hist.version.tag)] = snap
        existing = self._active_by_id.get(template.template_id)
        if existing is None or template.version_tag.sequence_number >= existing.version_tag.sequence_number:
            self._active_by_id[template.template_id] = template

    def get_active_by_id(self, prompt_id: str) -> PromptTemplate | None:
        return self._active_by_id.get(prompt_id)

    def get_by_id_and_version(
        self, template_id: str, version_tag: VersionTag | str
    ) -> PromptTemplate | None:
        tag = version_tag.tag if isinstance(version_tag, VersionTag) else str(version_tag).strip()
        direct = self._templates.get((template_id, tag))
        if direct is not None:
            return direct
        active = self._active_by_id.get(template_id)
        if active is not None:
            return active.get_version_snapshot(tag)
        return None

    def list_by_task(self, task_category: PromptTaskCategory | str) -> Sequence[PromptTemplate]:
        cat = (
            task_category
            if isinstance(task_category, PromptTaskCategory)
            else PromptTaskCategory(str(task_category))
        )
        matched = [t for t in self._templates.values() if t.task_category == cat]
        return tuple(sorted(matched, key=lambda t: t.version_tag.sequence_number))

    def get_latest_by_task(self, task_category: PromptTaskCategory | str) -> PromptTemplate | None:
        cat = (
            task_category
            if isinstance(task_category, PromptTaskCategory)
            else PromptTaskCategory(str(task_category))
        )
        for active_tpl in self._active_by_id.values():
            if active_tpl.task_category == cat:
                return active_tpl
        items = self.list_by_task(cat)
        return items[-1] if items else None


class InMemoryUnitOfWork(UnitOfWork):
    """In-memory UnitOfWork aggregating all 10 In-Memory repositories."""

    def __init__(self) -> None:
        self.layers = InMemoryLayerRepository()
        self.nodes = InMemoryNodeRepository()
        self.knowledge = InMemoryKnowledgeRepository()
        self.issues = InMemoryIssueRepository()
        self.executors = InMemoryExecutorRepository()
        self.ike = InMemoryIKERepository()
        self.transfers = InMemoryTransferPackageRepository()
        self.rfps = InMemoryRFPRepository()
        self.models = InMemoryModelRegistryRepository()
        self.prompts = InMemoryPromptTemplateRepository()
        self.committed: bool = False
        self.rolled_back: bool = False

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


class InMemoryVectorIndex(VectorIndexPort):
    """In-memory VectorIndexPort implementation with cosine similarity search."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[KnowledgeChunk, tuple[float, ...]]] = {}

    def index_chunk(self, chunk: KnowledgeChunk, vector: Sequence[float]) -> None:
        self._entries[chunk.chunk_id] = (chunk, tuple(float(v) for v in vector))

    def remove_item_chunks(self, source_item_id: str) -> None:
        to_delete = [
            cid
            for cid, (ch, _) in self._entries.items()
            if ch.source_item_id == source_item_id
        ]
        for cid in to_delete:
            del self._entries[cid]

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return max(0.0, min(1.0, dot / (na * nb)))

    def search(
        self, query: RetrievalQuery, query_vector: Sequence[float]
    ) -> Sequence[RetrievedChunk]:
        max_conf_rank = CONFIDENTIALITY_RANK[query.max_confidentiality]
        results: list[RetrievedChunk] = []
        for chunk, vec in self._entries.values():
            meta = chunk.inherited_metadata
            if meta.node_id != query.requesting_node_id:
                continue
            if CONFIDENTIALITY_RANK[meta.confidentiality] > max_conf_rank:
                continue
            if query.domain_filter and meta.domain != query.domain_filter:
                continue
            if query.metadata_filters and not meta.matches_filters(query.metadata_filters):
                continue

            sim_val = self._cosine(query_vector, vec)
            sim = SimilarityScore(sim_val)
            if sim.meets_threshold(query.min_similarity):
                results.append(RetrievedChunk(chunk=chunk, similarity=sim))

        results.sort(key=lambda r: r.similarity.value, reverse=True)
        return tuple(results[: query.top_k])


class InMemoryDocumentStorage(DocumentStoragePort):
    """In-memory DocumentStoragePort Test Double."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def save_document(self, document_key: str, raw_bytes: bytes) -> str:
        self._store[document_key] = bytes(raw_bytes)
        return f"memory://{document_key}"

    def load_document(self, document_key: str) -> bytes:
        if document_key not in self._store:
            raise EntityNotFoundError(f"Document '{document_key}' not found in storage.")
        return self._store[document_key]

    def exists(self, document_key: str) -> bool:
        return document_key in self._store


class InMemoryCSVTransferExporter(CSVTransferExporterPort):
    """Deterministic CSV exporter for TransferPackages."""

    COLUMNS: tuple[str, ...] = (
        "package_id",
        "source_node_id",
        "target_parent_node_id",
        "target_layer_index",
        "detail_level",
        "version_tag",
        "reporting_phase",
        "status",
        "judge_score",
        "source_references",
        "payload_json",
    )

    def export_package(self, package: TransferPackage) -> str:
        return self.export_packages([package])

    def export_packages(self, packages: Sequence[TransferPackage]) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(self.COLUMNS)
        for pkg in packages:
            score_val = f"{pkg.judge_score.total_score:.2f}" if pkg.judge_score else ""
            writer.writerow(
                [
                    pkg.package_id,
                    str(pkg.source_node_id),
                    str(pkg.target_parent_node_id),
                    pkg.target_layer_index.value,
                    str(pkg.detail_level),
                    str(pkg.version_tag),
                    pkg.reporting_phase,
                    pkg.status.value,
                    score_val,
                    ";".join(pkg.source_references),
                    json.dumps(dict(pkg.payload_sections), ensure_ascii=False, sort_keys=True),
                ]
            )
        return buf.getvalue()


class DeterministicMockEmbeddingAdapter(EmbeddingModelPort):
    """Deterministic Mock Embedding Adapter for Unit Tests (no GPU/network needed)."""

    def __init__(self, dimension: int = 1024) -> None:
        self._dimension = dimension

    def embed_text(self, text: str) -> tuple[float, ...]:
        tokens = [t.strip(".,،؛:؟!()[]{}") for t in text.split() if t.strip()]
        vec = [0.0] * self._dimension
        if not tokens:
            vec[0] = 1.0
            return tuple(vec)
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            for i in range(0, len(digest), 2):
                idx = ((digest[i] << 8) | digest[i + 1]) % self._dimension
                vec[idx] += 1.0 if (digest[i] & 1) == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return tuple(round(v / norm, 6) for v in vec)

    def embed_batch(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self.embed_text(t) for t in texts)


class DeterministicMockLLMAdapter(LLMModelPort):
    """Deterministic Mock LLM Adapter for Unit Tests."""

    def __init__(self, canned_response: str | None = None) -> None:
        self._canned_response = canned_response
        self.call_history: list[dict[str, Any]] = []

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        self.call_history.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if self._canned_response is not None:
            return self._canned_response
        return json.dumps(
            {
                "answer": "پاسخ نمونه مبتنی بر دانش گره.",
                "source_references": ["DOC-SAMPLE@v1:c0"],
                "uncertainty_declared": False,
            },
            ensure_ascii=False,
        )


class DeterministicMockLVMAdapter(LVMModelPort):
    """Deterministic Mock LVM Adapter for Unit Tests."""

    def analyze_page_layout(self, document_id: str, page_number: int) -> dict[str, Any]:
        return {
            "document_id": document_id,
            "page_number": page_number,
            "detected_elements": [
                {"type": "header", "text": "گزارش فنی"},
                {"type": "table", "rows": 3, "columns": 4},
            ],
            "tables_count": 1,
            "ocr_confidence": 0.95,
        }


class InMemoryModelRuntime(ModelRuntimePort):
    """Spy/Test Double for sequential model runtime loading on constrained VRAM."""

    def __init__(self, max_concurrent_models: int = 1) -> None:
        self._max_concurrent = max_concurrent_models
        self._loaded: list[str] = []
        self.operations_log: list[tuple[str, str]] = []

    def load_model(self, model_id: str) -> None:
        while len(self._loaded) >= self._max_concurrent:
            evicted = self._loaded.pop(0)
            self.operations_log.append(("unload", evicted))
        if model_id not in self._loaded:
            self._loaded.append(model_id)
            self.operations_log.append(("load", model_id))

    def unload_model(self, model_id: str) -> None:
        if model_id in self._loaded:
            self._loaded.remove(model_id)
            self.operations_log.append(("unload", model_id))

    def get_loaded_models(self) -> tuple[str, ...]:
        return tuple(self._loaded)


__all__ = [
    "DeterministicMockEmbeddingAdapter",
    "DeterministicMockLLMAdapter",
    "DeterministicMockLVMAdapter",
    "InMemoryCSVTransferExporter",
    "InMemoryDocumentStorage",
    "InMemoryExecutorRepository",
    "InMemoryIKERepository",
    "InMemoryIssueRepository",
    "InMemoryKnowledgeRepository",
    "InMemoryLayerRepository",
    "InMemoryModelRegistryRepository",
    "InMemoryModelRuntime",
    "InMemoryNodeRepository",
    "InMemoryPromptTemplateRepository",
    "InMemoryRFPRepository",
    "InMemoryTransferPackageRepository",
    "InMemoryUnitOfWork",
    "InMemoryVectorIndex",
]
