"""Application Layer Ports: Storage Abstractions, UnitOfWork, Extraction Ports, and Offline Model Ports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Any, Sequence

from offline_rag.domain.executor import ExecutorRepository
from offline_rag.domain.ike_model import IKERepository
from offline_rag.domain.issue import IssueRepository
from offline_rag.domain.knowledge import (
    DocumentFormat,
    ExtractedDocumentContent,
    KnowledgeChunk,
    KnowledgeRepository,
)
from offline_rag.domain.model_prompt import (
    ModelRegistryRepository,
    PromptTemplateRepository,
)
from offline_rag.domain.organization import LayerRepository, NodeRepository
from offline_rag.domain.rag import RetrievalQuery, RetrievedChunk
from offline_rag.domain.rfp import RFPRepository
from offline_rag.domain.transfer import TransferPackage, TransferPackageRepository


class UnitOfWork(ABC):
    """Transactional boundary aggregating all 10 domain repositories."""

    layers: LayerRepository
    nodes: NodeRepository
    knowledge: KnowledgeRepository
    issues: IssueRepository
    executors: ExecutorRepository
    ike: IKERepository
    transfers: TransferPackageRepository
    rfps: RFPRepository
    models: ModelRegistryRepository
    prompts: PromptTemplateRepository

    def __enter__(self) -> UnitOfWork:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self.rollback()

    @abstractmethod
    def commit(self) -> None:
        """Commit pending changes in the unit of work."""

    @abstractmethod
    def rollback(self) -> None:
        """Roll back uncommitted changes."""


class DocumentExtractorPort(ABC):
    """Port for extracting structured text, headings, tables, and page numbers from DOC/DOCX/PDF."""

    @property
    @abstractmethod
    def supported_format(self) -> DocumentFormat:
        """Return the DocumentFormat handled by this extractor."""

    @abstractmethod
    def extract(
        self, file_path: Path, document_id: str | None = None
    ) -> ExtractedDocumentContent:
        """Extract structured content from a local document file."""

    @abstractmethod
    def extract_from_bytes(
        self, raw_bytes: bytes, document_id: str
    ) -> ExtractedDocumentContent:
        """Extract structured content from raw document bytes."""


class LvmLayoutAssistantPort(ABC):
    """Port for local LVM layout & visual interpretation of flagged documents/pages (Section 7.3)."""

    @abstractmethod
    def describe_visual_page(
        self, document_id: str, version_id: str, page_number: int
    ) -> str:
        """Generate structured visual/layout description for a flagged page."""

    @abstractmethod
    def enrich_extracted_content(
        self, extracted: ExtractedDocumentContent, version_id: str = "v1"
    ) -> ExtractedDocumentContent:
        """Attach visual layout interpretations to an ExtractedDocumentContent."""


class VectorIndexPort(ABC):
    """Port for local vector indexing and similarity search over KnowledgeChunks."""

    @abstractmethod
    def index_chunk(self, chunk: KnowledgeChunk, vector: Sequence[float]) -> None:
        """Store or update the embedding vector for a KnowledgeChunk."""

    @abstractmethod
    def search(
        self, query: RetrievalQuery, query_vector: Sequence[float]
    ) -> Sequence[RetrievedChunk]:
        """Search indexed chunks respecting query node_id, confidentiality, and top_k."""

    @abstractmethod
    def remove_item_chunks(self, source_item_id: str) -> None:
        """Remove all indexed chunks belonging to a KnowledgeItem."""


class DocumentStoragePort(ABC):
    """Port for local filesystem storage of raw documents (DOC, DOCX, PDF) and artifacts."""

    @abstractmethod
    def save_document(self, document_key: str, raw_bytes: bytes) -> str:
        """Store raw document bytes and return local storage URI/path."""

    @abstractmethod
    def load_document(self, document_key: str) -> bytes:
        """Load raw document bytes by key."""

    @abstractmethod
    def exists(self, document_key: str) -> bool:
        """Return True if the document exists in storage."""


class CSVTransferExporterPort(ABC):
    """Port for exporting TransferPackages to CSV format for offline exchange (Section 9.3)."""

    @abstractmethod
    def export_package(self, package: TransferPackage) -> str:
        """Serialize a single TransferPackage to CSV content."""

    @abstractmethod
    def export_packages(self, packages: Sequence[TransferPackage]) -> str:
        """Serialize multiple TransferPackages to CSV content."""


class EmbeddingModelPort(ABC):
    """Port for local offline Embedding generation (e.g. BGE-M3)."""

    @abstractmethod
    def embed_text(self, text: str) -> tuple[float, ...]:
        """Generate normalized embedding vector for a single text."""

    @abstractmethod
    def embed_batch(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """Generate normalized embedding vectors for a batch of texts."""


class LLMModelPort(ABC):
    """Port for local offline Large Language Model inference (e.g. Diba-Base)."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Generate text or structured JSON response from the local LLM."""


class LVMModelPort(ABC):
    """Port for local offline Vision-Language Model layout & OCR extraction (e.g. GLM-OCR)."""

    @abstractmethod
    def analyze_page_layout(self, document_id: str, page_number: int) -> dict[str, Any]:
        """Extract layout structure, tables, and OCR text for a document page."""


class ModelRuntimePort(ABC):
    """Port managing sequential model load/unload lifecycle on VRAM-constrained hardware (ADR-P02-005)."""

    @abstractmethod
    def load_model(self, model_id: str) -> None:
        """Load the specified model into memory (unloading any conflicting active model)."""

    @abstractmethod
    def unload_model(self, model_id: str) -> None:
        """Unload the specified model from memory."""

    @abstractmethod
    def get_loaded_models(self) -> tuple[str, ...]:
        """Return currently loaded model IDs."""


__all__ = [
    "CSVTransferExporterPort",
    "DocumentExtractorPort",
    "DocumentStoragePort",
    "EmbeddingModelPort",
    "ExecutorRepository",
    "IKERepository",
    "IssueRepository",
    "KnowledgeRepository",
    "LLMModelPort",
    "LVMModelPort",
    "LayerRepository",
    "LvmLayoutAssistantPort",
    "ModelRegistryRepository",
    "ModelRuntimePort",
    "NodeRepository",
    "PromptTemplateRepository",
    "RFPRepository",
    "TransferPackageRepository",
    "UnitOfWork",
    "VectorIndexPort",
]
