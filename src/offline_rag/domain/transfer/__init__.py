"""Knowledge Transfer Bounded Context: TransferPackage, Upward Delivery Fallback, and Repository (Sections 9 & 10)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from offline_rag.domain.evaluation import JudgeScore
from offline_rag.domain.shared.base import AggregateRoot, DomainEvent, utc_now
from offline_rag.domain.shared.errors import DomainValidationError, InvariantViolationError
from offline_rag.domain.shared.value_objects import (
    LayerIndex,
    NodeIdentifier,
    TransferDetailLevel,
    VersionTag,
)


class TransferPackageStatus(str, Enum):
    """Lifecycle status of a TransferPackage."""

    DRAFT = "draft"
    ACCEPTED = "accepted"
    REJECTED_BY_JUDGE = "rejected_by_judge"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class TransferPackageEvaluatedEvent(DomainEvent):
    """Emitted when a TransferPackage is evaluated by LLM-as-a-Judge."""

    package_id: str
    source_node_id: NodeIdentifier
    target_parent_node_id: NodeIdentifier
    score: float
    accepted: bool


@dataclass(frozen=True)
class TransferDeliveryResolution:
    """Result of resolving the latest deliverable TransferPackage for an upward layer request (Section 9.2)."""

    delivered_package: TransferPackage
    used_previous_accepted_fallback: bool
    warning_message: str | None = None
    rejected_latest_package_id: str | None = None


class TransferPackage(AggregateRoot):
    """Aggregate Root representing a cyclic upward knowledge transfer package (Sections 9 & 10)."""

    def __init__(
        self,
        package_id: str,
        source_node_id: NodeIdentifier,
        target_parent_node_id: NodeIdentifier,
        target_layer_index: LayerIndex,
        detail_level: TransferDetailLevel,
        version_tag: VersionTag,
        reporting_phase: str,
        payload_sections: Mapping[str, Any],
        source_references: Iterable[str],
        required_fields: Iterable[str] = (),
        max_length: int = 20000,
        previous_accepted_package_id: str | None = None,
        created_at: datetime | None = None,
    ) -> None:
        if not isinstance(package_id, str) or not package_id.strip():
            raise DomainValidationError("TransferPackage.package_id cannot be empty.")
        if not isinstance(source_node_id, NodeIdentifier):
            raise DomainValidationError("TransferPackage.source_node_id must be a NodeIdentifier.")
        if not isinstance(target_parent_node_id, NodeIdentifier):
            raise DomainValidationError(
                "TransferPackage.target_parent_node_id must be a NodeIdentifier."
            )
        if source_node_id == target_parent_node_id:
            raise InvariantViolationError(
                "TransferPackage source_node_id and target_parent_node_id cannot be identical."
            )
        if not isinstance(target_layer_index, LayerIndex):
            raise DomainValidationError("TransferPackage.target_layer_index must be a LayerIndex.")
        if not isinstance(detail_level, TransferDetailLevel):
            raise DomainValidationError(
                "TransferPackage.detail_level must be a TransferDetailLevel."
            )
        if not isinstance(version_tag, VersionTag):
            raise DomainValidationError("TransferPackage.version_tag must be a VersionTag.")
        if not isinstance(reporting_phase, str) or not reporting_phase.strip():
            raise DomainValidationError("TransferPackage.reporting_phase cannot be empty.")
        if not isinstance(payload_sections, Mapping) or not payload_sections:
            raise DomainValidationError("TransferPackage.payload_sections cannot be empty.")

        req_fields = tuple(f.strip() for f in required_fields if isinstance(f, str) and f.strip())
        missing = [f for f in req_fields if f not in payload_sections or payload_sections[f] in (None, "", [])]
        if missing:
            raise InvariantViolationError(
                f"TransferPackage '{package_id}' is missing required layer fields: {missing}"
            )

        serialized_len = sum(len(str(v)) for v in payload_sections.values())
        if max_length > 0 and serialized_len > max_length:
            raise InvariantViolationError(
                f"TransferPackage '{package_id}' content length ({serialized_len}) "
                f"exceeds max_length ({max_length})."
            )

        refs = tuple(r.strip() for r in source_references if isinstance(r, str) and r.strip())
        if not refs:
            raise InvariantViolationError(
                f"TransferPackage '{package_id}' must contain at least one source_reference."
            )

        super().__init__(package_id.strip())
        self._package_id = package_id.strip()
        self._source_node_id = source_node_id
        self._target_parent_node_id = target_parent_node_id
        self._target_layer_index = target_layer_index
        self._detail_level = detail_level
        self._version_tag = version_tag
        self._reporting_phase = reporting_phase.strip()
        self._payload_sections = MappingProxyType(dict(payload_sections))
        self._source_references = refs
        self._status = TransferPackageStatus.DRAFT
        self._judge_score: JudgeScore | None = None
        self._previous_accepted_package_id = previous_accepted_package_id
        self._created_at = created_at or utc_now()

    @property
    def package_id(self) -> str:
        return self._package_id

    @property
    def source_node_id(self) -> NodeIdentifier:
        return self._source_node_id

    @property
    def target_parent_node_id(self) -> NodeIdentifier:
        return self._target_parent_node_id

    @property
    def target_layer_index(self) -> LayerIndex:
        return self._target_layer_index

    @property
    def detail_level(self) -> TransferDetailLevel:
        return self._detail_level

    @property
    def version_tag(self) -> VersionTag:
        return self._version_tag

    @property
    def reporting_phase(self) -> str:
        return self._reporting_phase

    @property
    def payload_sections(self) -> Mapping[str, Any]:
        return self._payload_sections

    @property
    def source_references(self) -> tuple[str, ...]:
        return self._source_references

    @property
    def status(self) -> TransferPackageStatus:
        return self._status

    @property
    def judge_score(self) -> JudgeScore | None:
        return self._judge_score

    @property
    def is_accepted(self) -> bool:
        return self._status == TransferPackageStatus.ACCEPTED

    @property
    def previous_accepted_package_id(self) -> str | None:
        return self._previous_accepted_package_id

    @property
    def created_at(self) -> datetime:
        return self._created_at

    def apply_judge_score(self, score: JudgeScore) -> bool:
        """Evaluate this package with a JudgeScore and transition to ACCEPTED or REJECTED_BY_JUDGE."""
        if not isinstance(score, JudgeScore):
            raise DomainValidationError("score must be a JudgeScore instance.")
        if score.target_artifact_id != self._package_id:
            raise InvariantViolationError(
                f"JudgeScore target_artifact_id ('{score.target_artifact_id}') "
                f"does not match TransferPackage ('{self._package_id}')."
            )
        self._judge_score = score
        if score.passed:
            self._status = TransferPackageStatus.ACCEPTED
        else:
            self._status = TransferPackageStatus.REJECTED_BY_JUDGE

        self.record_event(
            TransferPackageEvaluatedEvent(
                package_id=self._package_id,
                source_node_id=self._source_node_id,
                target_parent_node_id=self._target_parent_node_id,
                score=score.total_score,
                accepted=score.passed,
            )
        )
        return score.passed

    def resolve_delivery(
        self, previous_accepted_package: TransferPackage | None = None
    ) -> TransferDeliveryResolution:
        """Resolve which package to deliver upward, falling back to previous accepted if rejected (Section 9.2)."""
        if self.is_accepted:
            return TransferDeliveryResolution(
                delivered_package=self,
                used_previous_accepted_fallback=False,
                warning_message=None,
                rejected_latest_package_id=None,
            )

        if previous_accepted_package is not None and previous_accepted_package.is_accepted:
            score_val = self._judge_score.total_score if self._judge_score else 0.0
            return TransferDeliveryResolution(
                delivered_package=previous_accepted_package,
                used_previous_accepted_fallback=True,
                warning_message=(
                    f"Latest package '{self._package_id}' ({self._version_tag.tag}) scored {score_val} "
                    f"below threshold; delivering previous accepted package "
                    f"'{previous_accepted_package.package_id}' ({previous_accepted_package.version_tag.tag})."
                ),
                rejected_latest_package_id=self._package_id,
            )

        raise InvariantViolationError(
            f"Package '{self._package_id}' is not accepted by Judge and no previous accepted "
            f"fallback package is available for node '{self._source_node_id}'."
        )


class TransferPackageRepository(ABC):
    """Abstract repository for TransferPackage aggregates."""

    @abstractmethod
    def save(self, package: TransferPackage) -> None:
        """Persist or update a TransferPackage."""

    @abstractmethod
    def get_by_id(self, package_id: str) -> TransferPackage | None:
        """Retrieve a TransferPackage by ID."""

    @abstractmethod
    def get_latest_by_node(self, source_node_id: NodeIdentifier) -> TransferPackage | None:
        """Return the most recently created TransferPackage for a source Node."""

    @abstractmethod
    def get_latest_accepted_by_node(self, source_node_id: NodeIdentifier) -> TransferPackage | None:
        """Return the most recent TransferPackage with status==ACCEPTED for a source Node."""

    @abstractmethod
    def list_by_node(self, source_node_id: NodeIdentifier) -> Sequence[TransferPackage]:
        """Return all TransferPackages for a source Node ordered by version sequence."""


__all__ = [
    "TransferDeliveryResolution",
    "TransferDetailLevel",
    "TransferPackage",
    "TransferPackageEvaluatedEvent",
    "TransferPackageRepository",
    "TransferPackageStatus",
]
