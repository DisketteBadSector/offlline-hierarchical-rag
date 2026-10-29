"""Immutable Value Objects (11 core Value Objects from Section 5.3 of the proposal)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from offline_rag.domain.shared.errors import InvalidValueObjectError

NODE_ID_PATTERN = re.compile(r"^N-[0-9A-Za-z]{2,12}(-[0-9A-Za-z]{1,12})*$")

VALID_CONFIDENTIALITY_LEVELS: tuple[str, ...] = (
    "public",
    "internal",
    "restricted",
    "confidential",
)
CONFIDENTIALITY_RANK: Mapping[str, int] = {
    "public": 0,
    "internal": 1,
    "restricted": 2,
    "confidential": 3,
}

VALID_TRANSFER_DETAIL_LEVELS: tuple[str, ...] = (
    "executive_high",
    "managerial",
    "technical_summary",
    "technical_detailed",
)

VALID_PRIORITIES: tuple[str, ...] = ("low", "medium", "high", "critical")


def _ensure_not_bool_int(val: object, field_name: str) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise InvalidValueObjectError(f"{field_name} must be an integer, got {type(val).__name__}.")
    return val


def _ensure_not_bool_float(val: object, field_name: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise InvalidValueObjectError(f"{field_name} must be a number, got {type(val).__name__}.")
    fval = float(val)
    if math.isnan(fval) or math.isinf(fval):
        raise InvalidValueObjectError(f"{field_name} cannot be NaN or infinite.")
    return fval


def normalize_persian_text(text: str) -> str:
    """Normalize Persian/Arabic characters and collapse extra whitespace."""
    normalized = text.replace("ي", "ی").replace("ك", "ک").replace("\u200c", " ")
    return " ".join(normalized.strip().split())


@dataclass(frozen=True, order=True)
class NodeIdentifier:
    """1. Stable identifier for an organizational Node (e.g. 'N-000', 'N-301')."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value.strip():
            raise InvalidValueObjectError("NodeIdentifier cannot be empty.")
        cleaned = self.value.strip()
        if not NODE_ID_PATTERN.match(cleaned):
            raise InvalidValueObjectError(
                f"Invalid NodeIdentifier '{self.value}'. Expected pattern like 'N-000' or 'N-301'."
            )
        object.__setattr__(self, "value", cleaned)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, order=True)
class LayerIndex:
    """2. Hierarchical layer position (integer between 0 and 4 inclusive)."""

    value: int

    def __post_init__(self) -> None:
        ival = _ensure_not_bool_int(self.value, "LayerIndex.value")
        if ival < 0 or ival > 4:
            raise InvalidValueObjectError(
                f"LayerIndex must be between 0 and 4 inclusive, got {ival}."
            )

    @property
    def is_strategic_root(self) -> bool:
        return self.value == 0

    @property
    def is_knowledge_source(self) -> bool:
        return self.value == 4

    def expected_parent_rule_layer(self) -> LayerIndex:
        """Return LayerIndex - 1 as required by rule R-07."""
        if self.value <= 0:
            raise InvalidValueObjectError("Layer 0 has no upward parent transfer rule.")
        return LayerIndex(self.value - 1)

    def __int__(self) -> int:
        return self.value


@dataclass(frozen=True)
class KnowledgeScope:
    """3. Scope, confidentiality level, and usage boundary of a KnowledgeItem."""

    owner_node_id: NodeIdentifier
    confidentiality: str = "internal"
    usage_domain: str = "general"
    allow_upward_transfer: bool = True
    allowed_layer_indices: tuple[LayerIndex, ...] = field(
        default_factory=lambda: tuple(LayerIndex(i) for i in range(5))
    )

    def __post_init__(self) -> None:
        if not isinstance(self.owner_node_id, NodeIdentifier):
            raise InvalidValueObjectError("KnowledgeScope.owner_node_id must be a NodeIdentifier.")
        if not isinstance(self.confidentiality, str) or (
            self.confidentiality not in VALID_CONFIDENTIALITY_LEVELS
        ):
            raise InvalidValueObjectError(
                f"Invalid confidentiality '{self.confidentiality}'. "
                f"Allowed: {VALID_CONFIDENTIALITY_LEVELS}"
            )
        if not isinstance(self.usage_domain, str) or not self.usage_domain.strip():
            raise InvalidValueObjectError("KnowledgeScope.usage_domain cannot be empty.")
        if not isinstance(self.allow_upward_transfer, bool):
            raise InvalidValueObjectError("KnowledgeScope.allow_upward_transfer must be a boolean.")

        normalized_layers = tuple(
            lyr if isinstance(lyr, LayerIndex) else LayerIndex(lyr)
            for lyr in self.allowed_layer_indices
        )
        if not normalized_layers:
            raise InvalidValueObjectError("KnowledgeScope.allowed_layer_indices cannot be empty.")
        object.__setattr__(self, "usage_domain", self.usage_domain.strip())
        object.__setattr__(self, "allowed_layer_indices", normalized_layers)

    def is_accessible_at_confidentiality(self, max_allowed_confidentiality: str) -> bool:
        if max_allowed_confidentiality not in CONFIDENTIALITY_RANK:
            raise InvalidValueObjectError(
                f"Unknown confidentiality level: {max_allowed_confidentiality}"
            )
        return (
            CONFIDENTIALITY_RANK[self.confidentiality]
            <= CONFIDENTIALITY_RANK[max_allowed_confidentiality]
        )


@dataclass(frozen=True, order=True)
class VersionTag:
    """4. Immutable version tag supporting sequence numbers (>= 1) and semantic versions (e.g. '1.0.0')."""

    sequence_number: int
    tag: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.sequence_number, str):
            cleaned_str = self.sequence_number.strip()
            semver_match = re.match(r"^v?(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$", cleaned_str)
            if semver_match:
                major, minor, patch = (
                    int(semver_match.group(1)),
                    int(semver_match.group(2)),
                    int(semver_match.group(3)),
                )
                seq = max(major * 10000 + minor * 100 + patch, 1)
                object.__setattr__(self, "sequence_number", seq)
                object.__setattr__(self, "tag", cleaned_str)
                return

        seq = _ensure_not_bool_int(self.sequence_number, "VersionTag.sequence_number")
        if seq < 1:
            raise InvalidValueObjectError(
                f"VersionTag.sequence_number must be >= 1, got {seq}."
            )
        computed_tag = self.tag.strip() if isinstance(self.tag, str) and self.tag.strip() else f"v{seq}"
        object.__setattr__(self, "tag", computed_tag)

    @property
    def value(self) -> str:
        return self.tag

    @classmethod
    def initial(cls, tag: str = "v1") -> VersionTag:
        return cls(sequence_number=1, tag=tag)

    @classmethod
    def from_version_id(cls, version_id: str | VersionTag) -> VersionTag:
        """Parse a version identifier string (e.g. '1.0.0', '0.9.0', 'v1', 'v2') or return existing VersionTag."""
        if isinstance(version_id, VersionTag):
            return version_id
        if not isinstance(version_id, str) or not version_id.strip():
            raise InvalidValueObjectError("version_id must be a non-empty string or VersionTag.")
        cleaned = version_id.strip()
        semver_match = re.match(r"^v?(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$", cleaned)
        if semver_match:
            major, minor, patch = (
                int(semver_match.group(1)),
                int(semver_match.group(2)),
                int(semver_match.group(3)),
            )
            seq = max(major * 10000 + minor * 100 + patch, 1)
            return cls(sequence_number=seq, tag=cleaned)
        m = re.search(r"(\d+)$", cleaned)
        seq = int(m.group(1)) if m and int(m.group(1)) >= 1 else 1
        return cls(sequence_number=seq, tag=cleaned)

    def next_version(self, tag: str | None = None) -> VersionTag:
        next_seq = self.sequence_number + 1
        return VersionTag(sequence_number=next_seq, tag=tag or f"v{next_seq}")

    def __str__(self) -> str:
        return self.tag


@dataclass(frozen=True, order=True)
class ProgressPercentage:
    """5. Issue or SubIssue resolution progress percentage (0.0 to 100.0)."""

    value: float

    def __post_init__(self) -> None:
        fval = _ensure_not_bool_float(self.value, "ProgressPercentage.value")
        if fval < 0.0 or fval > 100.0:
            raise InvalidValueObjectError(
                f"ProgressPercentage must be between 0.0 and 100.0, got {fval}."
            )
        object.__setattr__(self, "value", round(fval, 2))

    @property
    def is_completed(self) -> bool:
        return self.value >= 100.0

    @property
    def is_not_started(self) -> bool:
        return self.value == 0.0


@dataclass(frozen=True)
class RubricCriterion:
    """Single scoring criterion inside a JudgeRubric."""

    key: str
    title_fa: str
    max_points: float

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise InvalidValueObjectError("RubricCriterion.key cannot be empty.")
        if not isinstance(self.title_fa, str) or not self.title_fa.strip():
            raise InvalidValueObjectError("RubricCriterion.title_fa cannot be empty.")
        pts = _ensure_not_bool_float(self.max_points, "RubricCriterion.max_points")
        if pts <= 0.0:
            raise InvalidValueObjectError("RubricCriterion.max_points must be > 0.")
        object.__setattr__(self, "key", self.key.strip())
        object.__setattr__(self, "title_fa", self.title_fa.strip())
        object.__setattr__(self, "max_points", pts)


DEFAULT_RUBRIC_CRITERIA: tuple[RubricCriterion, ...] = (
    RubricCriterion("source_faithfulness", "وفاداری به دانش منبع", 5.0),
    RubricCriterion("completeness", "کامل‌بودن نسبت به درخواست کاربر", 4.0),
    RubricCriterion("detail_level_fit", "تناسب سطح جزئیات با لایه هدف", 3.0),
    RubricCriterion("uncertainty_handling", "مدیریت درست عدم قطعیت و شکاف‌ها", 3.0),
    RubricCriterion("clarity_and_structure", "وضوح و ساختار حرفه‌ای", 2.0),
    RubricCriterion("source_citability", "قابلیت ارجاع به منبع", 2.0),
    RubricCriterion("no_unsupported_claims", "نبود ادعای بدون پشتوانه", 1.0),
)


@dataclass(frozen=True)
class JudgeRubric:
    """6. Scoring rubric used by LLM-as-a-Judge (default total ceiling: 20 points)."""

    criteria: tuple[RubricCriterion, ...] = DEFAULT_RUBRIC_CRITERIA
    expected_total_ceiling: float = 20.0

    def __post_init__(self) -> None:
        if not self.criteria:
            raise InvalidValueObjectError("JudgeRubric.criteria cannot be empty.")
        ceiling = _ensure_not_bool_float(
            self.expected_total_ceiling, "JudgeRubric.expected_total_ceiling"
        )
        if ceiling <= 0.0:
            raise InvalidValueObjectError("JudgeRubric.expected_total_ceiling must be > 0.")

        normalized_criteria = tuple(self.criteria)
        keys = [c.key for c in normalized_criteria]
        if len(keys) != len(set(keys)):
            raise InvalidValueObjectError("JudgeRubric contains duplicate criterion keys.")

        total = round(sum(c.max_points for c in normalized_criteria), 4)
        if abs(total - ceiling) > 1e-4:
            raise InvalidValueObjectError(
                f"Sum of rubric criteria ({total}) does not match expected_total_ceiling ({ceiling})."
            )
        object.__setattr__(self, "criteria", normalized_criteria)
        object.__setattr__(self, "expected_total_ceiling", ceiling)

    @property
    def max_score(self) -> float:
        return self.expected_total_ceiling

    def validate_and_sum_scores(self, criterion_scores: Mapping[str, float]) -> float:
        """Validate individual criterion scores against ceilings and return total score."""
        criteria_map = {c.key: c for c in self.criteria}
        missing = set(criteria_map.keys()) - set(criterion_scores.keys())
        if missing:
            raise InvalidValueObjectError(
                f"Missing rubric criterion scores for keys: {sorted(missing)}"
            )
        unknown = set(criterion_scores.keys()) - set(criteria_map.keys())
        if unknown:
            raise InvalidValueObjectError(
                f"Unknown rubric criterion keys provided: {sorted(unknown)}"
            )

        total = 0.0
        for key, raw_score in criterion_scores.items():
            score = _ensure_not_bool_float(raw_score, f"score[{key}]")
            max_pts = criteria_map[key].max_points
            if score < 0.0 or score > max_pts:
                raise InvalidValueObjectError(
                    f"Score for '{key}' must be between 0.0 and {max_pts}, got {score}."
                )
            total += score
        return round(total, 2)


@dataclass(frozen=True)
class JudgeThreshold:
    """7. Minimum acceptable Judge score (default 18 out of 20 for critical outputs)."""

    min_acceptable_score: float = 18.0
    max_possible_score: float = 20.0

    def __post_init__(self) -> None:
        min_s = _ensure_not_bool_float(
            self.min_acceptable_score, "JudgeThreshold.min_acceptable_score"
        )
        max_s = _ensure_not_bool_float(
            self.max_possible_score, "JudgeThreshold.max_possible_score"
        )
        if max_s <= 0.0:
            raise InvalidValueObjectError("JudgeThreshold.max_possible_score must be > 0.")
        if min_s <= 0.0 or min_s > max_s:
            raise InvalidValueObjectError(
                f"JudgeThreshold.min_acceptable_score must be in (0, {max_s}], got {min_s}."
            )
        object.__setattr__(self, "min_acceptable_score", round(min_s, 2))
        object.__setattr__(self, "max_possible_score", round(max_s, 2))

    def is_satisfied_by(self, score: float) -> bool:
        fscore = _ensure_not_bool_float(score, "score")
        return fscore >= self.min_acceptable_score


@dataclass(frozen=True, order=True)
class SimilarityScore:
    """9. Normalized vector or hybrid similarity score in [0.0, 1.0]."""

    value: float

    def __post_init__(self) -> None:
        fval = _ensure_not_bool_float(self.value, "SimilarityScore.value")
        if fval < 0.0 or fval > 1.0:
            raise InvalidValueObjectError(
                f"SimilarityScore must be between 0.0 and 1.0, got {fval}."
            )
        object.__setattr__(self, "value", round(fval, 6))

    def meets_threshold(self, threshold: SimilarityScore | float) -> bool:
        target = threshold.value if isinstance(threshold, SimilarityScore) else float(threshold)
        return self.value >= target


@dataclass(frozen=True)
class RetrievalQuery:
    """8. Normalized user or system query with node boundary and metadata filters."""

    raw_text: str
    requesting_node_id: NodeIdentifier
    top_k: int = 5
    domain_filter: str | None = None
    max_confidentiality: str = "internal"
    metadata_filters: tuple[tuple[str, str], ...] = ()
    min_similarity: SimilarityScore = field(default_factory=lambda: SimilarityScore(0.0))

    def __post_init__(self) -> None:
        if not isinstance(self.raw_text, str) or not self.raw_text.strip():
            raise InvalidValueObjectError("RetrievalQuery.raw_text cannot be empty.")
        normalized = normalize_persian_text(self.raw_text)
        if not normalized:
            raise InvalidValueObjectError("RetrievalQuery normalized text cannot be empty.")
        if not isinstance(self.requesting_node_id, NodeIdentifier):
            raise InvalidValueObjectError(
                "RetrievalQuery.requesting_node_id must be a NodeIdentifier."
            )
        k = _ensure_not_bool_int(self.top_k, "RetrievalQuery.top_k")
        if k < 1 or k > 100:
            raise InvalidValueObjectError(
                f"RetrievalQuery.top_k must be between 1 and 100, got {k}."
            )
        if self.max_confidentiality not in VALID_CONFIDENTIALITY_LEVELS:
            raise InvalidValueObjectError(
                f"Invalid max_confidentiality '{self.max_confidentiality}'."
            )
        if not isinstance(self.min_similarity, SimilarityScore):
            raise InvalidValueObjectError(
                "RetrievalQuery.min_similarity must be a SimilarityScore."
            )
        object.__setattr__(self, "metadata_filters", tuple(self.metadata_filters))

    @property
    def normalized_text(self) -> str:
        return normalize_persian_text(self.raw_text)


@dataclass(frozen=True)
class TransferDetailLevel:
    """10. Knowledge transfer detail level ('executive_high', 'managerial', 'technical_summary', 'technical_detailed')."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or self.value.strip() not in VALID_TRANSFER_DETAIL_LEVELS:
            raise InvalidValueObjectError(
                f"Invalid TransferDetailLevel '{self.value}'. "
                f"Allowed: {VALID_TRANSFER_DETAIL_LEVELS}"
            )
        object.__setattr__(self, "value", self.value.strip())

    @classmethod
    def for_target_layer(cls, target_layer: LayerIndex) -> TransferDetailLevel:
        mapping = {
            0: "executive_high",
            1: "managerial",
            2: "technical_summary",
            3: "technical_detailed",
        }
        if target_layer.value not in mapping:
            raise InvalidValueObjectError(
                f"Layer {target_layer.value} does not have a transfer detail level."
            )
        return cls(mapping[target_layer.value])

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class MetadataRecord:
    """11. Structured metadata attached to a KnowledgeItem or Issue (Section 7.5).

    Enforces mandatory schema fields:
    - domain
    - node_id
    - document_type
    - confidentiality
    - initial_version_id (or initial_version_tag)
    - active_version_id (or active_version_tag)
    """

    domain: str
    node_id: NodeIdentifier
    document_type: str
    confidentiality: str = "internal"
    initial_version_tag: VersionTag | None = None
    active_version_tag: VersionTag | None = None
    initial_version_id: str | VersionTag | None = None
    active_version_id: str | VersionTag | None = None
    author_or_owner: str = "system"
    sub_domain: str | None = None
    project_id: str | None = None
    related_issue_id: str | None = None
    related_executor_id: str | None = None
    related_rfp_id: str | None = None
    status: str = "active"
    priority: str = "medium"
    tags: tuple[str, ...] = ()
    source_reliability: float = 1.0
    target_transfer_layer: LayerIndex | None = None
    suggested_detail_level: TransferDetailLevel | None = None

    def __post_init__(self) -> None:
        for field_name, val in (
            ("domain", self.domain),
            ("author_or_owner", self.author_or_owner),
            ("document_type", self.document_type),
            ("status", self.status),
        ):
            if not isinstance(val, str) or not val.strip():
                raise InvalidValueObjectError(f"MetadataRecord.{field_name} cannot be empty.")
            object.__setattr__(self, field_name, val.strip())

        if isinstance(self.node_id, str):
            object.__setattr__(self, "node_id", NodeIdentifier(self.node_id))
        elif not isinstance(self.node_id, NodeIdentifier):
            raise InvalidValueObjectError("MetadataRecord.node_id must be a NodeIdentifier.")

        if not isinstance(self.confidentiality, str) or (
            self.confidentiality not in VALID_CONFIDENTIALITY_LEVELS
        ):
            raise InvalidValueObjectError(
                f"Invalid MetadataRecord.confidentiality '{self.confidentiality}'."
            )

        # Resolve initial_version_tag and initial_version_id
        raw_init = (
            self.initial_version_tag
            if self.initial_version_tag is not None
            else self.initial_version_id
        )
        if raw_init is None:
            raise InvalidValueObjectError(
                "MetadataRecord requires 'initial_version_id' or 'initial_version_tag'."
            )
        resolved_init = VersionTag.from_version_id(raw_init)

        # Resolve active_version_tag and active_version_id
        raw_active = (
            self.active_version_tag
            if self.active_version_tag is not None
            else self.active_version_id
        )
        if raw_active is None:
            raise InvalidValueObjectError(
                "MetadataRecord requires 'active_version_id' or 'active_version_tag'."
            )
        resolved_active = VersionTag.from_version_id(raw_active)

        if resolved_active.sequence_number < resolved_init.sequence_number:
            raise InvalidValueObjectError(
                "MetadataRecord.active_version_tag sequence_number cannot be less than initial_version_tag."
            )

        object.__setattr__(self, "initial_version_tag", resolved_init)
        object.__setattr__(self, "active_version_tag", resolved_active)
        object.__setattr__(self, "initial_version_id", resolved_init.tag)
        object.__setattr__(self, "active_version_id", resolved_active.tag)

        if self.priority not in VALID_PRIORITIES:
            raise InvalidValueObjectError(
                f"Invalid MetadataRecord.priority '{self.priority}'. Allowed: {VALID_PRIORITIES}"
            )
        rel = _ensure_not_bool_float(self.source_reliability, "MetadataRecord.source_reliability")
        if rel < 0.0 or rel > 1.0:
            raise InvalidValueObjectError(
                f"MetadataRecord.source_reliability must be in [0.0, 1.0], got {rel}."
            )
        if self.target_transfer_layer is not None and not isinstance(
            self.target_transfer_layer, LayerIndex
        ):
            raise InvalidValueObjectError(
                "MetadataRecord.target_transfer_layer must be a LayerIndex or None."
            )
        if self.suggested_detail_level is not None and not isinstance(
            self.suggested_detail_level, TransferDetailLevel
        ):
            raise InvalidValueObjectError(
                "MetadataRecord.suggested_detail_level must be a TransferDetailLevel or None."
            )
        cleaned_tags = tuple(t.strip() for t in self.tags if isinstance(t, str) and t.strip())
        object.__setattr__(self, "tags", cleaned_tags)
        object.__setattr__(self, "source_reliability", round(rel, 4))

    def with_active_version(self, new_version_tag: VersionTag | str) -> MetadataRecord:
        """Return a new immutable MetadataRecord with updated active_version_tag and active_version_id."""
        resolved = VersionTag.from_version_id(new_version_tag)
        return replace(
            self,
            active_version_tag=resolved,
            active_version_id=resolved.tag,
        )

    def matches_filters(self, filters: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> bool:
        """Check whether this MetadataRecord satisfies a set of retrieval metadata filters."""
        items = filters.items() if isinstance(filters, Mapping) else filters
        for key, expected in items:
            if expected is None:
                continue
            if key == "tag" or key == "tags":
                if isinstance(expected, str):
                    if expected not in self.tags:
                        return False
                elif isinstance(expected, (list, tuple, set)):
                    if not set(expected).issubset(set(self.tags)):
                        return False
                continue

            if not hasattr(self, key):
                return False
            actual = getattr(self, key)
            if isinstance(actual, (NodeIdentifier, VersionTag, TransferDetailLevel)):
                if str(actual) != str(expected):
                    return False
            elif isinstance(actual, LayerIndex):
                exp_int = expected.value if isinstance(expected, LayerIndex) else int(expected)
                if actual.value != exp_int:
                    return False
            else:
                if str(actual) != str(expected):
                    return False
        return True
