"""Evaluation and LLM-as-a-Judge Bounded Context: JudgeScore, JudgeRubric, JudgeThreshold (Section 16)."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

from offline_rag.domain.shared.base import AggregateRoot, DomainEvent
from offline_rag.domain.shared.errors import DomainValidationError
from offline_rag.domain.shared.value_objects import (
    DEFAULT_RUBRIC_CRITERIA,
    JudgeRubric,
    JudgeThreshold,
    RubricCriterion,
)


@dataclass(frozen=True)
class OutputJudgedEvent(DomainEvent):
    """Emitted when an output or TransferPackage is evaluated by LLM-as-a-Judge."""

    score_id: str
    target_artifact_id: str
    total_score: float
    passed: bool


class JudgeScore(AggregateRoot):
    """Structured LLM-as-a-Judge evaluation entity (Section 16.1 & 16.2)."""

    def __init__(
        self,
        score_id: str,
        target_artifact_id: str,
        criterion_scores: Mapping[str, float],
        explanation: str,
        rubric: JudgeRubric | None = None,
        threshold: JudgeThreshold | None = None,
        missing_information: Iterable[str] = (),
        unsupported_claims: Iterable[str] = (),
        excessive_details: Iterable[str] = (),
        insufficient_details: Iterable[str] = (),
        revision_suggestions: Iterable[str] = (),
    ) -> None:
        if not isinstance(score_id, str) or not score_id.strip():
            raise DomainValidationError("JudgeScore.score_id cannot be empty.")
        if not isinstance(target_artifact_id, str) or not target_artifact_id.strip():
            raise DomainValidationError("JudgeScore.target_artifact_id cannot be empty.")
        if not isinstance(explanation, str) or not explanation.strip():
            raise DomainValidationError("JudgeScore.explanation cannot be empty.")

        active_rubric = rubric or JudgeRubric()
        active_threshold = threshold or JudgeThreshold(
            min_acceptable_score=18.0, max_possible_score=active_rubric.max_score
        )
        if not isinstance(active_rubric, JudgeRubric):
            raise DomainValidationError("JudgeScore.rubric must be a JudgeRubric.")
        if not isinstance(active_threshold, JudgeThreshold):
            raise DomainValidationError("JudgeScore.threshold must be a JudgeThreshold.")

        computed_total = active_rubric.validate_and_sum_scores(criterion_scores)
        is_passed = active_threshold.is_satisfied_by(computed_total)

        super().__init__(score_id.strip())
        self._score_id = score_id.strip()
        self._target_artifact_id = target_artifact_id.strip()
        self._rubric = active_rubric
        self._threshold = active_threshold
        self._criterion_scores = MappingProxyType(
            {k: float(v) for k, v in criterion_scores.items()}
        )
        self._total_score = computed_total
        self._passed = is_passed
        self._explanation = explanation.strip()
        self._missing_information = tuple(
            s.strip() for s in missing_information if isinstance(s, str) and s.strip()
        )
        self._unsupported_claims = tuple(
            s.strip() for s in unsupported_claims if isinstance(s, str) and s.strip()
        )
        self._excessive_details = tuple(
            s.strip() for s in excessive_details if isinstance(s, str) and s.strip()
        )
        self._insufficient_details = tuple(
            s.strip() for s in insufficient_details if isinstance(s, str) and s.strip()
        )
        self._revision_suggestions = tuple(
            s.strip() for s in revision_suggestions if isinstance(s, str) and s.strip()
        )

        self.record_event(
            OutputJudgedEvent(
                score_id=self._score_id,
                target_artifact_id=self._target_artifact_id,
                total_score=self._total_score,
                passed=self._passed,
            )
        )

    @property
    def score_id(self) -> str:
        return self._score_id

    @property
    def target_artifact_id(self) -> str:
        return self._target_artifact_id

    @property
    def rubric(self) -> JudgeRubric:
        return self._rubric

    @property
    def threshold(self) -> JudgeThreshold:
        return self._threshold

    @property
    def criterion_scores(self) -> Mapping[str, float]:
        return self._criterion_scores

    @property
    def total_score(self) -> float:
        return self._total_score

    @property
    def passed(self) -> bool:
        return self._passed

    @property
    def explanation(self) -> str:
        return self._explanation

    @property
    def missing_information(self) -> tuple[str, ...]:
        return self._missing_information

    @property
    def unsupported_claims(self) -> tuple[str, ...]:
        return self._unsupported_claims

    @property
    def excessive_details(self) -> tuple[str, ...]:
        return self._excessive_details

    @property
    def insufficient_details(self) -> tuple[str, ...]:
        return self._insufficient_details

    @property
    def revision_suggestions(self) -> tuple[str, ...]:
        return self._revision_suggestions


__all__ = [
    "DEFAULT_RUBRIC_CRITERIA",
    "JudgeRubric",
    "JudgeScore",
    "JudgeThreshold",
    "OutputJudgedEvent",
    "RubricCriterion",
]
