"""RFP and Proposal Management Bounded Context: RFP, Proposal, FeedbackReport, and RFPRepository (Section 15)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterable, Sequence

from offline_rag.domain.shared.base import AggregateRoot
from offline_rag.domain.shared.errors import DomainValidationError, InvariantViolationError
from offline_rag.domain.shared.value_objects import NodeIdentifier


class ProposalRecommendation(str, Enum):
    """Recommendation decision in an automated FeedbackReport."""

    SELECT = "select"
    REVISE = "revise"
    REJECT = "reject"


def _clean_non_empty_tuple(items: Iterable[str], field_name: str) -> tuple[str, ...]:
    cleaned = tuple(s.strip() for s in items if isinstance(s, str) and s.strip())
    if not cleaned:
        raise DomainValidationError(f"{field_name} must contain at least one non-empty item.")
    return cleaned


class RFP(AggregateRoot):
    """Request for Proposal (RFP) generated from an Issue and research gaps (Section 15.1)."""

    def __init__(
        self,
        rfp_id: str,
        node_id: NodeIdentifier,
        origin_issue_id: str,
        title: str,
        problem_statement: str,
        scope: str,
        technical_requirements: Iterable[str],
        expected_deliverables: Iterable[str],
        evaluation_criteria: Iterable[str],
        constraints: Iterable[str],
        required_expertise: Iterable[str],
        research_gaps: Iterable[str] = (),
    ) -> None:
        for name, val in (
            ("rfp_id", rfp_id),
            ("origin_issue_id", origin_issue_id),
            ("title", title),
            ("problem_statement", problem_statement),
            ("scope", scope),
        ):
            if not isinstance(val, str) or not val.strip():
                raise DomainValidationError(f"RFP.{name} cannot be empty.")
        if not isinstance(node_id, NodeIdentifier):
            raise DomainValidationError("RFP.node_id must be a NodeIdentifier.")

        super().__init__(rfp_id.strip())
        self._rfp_id = rfp_id.strip()
        self._node_id = node_id
        self._origin_issue_id = origin_issue_id.strip()
        self._title = title.strip()
        self._problem_statement = problem_statement.strip()
        self._scope = scope.strip()
        self._technical_requirements = _clean_non_empty_tuple(
            technical_requirements, "RFP.technical_requirements"
        )
        self._expected_deliverables = _clean_non_empty_tuple(
            expected_deliverables, "RFP.expected_deliverables"
        )
        self._evaluation_criteria = _clean_non_empty_tuple(
            evaluation_criteria, "RFP.evaluation_criteria"
        )
        self._constraints = _clean_non_empty_tuple(constraints, "RFP.constraints")
        self._required_expertise = _clean_non_empty_tuple(
            required_expertise, "RFP.required_expertise"
        )
        self._research_gaps = tuple(
            g.strip() for g in research_gaps if isinstance(g, str) and g.strip()
        )

    @property
    def rfp_id(self) -> str:
        return self._rfp_id

    @property
    def node_id(self) -> NodeIdentifier:
        return self._node_id

    @property
    def origin_issue_id(self) -> str:
        return self._origin_issue_id

    @property
    def title(self) -> str:
        return self._title

    @property
    def problem_statement(self) -> str:
        return self._problem_statement

    @property
    def scope(self) -> str:
        return self._scope

    @property
    def technical_requirements(self) -> tuple[str, ...]:
        return self._technical_requirements

    @property
    def expected_deliverables(self) -> tuple[str, ...]:
        return self._expected_deliverables

    @property
    def evaluation_criteria(self) -> tuple[str, ...]:
        return self._evaluation_criteria

    @property
    def constraints(self) -> tuple[str, ...]:
        return self._constraints

    @property
    def required_expertise(self) -> tuple[str, ...]:
        return self._required_expertise

    @property
    def research_gaps(self) -> tuple[str, ...]:
        return self._research_gaps


class Proposal(AggregateRoot):
    """Submitted Proposal responding to an RFP (Section 15.2)."""

    def __init__(
        self,
        proposal_id: str,
        rfp_id: str,
        proposer_name: str,
        proposed_methodology: str,
        technical_summary: str,
        meets_mandatory_requirements: bool = True,
        evaluation_score: float | None = None,
        status: str = "submitted",
    ) -> None:
        for name, val in (
            ("proposal_id", proposal_id),
            ("rfp_id", rfp_id),
            ("proposer_name", proposer_name),
            ("proposed_methodology", proposed_methodology),
            ("technical_summary", technical_summary),
        ):
            if not isinstance(val, str) or not val.strip():
                raise DomainValidationError(f"Proposal.{name} cannot be empty.")

        if evaluation_score is not None and (evaluation_score < 0.0 or evaluation_score > 100.0):
            raise DomainValidationError("Proposal.evaluation_score must be in [0.0, 100.0].")

        super().__init__(proposal_id.strip())
        self._proposal_id = proposal_id.strip()
        self._rfp_id = rfp_id.strip()
        self._proposer_name = proposer_name.strip()
        self._proposed_methodology = proposed_methodology.strip()
        self._technical_summary = technical_summary.strip()
        self._meets_mandatory_requirements = bool(meets_mandatory_requirements)
        self._evaluation_score = evaluation_score
        self._status = status.strip() or "submitted"

    @property
    def proposal_id(self) -> str:
        return self._proposal_id

    @property
    def rfp_id(self) -> str:
        return self._rfp_id

    @property
    def proposer_name(self) -> str:
        return self._proposer_name

    @property
    def proposed_methodology(self) -> str:
        return self._proposed_methodology

    @property
    def technical_summary(self) -> str:
        return self._technical_summary

    @property
    def meets_mandatory_requirements(self) -> bool:
        return self._meets_mandatory_requirements

    @property
    def evaluation_score(self) -> float | None:
        return self._evaluation_score

    @property
    def status(self) -> str:
        return self._status

    def record_evaluation(self, score: float, status: str = "evaluated") -> None:
        if score < 0.0 or score > 100.0:
            raise DomainValidationError("Evaluation score must be between 0.0 and 100.0.")
        self._evaluation_score = round(float(score), 2)
        self._status = status


class FeedbackReport(AggregateRoot):
    """Structured automated feedback report for a submitted Proposal against RFP criteria (Section 15.3)."""

    def __init__(
        self,
        feedback_id: str,
        proposal_id: str,
        rfp_id: str,
        strengths: Iterable[str],
        weaknesses: Iterable[str],
        missing_requirements: Iterable[str],
        technical_risks: Iterable[str],
        suggested_score: float,
        recommendation: ProposalRecommendation | str,
        criterion_references: Iterable[str],
    ) -> None:
        for name, val in (
            ("feedback_id", feedback_id),
            ("proposal_id", proposal_id),
            ("rfp_id", rfp_id),
        ):
            if not isinstance(val, str) or not val.strip():
                raise DomainValidationError(f"FeedbackReport.{name} cannot be empty.")

        if isinstance(suggested_score, bool) or not isinstance(suggested_score, (int, float)):
            raise DomainValidationError("FeedbackReport.suggested_score must be a number.")
        if suggested_score < 0.0 or suggested_score > 100.0:
            raise DomainValidationError("FeedbackReport.suggested_score must be in [0.0, 100.0].")

        try:
            rec = (
                recommendation
                if isinstance(recommendation, ProposalRecommendation)
                else ProposalRecommendation(str(recommendation).lower())
            )
        except ValueError as exc:
            raise DomainValidationError(
                f"Invalid ProposalRecommendation: {recommendation}"
            ) from exc

        refs = tuple(r.strip() for r in criterion_references if isinstance(r, str) and r.strip())
        if not refs:
            raise InvariantViolationError(
                "FeedbackReport must reference at least one RFP criterion or proposal section."
            )

        super().__init__(feedback_id.strip())
        self._feedback_id = feedback_id.strip()
        self._proposal_id = proposal_id.strip()
        self._rfp_id = rfp_id.strip()
        self._strengths = tuple(s.strip() for s in strengths if isinstance(s, str) and s.strip())
        self._weaknesses = tuple(w.strip() for w in weaknesses if isinstance(w, str) and w.strip())
        self._missing_requirements = tuple(
            m.strip() for m in missing_requirements if isinstance(m, str) and m.strip()
        )
        self._technical_risks = tuple(
            r.strip() for r in technical_risks if isinstance(r, str) and r.strip()
        )
        self._suggested_score = round(float(suggested_score), 2)
        self._recommendation = rec
        self._criterion_references = refs

    @property
    def feedback_id(self) -> str:
        return self._feedback_id

    @property
    def proposal_id(self) -> str:
        return self._proposal_id

    @property
    def rfp_id(self) -> str:
        return self._rfp_id

    @property
    def strengths(self) -> tuple[str, ...]:
        return self._strengths

    @property
    def weaknesses(self) -> tuple[str, ...]:
        return self._weaknesses

    @property
    def missing_requirements(self) -> tuple[str, ...]:
        return self._missing_requirements

    @property
    def technical_risks(self) -> tuple[str, ...]:
        return self._technical_risks

    @property
    def suggested_score(self) -> float:
        return self._suggested_score

    @property
    def recommendation(self) -> ProposalRecommendation:
        return self._recommendation

    @property
    def criterion_references(self) -> tuple[str, ...]:
        return self._criterion_references


class RFPRepository(ABC):
    """Abstract repository for RFP, Proposal, and FeedbackReport aggregates."""

    @abstractmethod
    def save_rfp(self, rfp: RFP) -> None:
        """Persist or update an RFP."""

    @abstractmethod
    def get_rfp_by_id(self, rfp_id: str) -> RFP | None:
        """Retrieve an RFP by ID."""

    @abstractmethod
    def list_rfps_by_node(self, node_id: NodeIdentifier) -> Sequence[RFP]:
        """List all RFPs belonging to a Node."""

    @abstractmethod
    def save_proposal(self, proposal: Proposal) -> None:
        """Persist or update a Proposal."""

    @abstractmethod
    def get_proposal_by_id(self, proposal_id: str) -> Proposal | None:
        """Retrieve a Proposal by ID."""

    @abstractmethod
    def list_proposals_by_rfp(self, rfp_id: str) -> Sequence[Proposal]:
        """List all Proposals submitted for an RFP."""

    @abstractmethod
    def save_feedback_report(self, report: FeedbackReport) -> None:
        """Persist or update a FeedbackReport."""

    @abstractmethod
    def get_feedback_by_proposal(self, proposal_id: str) -> FeedbackReport | None:
        """Retrieve a FeedbackReport for a given Proposal ID."""


__all__ = [
    "FeedbackReport",
    "Proposal",
    "ProposalRecommendation",
    "RFP",
    "RFPRepository",
]
