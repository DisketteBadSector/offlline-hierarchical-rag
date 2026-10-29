"""Issue Management Bounded Context: Issue, SubIssue, Status Lifecycle, and Progress Calculation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

from offline_rag.domain.shared.base import AggregateRoot, DomainEvent, Entity
from offline_rag.domain.shared.errors import DomainValidationError, InvariantViolationError
from offline_rag.domain.shared.value_objects import (
    VALID_PRIORITIES,
    MetadataRecord,
    NodeIdentifier,
    ProgressPercentage,
)


class IssueStatus(str, Enum):
    """Issue lifecycle statuses from Section 14.1 of the proposal."""

    NEW = "new"
    ANALYZING = "analyzing"
    KNOWLEDGE_IDENTIFIED = "knowledge_identified"
    EXECUTOR_IDENTIFIED = "executor_identified"
    SOLUTION_PROPOSED = "solution_proposed"
    IN_PROGRESS = "in_progress"
    AWAITING_DECISION = "awaiting_decision"
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class IssueDecomposedEvent(DomainEvent):
    """Emitted when a SubIssue is added to a parent Issue."""

    issue_id: str
    sub_issue_id: str
    relative_order: int


class SubIssue(Entity):
    """Derived SubIssue entity with independent status, progress, and relative order (Section 12.2)."""

    def __init__(
        self,
        sub_issue_id: str,
        parent_issue_id: str,
        title: str,
        description: str,
        relative_order: int,
        status: IssueStatus | str = IssueStatus.NEW,
        progress: ProgressPercentage | None = None,
        dependency_sub_issue_ids: Iterable[str] = (),
    ) -> None:
        if not isinstance(sub_issue_id, str) or not sub_issue_id.strip():
            raise DomainValidationError("SubIssue.sub_issue_id cannot be empty.")
        if not isinstance(parent_issue_id, str) or not parent_issue_id.strip():
            raise DomainValidationError("SubIssue.parent_issue_id cannot be empty.")
        if sub_issue_id.strip() == parent_issue_id.strip():
            raise InvariantViolationError("SubIssue cannot be its own parent Issue.")
        if not isinstance(title, str) or not title.strip():
            raise DomainValidationError("SubIssue.title cannot be empty.")
        if not isinstance(description, str) or not description.strip():
            raise DomainValidationError("SubIssue.description cannot be empty.")
        if isinstance(relative_order, bool) or not isinstance(relative_order, int) or relative_order < 1:
            raise DomainValidationError("SubIssue.relative_order must be an integer >= 1.")

        try:
            st = status if isinstance(status, IssueStatus) else IssueStatus(str(status))
        except ValueError as exc:
            raise DomainValidationError(f"Invalid IssueStatus: {status}") from exc

        prog = progress if progress is not None else (
            ProgressPercentage(100.0) if st == IssueStatus.RESOLVED else ProgressPercentage(0.0)
        )
        if not isinstance(prog, ProgressPercentage):
            raise DomainValidationError("SubIssue.progress must be a ProgressPercentage.")

        super().__init__(sub_issue_id.strip())
        self._sub_issue_id = sub_issue_id.strip()
        self._parent_issue_id = parent_issue_id.strip()
        self._title = title.strip()
        self._description = description.strip()
        self._relative_order = relative_order
        self._status = st
        self._progress = prog
        self._dependency_sub_issue_ids = tuple(
            d.strip() for d in dependency_sub_issue_ids if isinstance(d, str) and d.strip()
        )

    @property
    def sub_issue_id(self) -> str:
        return self._sub_issue_id

    @property
    def parent_issue_id(self) -> str:
        return self._parent_issue_id

    @property
    def title(self) -> str:
        return self._title

    @property
    def description(self) -> str:
        return self._description

    @property
    def relative_order(self) -> int:
        return self._relative_order

    @property
    def status(self) -> IssueStatus:
        return self._status

    @property
    def progress(self) -> ProgressPercentage:
        return self._progress

    @property
    def dependency_sub_issue_ids(self) -> tuple[str, ...]:
        return self._dependency_sub_issue_ids

    def update_status_and_progress(
        self,
        progress: ProgressPercentage,
        status: IssueStatus | None = None,
    ) -> None:
        if not isinstance(progress, ProgressPercentage):
            raise DomainValidationError("progress must be a ProgressPercentage.")
        self._progress = progress
        if status is not None:
            self._status = status
        elif progress.is_completed:
            self._status = IssueStatus.RESOLVED


class Issue(AggregateRoot):
    """Aggregate Root representing an organizational Issue or research gap (Section 12 & 14)."""

    def __init__(
        self,
        issue_id: str,
        node_id: NodeIdentifier,
        title: str,
        description: str,
        metadata: MetadataRecord,
        status: IssueStatus | str = IssueStatus.NEW,
        priority: str = "medium",
        progress: ProgressPercentage | None = None,
        sub_issues: Sequence[SubIssue] = (),
    ) -> None:
        if not isinstance(issue_id, str) or not issue_id.strip():
            raise DomainValidationError("Issue.issue_id cannot be empty.")
        if not isinstance(node_id, NodeIdentifier):
            raise DomainValidationError("Issue.node_id must be a NodeIdentifier.")
        if not isinstance(title, str) or not title.strip():
            raise DomainValidationError("Issue.title cannot be empty.")
        if not isinstance(description, str) or not description.strip():
            raise DomainValidationError("Issue.description cannot be empty.")
        if not isinstance(metadata, MetadataRecord):
            raise DomainValidationError("Issue.metadata must be a MetadataRecord.")
        if metadata.node_id != node_id:
            raise InvariantViolationError(
                f"Issue node_id ({node_id}) must match metadata.node_id ({metadata.node_id})."
            )
        if priority not in VALID_PRIORITIES:
            raise DomainValidationError(f"Invalid Issue priority '{priority}'.")

        try:
            st = status if isinstance(status, IssueStatus) else IssueStatus(str(status))
        except ValueError as exc:
            raise DomainValidationError(f"Invalid IssueStatus: {status}") from exc

        super().__init__(issue_id.strip())
        self._issue_id = issue_id.strip()
        self._node_id = node_id
        self._title = title.strip()
        self._description = description.strip()
        self._metadata = metadata
        self._status = st
        self._priority = priority
        self._progress = progress if progress is not None else ProgressPercentage(0.0)
        self._sub_issues: list[SubIssue] = []
        self._proposed_solution: str | None = None

        for sub in sub_issues:
            self.add_sub_issue(sub)

    @property
    def issue_id(self) -> str:
        return self._issue_id

    @property
    def node_id(self) -> NodeIdentifier:
        return self._node_id

    @property
    def title(self) -> str:
        return self._title

    @property
    def description(self) -> str:
        return self._description

    @property
    def metadata(self) -> MetadataRecord:
        return self._metadata

    @property
    def status(self) -> IssueStatus:
        return self._status

    @property
    def priority(self) -> str:
        return self._priority

    @property
    def progress(self) -> ProgressPercentage:
        return self._progress

    @property
    def sub_issues(self) -> tuple[SubIssue, ...]:
        return tuple(sorted(self._sub_issues, key=lambda s: s.relative_order))

    @property
    def proposed_solution(self) -> str | None:
        return self._proposed_solution

    def add_sub_issue(self, sub_issue: SubIssue) -> None:
        """Register a SubIssue under this Issue while enforcing uniqueness and parent linkage."""
        if not isinstance(sub_issue, SubIssue):
            raise DomainValidationError("sub_issue must be a SubIssue instance.")
        if sub_issue.parent_issue_id != self._issue_id:
            raise InvariantViolationError(
                f"SubIssue '{sub_issue.sub_issue_id}' parent_issue_id "
                f"('{sub_issue.parent_issue_id}') does not match Issue '{self._issue_id}'."
            )
        if any(s.sub_issue_id == sub_issue.sub_issue_id for s in self._sub_issues):
            raise InvariantViolationError(
                f"Duplicate SubIssue ID '{sub_issue.sub_issue_id}' in Issue '{self._issue_id}'."
            )
        if any(s.relative_order == sub_issue.relative_order for s in self._sub_issues):
            raise InvariantViolationError(
                f"Duplicate relative_order {sub_issue.relative_order} in Issue '{self._issue_id}'."
            )
        self._sub_issues.append(sub_issue)
        self.record_event(
            IssueDecomposedEvent(
                issue_id=self._issue_id,
                sub_issue_id=sub_issue.sub_issue_id,
                relative_order=sub_issue.relative_order,
            )
        )

    def propose_solution(self, solution_text: str) -> None:
        if not isinstance(solution_text, str) or not solution_text.strip():
            raise DomainValidationError("solution_text cannot be empty.")
        self._proposed_solution = solution_text.strip()
        if self._status in (
            IssueStatus.NEW,
            IssueStatus.ANALYZING,
            IssueStatus.KNOWLEDGE_IDENTIFIED,
            IssueStatus.EXECUTOR_IDENTIFIED,
        ):
            self._status = IssueStatus.SOLUTION_PROPOSED

    def recalculate_progress(
        self,
        has_supporting_knowledge: bool = False,
        has_executor_mapping: bool = False,
    ) -> ProgressPercentage:
        """Deterministically compute Issue progress based on Section 14.1 rules."""
        if self._status == IssueStatus.RESOLVED:
            self._progress = ProgressPercentage(100.0)
            return self._progress

        score = 0.0
        # 1. Sub-issue definition & average completion (up to 50%)
        if self._sub_issues:
            score += 15.0
            avg_sub = sum(s.progress.value for s in self._sub_issues) / len(self._sub_issues)
            score += 0.35 * avg_sub
        # 2. Supporting knowledge identified (20%)
        if has_supporting_knowledge or self._status == IssueStatus.KNOWLEDGE_IDENTIFIED:
            score += 20.0
        # 3. Executor mapped (15%)
        if has_executor_mapping or self._status == IssueStatus.EXECUTOR_IDENTIFIED:
            score += 15.0
        # 4. Proposed solution available (15%)
        if self._proposed_solution is not None or self._status == IssueStatus.SOLUTION_PROPOSED:
            score += 15.0

        self._progress = ProgressPercentage(min(score, 100.0))
        return self._progress
