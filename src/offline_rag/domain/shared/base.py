"""Foundational DDD building blocks: Entity, AggregateRoot, DomainEvent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class DomainEvent:
    """Base class for immutable domain events."""

    event_id: str = field(default_factory=lambda: str(uuid4()), kw_only=True)
    occurred_at: datetime = field(default_factory=utc_now, kw_only=True)

    @property
    def event_name(self) -> str:
        return self.__class__.__name__


class Entity:
    """Base class for domain entities with identity-based equality."""

    def __init__(self, entity_id: Any) -> None:
        if entity_id is None or (isinstance(entity_id, str) and not entity_id.strip()):
            raise ValueError("Entity identity cannot be empty or None.")
        self._id = entity_id

    @property
    def id(self) -> Any:
        return self._id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._id == other._id

    def __hash__(self) -> int:
        return hash((self.__class__, self._id))


class AggregateRoot(Entity):
    """Base class for Aggregate Roots managing domain events and transactional invariants."""

    def __init__(self, entity_id: Any) -> None:
        super().__init__(entity_id)
        self._domain_events: list[DomainEvent] = []

    def record_event(self, event: DomainEvent) -> None:
        """Register a domain event emitted by a state transition."""
        self._domain_events.append(event)

    def pull_domain_events(self) -> list[DomainEvent]:
        """Return and clear all pending domain events."""
        events = list(self._domain_events)
        self._domain_events.clear()
        return events

    @property
    def domain_events(self) -> tuple[DomainEvent, ...]:
        """Read-only view of pending domain events."""
        return tuple(self._domain_events)
