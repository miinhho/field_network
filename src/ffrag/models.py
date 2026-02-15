from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


QueryType = str


@dataclass(slots=True)
class Actant:
    actant_id: str
    kind: str
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.actant_id:
            raise ValueError("actant_id is required")
        if not self.kind:
            raise ValueError("kind is required")


@dataclass(slots=True)
class Interaction:
    interaction_id: str
    timestamp: datetime
    source_id: str
    target_id: str
    layer: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.interaction_id:
            raise ValueError("interaction_id is required")
        if self.weight < 0:
            raise ValueError("weight must be >= 0")
        if not self.layer:
            raise ValueError("layer is required")


@dataclass(slots=True)
class LayeredGraph:
    graph_id: str
    schema_version: str
    actants: dict[str, Actant] = field(default_factory=dict)
    interactions: list[Interaction] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.graph_id:
            raise ValueError("graph_id is required")
        if not self.schema_version:
            raise ValueError("schema_version is required")


@dataclass(slots=True)
class StateVector:
    entity_id: str
    timestamp: datetime
    values: dict[str, float]

    def __post_init__(self) -> None:
        if not self.entity_id:
            raise ValueError("entity_id is required")
        if not self.values:
            raise ValueError("values is required")


@dataclass(slots=True)
class Perturbation:
    perturbation_id: str
    timestamp: datetime
    targets: list[str]
    intensity: float
    kind: str = "generic"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.perturbation_id:
            raise ValueError("perturbation_id is required")
        if self.intensity < 0:
            raise ValueError("intensity must be >= 0")


@dataclass(slots=True)
class PropagationResult:
    impact_by_actant: dict[str, float]
    hops_executed: int
    stabilized: bool
    rewired_edges: list[tuple[str, str]]


@dataclass(slots=True)
class Query:
    text: str
    query_type: QueryType | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Answer:
    query_type: QueryType
    claims: list[str]
    evidence_ids: list[str]
    metrics_used: dict[str, float]
    uncertainty: float
    blocked_by_guardrail: bool = False
    block_reason: str | None = None
