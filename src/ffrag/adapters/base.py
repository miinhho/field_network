from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..models import LayeredGraph, Perturbation
from .validation import GraphContractValidator, GraphValidationResult


@dataclass(slots=True)
class AdapterBuildResult:
    graph: LayeredGraph
    default_perturbation: Perturbation
    validation: GraphValidationResult
    mapping_report: dict[str, float] = field(default_factory=dict)


class BaseAdapter(ABC):
    """Base class for domain adapters that emit canonical LayeredGraph."""

    def __init__(self, validator: GraphContractValidator | None = None) -> None:
        self.validator = validator or GraphContractValidator()

    @abstractmethod
    def to_graph(self) -> LayeredGraph:
        raise NotImplementedError

    @abstractmethod
    def default_perturbation(self) -> Perturbation:
        raise NotImplementedError

    def mapping_report(self) -> dict[str, float]:
        return {}

    def build(self) -> AdapterBuildResult:
        graph = self.to_graph()
        validation = self.validator.validate(graph)
        return AdapterBuildResult(
            graph=graph,
            default_perturbation=self.default_perturbation(),
            validation=validation,
            mapping_report=self.mapping_report(),
        )
