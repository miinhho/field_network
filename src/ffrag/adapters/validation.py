from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..models import LayeredGraph


@dataclass(slots=True)
class GraphValidationIssue:
    code: str
    message: str
    interaction_id: str | None = None


@dataclass(slots=True)
class GraphValidationResult:
    valid: bool
    error_count: int
    warning_count: int
    issues: list[GraphValidationIssue]


class GraphContractValidator:
    """Validates canonical graph contract for adapter outputs."""

    def __init__(
        self,
        min_weight: float = 0.0,
        max_weight: float = 5.0,
        allow_self_loop: bool = False,
    ) -> None:
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.allow_self_loop = allow_self_loop

    def validate(self, graph: LayeredGraph) -> GraphValidationResult:
        issues: list[GraphValidationIssue] = []

        if not graph.graph_id:
            issues.append(GraphValidationIssue(code="graph_id_missing", message="graph_id is required"))
        if not graph.schema_version:
            issues.append(GraphValidationIssue(code="schema_missing", message="schema_version is required"))

        node_ids = set(graph.actants.keys())
        seen_interactions: set[str] = set()
        last_ts_by_pair: dict[tuple[str, str, str], datetime] = {}

        for edge in graph.interactions:
            if edge.interaction_id in seen_interactions:
                issues.append(
                    GraphValidationIssue(
                        code="duplicate_interaction_id",
                        message="duplicate interaction_id",
                        interaction_id=edge.interaction_id,
                    )
                )
            seen_interactions.add(edge.interaction_id)

            if edge.source_id not in node_ids or edge.target_id not in node_ids:
                issues.append(
                    GraphValidationIssue(
                        code="dangling_edge_node",
                        message="edge endpoint not in actants",
                        interaction_id=edge.interaction_id,
                    )
                )
            if (not self.allow_self_loop) and edge.source_id == edge.target_id:
                issues.append(
                    GraphValidationIssue(
                        code="self_loop_not_allowed",
                        message="self loop is not allowed by contract",
                        interaction_id=edge.interaction_id,
                    )
                )
            if edge.weight < self.min_weight or edge.weight > self.max_weight:
                issues.append(
                    GraphValidationIssue(
                        code="weight_out_of_range",
                        message=f"weight must be in [{self.min_weight}, {self.max_weight}]",
                        interaction_id=edge.interaction_id,
                    )
                )

            key = (edge.source_id, edge.target_id, edge.layer)
            prev_ts = last_ts_by_pair.get(key)
            if prev_ts is not None and edge.timestamp < prev_ts:
                issues.append(
                    GraphValidationIssue(
                        code="timestamp_backwards",
                        message="interaction timestamp goes backwards for same (src,dst,layer)",
                        interaction_id=edge.interaction_id,
                    )
                )
            if prev_ts is None or edge.timestamp > prev_ts:
                last_ts_by_pair[key] = edge.timestamp

        errors = len(issues)
        return GraphValidationResult(
            valid=errors == 0,
            error_count=errors,
            warning_count=0,
            issues=issues,
        )
