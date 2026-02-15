from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from ..models import Interaction, LayeredGraph


@dataclass(slots=True)
class GraphAdjustmentResult:
    adjusted_graph: LayeredGraph
    strengthened_edges: int
    weakened_edges: int
    unchanged_edges: int
    mean_weight_shift: float
    suggested_new_edges: list[tuple[str, str]]
    suggested_drop_edges: list[tuple[str, str]]


class DynamicGraphAdjuster:
    """Adjusts graph connectivity based on flow impact and dynamic state signals."""

    def __init__(
        self,
        learning_rate: float = 0.25,
        min_weight: float = 0.05,
        max_weight: float = 5.0,
    ) -> None:
        self.learning_rate = learning_rate
        self.min_weight = min_weight
        self.max_weight = max_weight

    def adjust(
        self,
        graph: LayeredGraph,
        impact_by_actant: dict[str, float],
        state: dict[str, float],
    ) -> GraphAdjustmentResult:
        adjusted = LayeredGraph(
            graph_id=f"{graph.graph_id}:adjusted",
            schema_version=graph.schema_version,
            actants=dict(graph.actants),
            interactions=[],
        )

        viscosity = self._estimate_viscosity(state)
        instability = self._estimate_instability(state)

        strengthened = 0
        weakened = 0
        unchanged = 0
        shifts: list[float] = []

        for edge in graph.interactions:
            src_impact = float(impact_by_actant.get(edge.source_id, 0.0))
            dst_impact = float(impact_by_actant.get(edge.target_id, 0.0))
            pressure = 0.5 * (src_impact + dst_impact)

            delta = self.learning_rate * ((0.22 * pressure) + (0.10 * instability) - (0.08 * viscosity))
            if pressure < 0.08:
                delta -= self.learning_rate * 0.04

            new_weight = self._clip(edge.weight + delta)
            shift = new_weight - edge.weight
            shifts.append(shift)

            if shift > 1e-6:
                strengthened += 1
            elif shift < -1e-6:
                weakened += 1
            else:
                unchanged += 1

            adjusted.interactions.append(
                Interaction(
                    interaction_id=edge.interaction_id,
                    timestamp=edge.timestamp,
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    layer=edge.layer,
                    weight=round(new_weight, 6),
                    metadata=dict(edge.metadata),
                )
            )

        suggested_new = self._suggest_new_edges(graph, impact_by_actant)
        suggested_drop = self._suggest_drop_edges(adjusted, impact_by_actant)
        mean_shift = sum(shifts) / len(shifts) if shifts else 0.0

        return GraphAdjustmentResult(
            adjusted_graph=adjusted,
            strengthened_edges=strengthened,
            weakened_edges=weakened,
            unchanged_edges=unchanged,
            mean_weight_shift=round(mean_shift, 6),
            suggested_new_edges=suggested_new,
            suggested_drop_edges=suggested_drop,
        )

    def _suggest_new_edges(
        self,
        graph: LayeredGraph,
        impact_by_actant: dict[str, float],
        max_edges: int = 3,
    ) -> list[tuple[str, str]]:
        existing = set()
        neighbors: dict[str, set[str]] = {node: set() for node in graph.actants.keys()}
        for edge in graph.interactions:
            existing.add(tuple(sorted((edge.source_id, edge.target_id))))
            neighbors.setdefault(edge.source_id, set()).add(edge.target_id)
            neighbors.setdefault(edge.target_id, set()).add(edge.source_id)

        ranked_nodes = [n for n, _ in sorted(impact_by_actant.items(), key=lambda item: item[1], reverse=True)]
        if not ranked_nodes:
            return []

        # Vector-search inspired approximate retrieval:
        # 1) shortlist high-impact anchors
        # 2) compute cosine similarity in lightweight node embeddings
        # 3) rank by similarity + topological bridge score
        candidate_nodes = ranked_nodes[: min(12, len(ranked_nodes))]
        node_vec = {node: self._node_embedding(node, impact_by_actant, neighbors) for node in candidate_nodes}

        scored_pairs: list[tuple[float, tuple[str, str]]] = []
        for a, b in combinations(candidate_nodes, 2):
            key = tuple(sorted((a, b)))
            if a == b or key in existing:
                continue
            sim = self._cosine(node_vec[a], node_vec[b])
            bridge = self._bridge_score(a, b, neighbors)
            score = 0.72 * sim + 0.28 * bridge
            scored_pairs.append((score, (a, b)))

        scored_pairs.sort(key=lambda item: item[0], reverse=True)
        return [pair for _, pair in scored_pairs[:max_edges]]

    def _suggest_drop_edges(
        self,
        graph: LayeredGraph,
        impact_by_actant: dict[str, float],
        max_edges: int = 3,
    ) -> list[tuple[str, str]]:
        scored: list[tuple[float, tuple[str, str]]] = []
        for edge in graph.interactions:
            src_impact = float(impact_by_actant.get(edge.source_id, 0.0))
            dst_impact = float(impact_by_actant.get(edge.target_id, 0.0))
            score = edge.weight + 0.2 * (src_impact + dst_impact)
            scored.append((score, (edge.source_id, edge.target_id)))

        scored.sort(key=lambda item: item[0])
        return [edge for _, edge in scored[:max_edges]]

    def _node_embedding(
        self,
        node: str,
        impact_by_actant: dict[str, float],
        neighbors: dict[str, set[str]],
    ) -> np.ndarray:
        nbrs = neighbors.get(node, set())
        deg = float(len(nbrs))
        impact = float(impact_by_actant.get(node, 0.0))
        nbr_impact = sum(float(impact_by_actant.get(n, 0.0)) for n in nbrs)
        avg_nbr_impact = nbr_impact / max(1.0, deg)
        return np.array([impact, deg, avg_nbr_impact], dtype=np.float64)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-9:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _bridge_score(self, a: str, b: str, neighbors: dict[str, set[str]]) -> float:
        na = neighbors.get(a, set())
        nb = neighbors.get(b, set())
        if not na and not nb:
            return 0.0
        union = na | nb
        inter = na & nb
        if not union:
            return 0.0
        jaccard = len(inter) / len(union)
        # Prefer connecting complementary neighborhoods (structural holes).
        return 1.0 - jaccard

    def _estimate_viscosity(self, state: dict[str, float]) -> float:
        reg = float(state.get("temporal_regularity", 0.0))
        density = float(state.get("schedule_density", 0.0))
        return max(0.0, min(1.0, 0.55 * reg + 0.04 * density))

    def _estimate_instability(self, state: dict[str, float]) -> float:
        speed = float(state.get("transition_speed", 0.0))
        entropy = float(state.get("social_entropy", 0.0))
        regularity = float(state.get("temporal_regularity", 0.0))
        return max(0.0, (0.7 * speed + 0.3 * entropy) - 0.25 * regularity)

    def _clip(self, value: float) -> float:
        return max(self.min_weight, min(self.max_weight, value))
