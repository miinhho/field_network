from __future__ import annotations

from collections import defaultdict

import networkx as nx

from ..models import LayeredGraph, Perturbation, PropagationResult


class FlowSimulator:
    """PoC flow simulator using hop-based attenuation and simple rewiring."""

    def __init__(self, attenuation: float = 0.5, max_hops: int = 3, rewire_threshold: float = 0.25) -> None:
        if not (0 < attenuation < 1):
            raise ValueError("attenuation must be between 0 and 1")
        self.attenuation = attenuation
        self.max_hops = max_hops
        self.rewire_threshold = rewire_threshold

    def propagate(self, graph: LayeredGraph, perturbation: Perturbation) -> PropagationResult:
        nx_graph = nx.Graph()
        adjacency: dict[str, list[str]] = defaultdict(list)
        for edge in graph.interactions:
            nx_graph.add_edge(edge.source_id, edge.target_id, weight=edge.weight)
            adjacency[edge.source_id].append(edge.target_id)
            adjacency[edge.target_id].append(edge.source_id)

        impact: dict[str, float] = defaultdict(float)
        frontier = list(perturbation.targets)
        seen = set(frontier)
        for node in frontier:
            impact[node] = perturbation.intensity

        current_strength = perturbation.intensity
        hop_count = 0

        while frontier and hop_count < self.max_hops and current_strength > 0.01:
            hop_count += 1
            current_strength *= self.attenuation
            next_frontier: list[str] = []
            for src in frontier:
                for dst in nx_graph.neighbors(src) if src in nx_graph else []:
                    impact[dst] += current_strength
                    if dst not in seen:
                        seen.add(dst)
                        next_frontier.append(dst)
            frontier = next_frontier

        rewired_edges = self._suggest_rewiring(adjacency, impact)
        stabilized = current_strength <= 0.01 or hop_count >= self.max_hops
        return PropagationResult(
            impact_by_actant=dict(impact),
            hops_executed=hop_count,
            stabilized=stabilized,
            rewired_edges=rewired_edges,
        )

    def _suggest_rewiring(
        self,
        adjacency: dict[str, list[str]],
        impact: dict[str, float],
    ) -> list[tuple[str, str]]:
        candidates = sorted(impact.items(), key=lambda item: item[1], reverse=True)
        rewired: list[tuple[str, str]] = []
        for i in range(len(candidates) - 1):
            src, src_score = candidates[i]
            dst, dst_score = candidates[i + 1]
            if src == dst:
                continue
            if dst in adjacency.get(src, []):
                continue
            if min(src_score, dst_score) >= self.rewire_threshold:
                rewired.append((src, dst))
            if len(rewired) >= 3:
                break
        return rewired
