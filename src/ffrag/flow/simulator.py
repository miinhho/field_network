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
        edge_polarity: dict[tuple[str, str], float] = {}
        for edge in graph.interactions:
            nx_graph.add_edge(edge.source_id, edge.target_id, weight=edge.weight)
            adjacency[edge.source_id].append(edge.target_id)
            adjacency[edge.target_id].append(edge.source_id)
            key = tuple(sorted((edge.source_id, edge.target_id)))
            edge_polarity[key] = self._edge_polarity(edge.metadata)

        impact: dict[str, float] = defaultdict(float)
        target_seed = self._target_seed_strengths(perturbation)
        frontier = list(target_seed.keys())
        seen = set(frontier)
        for node in frontier:
            impact[node] = target_seed[node]

        current_strength = max(abs(v) for v in target_seed.values()) if target_seed else perturbation.intensity
        hop_count = 0

        while frontier and hop_count < self.max_hops and current_strength > 0.01:
            hop_count += 1
            current_strength *= self.attenuation
            next_frontier: list[str] = []
            for src in frontier:
                for dst in nx_graph.neighbors(src) if src in nx_graph else []:
                    key = tuple(sorted((src, dst)))
                    polarity = edge_polarity.get(key, 1.0)
                    src_sign = 1.0 if impact[src] >= 0.0 else -1.0
                    impact[dst] += current_strength * src_sign * polarity
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
        candidates = sorted(impact.items(), key=lambda item: abs(item[1]), reverse=True)
        rewired: list[tuple[str, str]] = []
        for i in range(len(candidates) - 1):
            src, src_score = candidates[i]
            dst, dst_score = candidates[i + 1]
            if src == dst:
                continue
            if dst in adjacency.get(src, []):
                continue
            if src_score * dst_score < 0.0:
                continue
            if min(abs(src_score), abs(dst_score)) >= self.rewire_threshold:
                rewired.append((src, dst))
            if len(rewired) >= 3:
                break
        return rewired

    def _target_seed_strengths(self, perturbation: Perturbation) -> dict[str, float]:
        weights = perturbation.metadata.get("target_weights", {})
        out: dict[str, float] = {}
        for node in perturbation.targets:
            w = 1.0
            if isinstance(weights, dict):
                try:
                    w = float(weights.get(node, 1.0))
                except (TypeError, ValueError):
                    w = 1.0
            out[node] = float(perturbation.intensity) * w
        return out

    def _edge_polarity(self, metadata: dict[str, object]) -> float:
        raw = metadata.get("polarity", 1.0)
        try:
            p = float(raw)
        except (TypeError, ValueError):
            p = 1.0
        if p >= 0.0:
            return 1.0
        return -1.0
