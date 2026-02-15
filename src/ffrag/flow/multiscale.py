from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import networkx as nx

from ..models import Actant, Interaction, LayeredGraph
from .control import TopologicalFlowController


@dataclass(slots=True)
class ClusterPlanResult:
    cluster_assignment: dict[str, str]
    cluster_control: dict[str, float]
    coarse_controlled_impact: dict[str, float]
    cluster_objective: float
    cross_scale_consistency: float


class ClusterFlowController:
    """Coarse-to-fine controller.

    1) Cluster graph into mesoscale groups
    2) Solve control at cluster level
    3) Project cluster control back to node impacts
    """

    def __init__(self) -> None:
        self.coarse_controller = TopologicalFlowController(
            k_div=0.28,
            k_phi=0.12,
            k_speed=0.12,
            k_cycle=0.1,
            k_curl=0.12,
            k_harmonic=0.1,
            k_higher=0.06,
            target_speed=0.4,
            residual_damping=0.45,
            control_clip=0.8,
        )

    def plan(
        self,
        graph: LayeredGraph,
        impact_by_actant: dict[str, float],
        state: dict[str, float],
    ) -> ClusterPlanResult:
        clusters = self._clusters(graph)
        assignment = self._assignment(clusters)
        cluster_graph = self._cluster_graph(graph, assignment)
        cluster_impact = self._cluster_impact(assignment, impact_by_actant)

        coarse_out = self.coarse_controller.compute(cluster_graph, cluster_impact, state)
        cluster_control = coarse_out.node_control

        coarse_controlled = {}
        for node in graph.actants.keys():
            c = assignment.get(node)
            base = float(impact_by_actant.get(node, 0.0))
            delta = float(cluster_control.get(c, 0.0)) if c else 0.0
            coarse_controlled[node] = max(0.0, base + delta)

        consistency = self._cross_scale_consistency(assignment, coarse_controlled)
        return ClusterPlanResult(
            cluster_assignment=assignment,
            cluster_control=cluster_control,
            coarse_controlled_impact=coarse_controlled,
            cluster_objective=coarse_out.objective_score,
            cross_scale_consistency=consistency,
        )

    def _clusters(self, graph: LayeredGraph) -> list[set[str]]:
        g = nx.Graph()
        for node in graph.actants.keys():
            g.add_node(node)
        for edge in graph.interactions:
            w = float(edge.weight)
            if g.has_edge(edge.source_id, edge.target_id):
                g[edge.source_id][edge.target_id]["weight"] += w
            else:
                g.add_edge(edge.source_id, edge.target_id, weight=w)

        if g.number_of_nodes() <= 2 or g.number_of_edges() == 0:
            return [{n} for n in g.nodes()]

        try:
            comms = list(nx.algorithms.community.greedy_modularity_communities(g, weight="weight"))
            if comms:
                return [set(c) for c in comms]
        except Exception:
            pass

        return [set(c) for c in nx.connected_components(g)]

    def _assignment(self, clusters: list[set[str]]) -> dict[str, str]:
        out: dict[str, str] = {}
        for i, cluster in enumerate(clusters):
            cid = f"cluster_{i}"
            for node in cluster:
                out[node] = cid
        return out

    def _cluster_graph(self, graph: LayeredGraph, assignment: dict[str, str]) -> LayeredGraph:
        out = LayeredGraph(graph_id=f"{graph.graph_id}:cluster", schema_version=graph.schema_version)
        now = datetime.now(timezone.utc)

        clusters = sorted(set(assignment.values()))
        for cid in clusters:
            out.actants[cid] = Actant(actant_id=cid, kind="cluster", label=cid)

        pair_weight: dict[tuple[str, str], float] = {}
        idx = 0
        for edge in graph.interactions:
            c1 = assignment.get(edge.source_id)
            c2 = assignment.get(edge.target_id)
            if not c1 or not c2 or c1 == c2:
                continue
            key = tuple(sorted((c1, c2)))
            pair_weight[key] = pair_weight.get(key, 0.0) + float(edge.weight)

        for (c1, c2), w in pair_weight.items():
            out.interactions.append(
                Interaction(
                    interaction_id=f"ce_{idx}",
                    timestamp=now,
                    source_id=c1,
                    target_id=c2,
                    layer="cluster",
                    weight=w,
                )
            )
            idx += 1

        if not out.interactions:
            # keep controller numerically stable with weak chain over clusters
            for i in range(len(clusters) - 1):
                out.interactions.append(
                    Interaction(
                        interaction_id=f"ce_fallback_{i}",
                        timestamp=now,
                        source_id=clusters[i],
                        target_id=clusters[i + 1],
                        layer="cluster",
                        weight=0.05,
                    )
                )

        return out

    def _cluster_impact(self, assignment: dict[str, str], node_impact: dict[str, float]) -> dict[str, float]:
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for node, cid in assignment.items():
            sums[cid] = sums.get(cid, 0.0) + float(node_impact.get(node, 0.0))
            counts[cid] = counts.get(cid, 0) + 1
        return {cid: (sums[cid] / max(1, counts[cid])) for cid in sums.keys()}

    def _cross_scale_consistency(self, assignment: dict[str, str], node_impact: dict[str, float]) -> float:
        by_cluster: dict[str, list[float]] = {}
        for node, cid in assignment.items():
            by_cluster.setdefault(cid, []).append(float(node_impact.get(node, 0.0)))

        if not by_cluster:
            return 0.0

        penalties = 0.0
        for values in by_cluster.values():
            if len(values) <= 1:
                continue
            mean_v = sum(values) / len(values)
            var = sum((v - mean_v) ** 2 for v in values) / len(values)
            penalties += var

        # Higher is better consistency (lower within-cluster variance)
        return 1.0 / (1.0 + penalties)
