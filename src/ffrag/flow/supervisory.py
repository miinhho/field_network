from __future__ import annotations

from dataclasses import dataclass
import math

import networkx as nx

from ..models import LayeredGraph


@dataclass(slots=True)
class SupervisoryMetrics:
    confusion_score: float
    forgetting_score: float
    cluster_margin: float
    mixing_entropy: float
    retention_loss: float
    connectivity_loss: float
    important_node_count: int


@dataclass(slots=True)
class SupervisoryControlState:
    step_count: int = 0
    important_nodes: tuple[str, ...] = ()
    baseline_connectivity: float | None = None
    confusion_risk: float = 0.0
    forgetting_risk: float = 0.0


class SupervisoryMetricsAnalyzer:
    """Computes confusion/forgetting proxies for supervisory control."""

    def analyze(
        self,
        graph: LayeredGraph,
        impact_by_actant: dict[str, float],
        state: SupervisoryControlState | None = None,
        important_nodes: list[str] | tuple[str, ...] | None = None,
        important_fraction: float = 0.2,
    ) -> tuple[SupervisoryMetrics, SupervisoryControlState]:
        current = state or SupervisoryControlState()
        g = self._to_networkx(graph)
        communities = self._communities(g)

        margin = self._cluster_margin(communities, impact_by_actant)
        mixing = self._mixing_entropy(g, communities)
        confusion = self._clip(0.55 * (1.0 - margin) + 0.45 * mixing)

        selected_important = self._important_nodes(
            graph=graph,
            impact_by_actant=impact_by_actant,
            provided=important_nodes,
            previous=current.important_nodes,
            fraction=important_fraction,
        )
        retention_loss = self._retention_loss(g, selected_important)
        connectivity = self._important_connectivity(g, selected_important)
        baseline = connectivity if current.baseline_connectivity is None else current.baseline_connectivity
        connectivity_loss = self._clip(max(0.0, baseline - connectivity))
        forgetting = self._clip(0.6 * retention_loss + 0.4 * connectivity_loss)

        metrics = SupervisoryMetrics(
            confusion_score=round(confusion, 6),
            forgetting_score=round(forgetting, 6),
            cluster_margin=round(margin, 6),
            mixing_entropy=round(mixing, 6),
            retention_loss=round(retention_loss, 6),
            connectivity_loss=round(connectivity_loss, 6),
            important_node_count=len(selected_important),
        )
        next_state = SupervisoryControlState(
            step_count=current.step_count + 1,
            important_nodes=tuple(selected_important),
            baseline_connectivity=float(baseline),
            confusion_risk=metrics.confusion_score,
            forgetting_risk=metrics.forgetting_score,
        )
        return metrics, next_state

    def _to_networkx(self, graph: LayeredGraph) -> nx.Graph:
        g = nx.Graph()
        for node in graph.actants.keys():
            g.add_node(node)
        for edge in graph.interactions:
            if edge.source_id == edge.target_id:
                continue
            if g.has_edge(edge.source_id, edge.target_id):
                g[edge.source_id][edge.target_id]["weight"] += float(edge.weight)
            else:
                g.add_edge(edge.source_id, edge.target_id, weight=float(edge.weight))
        return g

    def _communities(self, g: nx.Graph) -> list[set[str]]:
        if g.number_of_nodes() == 0:
            return []
        if g.number_of_edges() == 0:
            return [{n} for n in g.nodes()]
        try:
            comms = list(nx.algorithms.community.greedy_modularity_communities(g, weight="weight"))
            if comms:
                return [set(c) for c in comms]
        except Exception:
            pass
        return [set(c) for c in nx.connected_components(g)]

    def _cluster_margin(self, communities: list[set[str]], impact_by_actant: dict[str, float]) -> float:
        if not communities:
            return 0.0
        means: list[float] = []
        for c in communities:
            if not c:
                continue
            vals = [float(impact_by_actant.get(node, 0.0)) for node in c]
            means.append(sum(vals) / max(1, len(vals)))
        if not means:
            return 0.0
        means.sort(reverse=True)
        top = means[0]
        second = means[1] if len(means) > 1 else 0.0
        denom = abs(top) + abs(second) + 1e-9
        return self._clip((top - second) / denom)

    def _mixing_entropy(self, g: nx.Graph, communities: list[set[str]]) -> float:
        if g.number_of_nodes() == 0 or g.number_of_edges() == 0 or len(communities) <= 1:
            return 0.0
        cluster_of: dict[str, int] = {}
        for i, comm in enumerate(communities):
            for node in comm:
                cluster_of[node] = i

        entropy_weighted = 0.0
        degree_weight = 0.0
        for node in g.nodes():
            deg = g.degree(node)
            if deg <= 0:
                continue
            cnt: dict[int, int] = {}
            for nbr in g.neighbors(node):
                cid = cluster_of.get(nbr, -1)
                cnt[cid] = cnt.get(cid, 0) + 1
            probs = [v / deg for v in cnt.values() if v > 0]
            ent = -sum(p * math.log(p + 1e-12) for p in probs)
            max_ent = math.log(max(1, len(communities)))
            ent_norm = 0.0 if max_ent <= 1e-12 else ent / max_ent
            entropy_weighted += deg * ent_norm
            degree_weight += deg

        if degree_weight <= 1e-12:
            return 0.0
        return self._clip(entropy_weighted / degree_weight)

    def _important_nodes(
        self,
        graph: LayeredGraph,
        impact_by_actant: dict[str, float],
        provided: list[str] | tuple[str, ...] | None,
        previous: tuple[str, ...],
        fraction: float,
    ) -> list[str]:
        if provided:
            return [n for n in provided if n in graph.actants]
        if previous:
            return [n for n in previous if n in graph.actants]
        nodes = list(graph.actants.keys())
        if not nodes:
            return []
        ranked = sorted(nodes, key=lambda n: abs(float(impact_by_actant.get(n, 0.0))), reverse=True)
        k = max(1, int(round(len(nodes) * max(0.05, min(0.5, fraction)))))
        return ranked[:k]

    def _retention_loss(self, g: nx.Graph, important_nodes: list[str]) -> float:
        if not important_nodes:
            return 0.0
        retained = 0
        for node in important_nodes:
            if node in g and g.degree(node) > 0:
                retained += 1
        return self._clip(1.0 - (retained / max(1, len(important_nodes))))

    def _important_connectivity(self, g: nx.Graph, important_nodes: list[str]) -> float:
        if not important_nodes:
            return 1.0
        present = [n for n in important_nodes if n in g]
        if not present:
            return 0.0
        sub = g.subgraph(present)
        if sub.number_of_nodes() <= 1:
            return 1.0
        largest = max((len(c) for c in nx.connected_components(sub)), default=1)
        return self._clip(largest / sub.number_of_nodes())

    def _clip(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))
