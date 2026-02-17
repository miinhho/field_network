from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math

import networkx as nx
import numpy as np

from ..ann_index import create_cosine_ann_index
from ..models import Actant, Interaction, LayeredGraph
from .control import TopologicalFlowController


@dataclass(slots=True)
class ClusterPlanResult:
    cluster_assignment: dict[str, str]
    cluster_control: dict[str, float]
    coarse_controlled_impact: dict[str, float]
    cluster_objective: float
    cross_scale_consistency: float
    ann_cache_hit: float
    active_context_count: int
    evicted_context_count: int


@dataclass(slots=True)
class _AnnCacheEntry:
    node_ids: tuple[str, ...]
    feature_signature: int
    index: object
    backend_name: str


@dataclass(slots=True)
class _ContextStats:
    last_seen_step: int
    hit_count: int
    importance: float


class ClusterFlowController:
    """Coarse-to-fine controller.

    1) Cluster graph into mesoscale groups
    2) Solve control at cluster level
    3) Project cluster control back to node impacts
    """

    def __init__(self) -> None:
        self.cluster_mode = "hybrid_knn"
        self.hybrid_structure_weight = 0.65
        self.hybrid_dynamic_weight = 0.35
        self.hybrid_knn_k = 4
        self.hybrid_mutual_knn = True
        self.hybrid_temporal_inertia = 0.16
        self.hybrid_ann_backend = "auto"
        self.hybrid_ann_strict = True
        self.hybrid_ann_candidate_k = 12
        self.hybrid_ann_use_ivf = False
        self.hybrid_ann_nlist = 64
        self.hybrid_ann_nprobe = 8
        self.hybrid_pattern_separation_beta = 0.65
        self.last_ann_backend = "none"
        self.cluster_impact_coherence_floor = 0.3
        self.coarse_projection_clip = 5.0
        self.cross_scale_sign_weight = 0.55
        self.max_context_states = 128
        self.context_half_life_steps = 40.0
        self.context_frequency_weight = 0.25
        self.context_retention_floor = 0.05
        self._prev_assignment_by_context: dict[str, dict[str, str]] = {}
        self._ann_cache_by_context: dict[str, _AnnCacheEntry] = {}
        self._context_stats: dict[str, _ContextStats] = {}
        self._context_importance_override: dict[str, float] = {}
        self._step_counter = 0
        self._last_ann_cache_hit = 0.0
        self._last_evicted_context_count = 0
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
        phase_signal: float = 0.0,
        context_id: str | None = None,
    ) -> ClusterPlanResult:
        self._step_counter += 1
        ctx = self._context_key(graph.graph_id, context_id)
        hybrid_graph = self._hybrid_graph(graph, impact_by_actant, ctx)
        clusters = self._clusters_from_graph(hybrid_graph)
        assignment = self._assignment(clusters)
        self._remember_context_state(ctx, assignment, impact_by_actant)
        cluster_graph = self._cluster_graph(graph, assignment, hybrid_graph)
        cluster_impact = self._cluster_impact(assignment, impact_by_actant)

        coarse_out = self.coarse_controller.compute(cluster_graph, cluster_impact, state, phase_signal=phase_signal)
        cluster_control = coarse_out.node_control

        coarse_controlled = {}
        for node in graph.actants.keys():
            c = assignment.get(node)
            base = float(impact_by_actant.get(node, 0.0))
            delta = float(cluster_control.get(c, 0.0)) if c else 0.0
            coarse_controlled[node] = self._project_cluster_control(base, delta)

        consistency = self._cross_scale_consistency(assignment, coarse_controlled)
        return ClusterPlanResult(
            cluster_assignment=assignment,
            cluster_control=cluster_control,
            coarse_controlled_impact=coarse_controlled,
            cluster_objective=coarse_out.objective_score,
            cross_scale_consistency=consistency,
            ann_cache_hit=float(self._last_ann_cache_hit),
            active_context_count=len(self._context_stats),
            evicted_context_count=int(self._last_evicted_context_count),
        )

    def _clusters_from_graph(self, g: nx.Graph) -> list[set[str]]:
        if g.number_of_nodes() <= 2 or g.number_of_edges() == 0:
            return [{n} for n in g.nodes()]

        try:
            comms = list(nx.algorithms.community.greedy_modularity_communities(g, weight="weight"))
            if comms:
                return [set(c) for c in comms]
        except Exception:
            pass

        return [set(c) for c in nx.connected_components(g)]

    def _hybrid_graph(
        self,
        graph: LayeredGraph,
        impact_by_actant: dict[str, float],
        context_key: str,
    ) -> nx.Graph:
        struct = nx.Graph()
        for node in graph.actants.keys():
            struct.add_node(node)
        for edge in graph.interactions:
            w = max(0.0, float(edge.weight))
            if struct.has_edge(edge.source_id, edge.target_id):
                struct[edge.source_id][edge.target_id]["weight"] += w
            else:
                struct.add_edge(edge.source_id, edge.target_id, weight=w)

        out = nx.Graph()
        for node in struct.nodes():
            out.add_node(node)
        max_struct = max((float(data.get("weight", 0.0)) for _, _, data in struct.edges(data=True)), default=1.0)
        for src, dst, data in struct.edges(data=True):
            w = float(data.get("weight", 0.0)) / max(1e-9, max_struct)
            self._add_weight(out, src, dst, self.hybrid_structure_weight * w)

        if self.cluster_mode == "hybrid_knn":
            self._add_dynamic_knn_edges(out, struct, impact_by_actant, context_key)

        self._add_temporal_inertia_edges(out, context_key, impact_by_actant)
        return out

    def _add_dynamic_knn_edges(
        self,
        out: nx.Graph,
        struct: nx.Graph,
        impact_by_actant: dict[str, float],
        context_key: str,
    ) -> None:
        nodes = list(out.nodes())
        if len(nodes) < 2:
            return
        feats = {node: self._node_feature(node, struct, impact_by_actant) for node in nodes}
        k = max(1, min(self.hybrid_knn_k, len(nodes) - 1))
        neighbors: dict[str, list[tuple[float, str]]] = {node: [] for node in nodes}
        node_ids = list(nodes)
        matrix = np.vstack([feats[n] for n in node_ids]).astype(np.float64, copy=False)
        signature = self._feature_signature(matrix)
        cached = self._ann_cache_by_context.get(context_key)
        if cached and cached.node_ids == tuple(node_ids) and cached.feature_signature == signature:
            index = cached.index
            self.last_ann_backend = cached.backend_name
            self._last_ann_cache_hit = 1.0
        else:
            index = create_cosine_ann_index(
                backend=self.hybrid_ann_backend,
                allow_exact_fallback=not self.hybrid_ann_strict,
                faiss_use_ivf=self.hybrid_ann_use_ivf,
                faiss_nlist=self.hybrid_ann_nlist,
                faiss_nprobe=self.hybrid_ann_nprobe,
            )
            index.fit(node_ids, matrix)
            self.last_ann_backend = index.backend_name
            self._ann_cache_by_context[context_key] = _AnnCacheEntry(
                node_ids=tuple(node_ids),
                feature_signature=signature,
                index=index,
                backend_name=self.last_ann_backend,
            )
            self._last_ann_cache_hit = 0.0
        query_k = max(k + 1, min(len(node_ids), self.hybrid_ann_candidate_k))
        for node in node_ids:
            hits = index.query(feats[node], k=query_k)
            for hit in hits:
                if hit.item_id == node:
                    continue
                neighbors[node].append((self._affinity_score(hit.score), hit.item_id))

        topk: dict[str, set[str]] = {}
        top_scores: dict[str, list[tuple[float, str]]] = {}
        for node, items in neighbors.items():
            items.sort(key=lambda x: x[0], reverse=True)
            dedup: list[tuple[float, str]] = []
            seen: set[str] = set()
            for score, dst in items:
                if dst in seen:
                    continue
                seen.add(dst)
                dedup.append((score, dst))
                if len(dedup) >= k:
                    break
            top_scores[node] = dedup
            topk[node] = {dst for _, dst in dedup}

        for src in nodes:
            if not top_scores.get(src):
                continue
            runner_up = top_scores[src][1][0] if len(top_scores[src]) >= 2 else 0.0
            for score, dst in top_scores[src]:
                if self.hybrid_mutual_knn and src not in topk.get(dst, set()):
                    continue
                # Pattern-separation gate:
                # if neighbors are too similarly strong (high confusion), suppress but do not hard-block.
                margin = score - self.hybrid_pattern_separation_beta * runner_up
                sep = max(0.05, margin)
                self._add_weight(out, src, dst, self.hybrid_dynamic_weight * sep)

    def _add_temporal_inertia_edges(
        self,
        out: nx.Graph,
        context_key: str,
        impact_by_actant: dict[str, float],
    ) -> None:
        prev = self._prev_assignment_by_context.get(context_key)
        if not prev:
            return
        by_cluster: dict[str, list[str]] = {}
        for node, cid in prev.items():
            if node not in out:
                continue
            by_cluster.setdefault(cid, []).append(node)
        for members in by_cluster.values():
            if len(members) < 2:
                continue
            ranked = sorted(members, key=lambda n: abs(float(impact_by_actant.get(n, 0.0))), reverse=True)
            for i in range(len(ranked) - 1):
                self._add_weight(out, ranked[i], ranked[i + 1], self.hybrid_temporal_inertia)

    def _node_feature(
        self,
        node: str,
        struct: nx.Graph,
        impact_by_actant: dict[str, float],
    ) -> np.ndarray:
        impact = float(impact_by_actant.get(node, 0.0))
        max_imp = max((abs(float(v)) for v in impact_by_actant.values()), default=1.0) or 1.0
        impact_n = impact / max_imp
        deg = float(struct.degree(node, weight="weight"))
        deg_n = math.log1p(max(0.0, deg))
        cluster_coeff = float(nx.clustering(struct, node, weight="weight")) if struct.number_of_edges() > 0 else 0.0
        nbrs = list(struct.neighbors(node)) if node in struct else []
        if nbrs:
            nbr_imp = sum(float(impact_by_actant.get(n, 0.0)) for n in nbrs) / len(nbrs)
        else:
            nbr_imp = 0.0
        nbr_imp_n = nbr_imp / max_imp
        return np.array([impact_n, deg_n, cluster_coeff, nbr_imp_n], dtype=np.float64)

    def _affinity_score(self, cosine_score: float) -> float:
        return max(0.0, min(1.0, 0.5 * (float(cosine_score) + 1.0)))

    def _feature_signature(self, matrix: np.ndarray) -> int:
        quantized = np.round(matrix, 6)
        return hash((quantized.shape, quantized.tobytes()))

    def _context_key(self, graph_id: str, context_id: str | None) -> str:
        c = (context_id or "").strip()
        if c:
            return c
        return graph_id

    def _remember_context_state(
        self,
        context_key: str,
        assignment: dict[str, str],
        impact_by_actant: dict[str, float],
    ) -> None:
        self._prev_assignment_by_context[context_key] = dict(assignment)
        if context_key in self._context_stats:
            st = self._context_stats[context_key]
            st.last_seen_step = self._step_counter
            st.hit_count += 1
            st.importance = self._default_context_importance(assignment, impact_by_actant, context_key)
        else:
            self._context_stats[context_key] = _ContextStats(
                last_seen_step=self._step_counter,
                hit_count=1,
                importance=self._default_context_importance(assignment, impact_by_actant, context_key),
            )
        self._evict_contexts_if_needed()

    def _default_context_importance(
        self,
        assignment: dict[str, str],
        impact_by_actant: dict[str, float],
        context_key: str,
    ) -> float:
        override = self._context_importance_override.get(context_key)
        if override is not None:
            return max(0.01, float(override))
        if not assignment:
            return 1.0
        unique_clusters = len(set(assignment.values()))
        mean_abs_impact = sum(abs(float(v)) for v in impact_by_actant.values()) / max(1, len(impact_by_actant))
        return max(0.1, 0.8 + 0.15 * unique_clusters + 0.1 * mean_abs_impact)

    def _retention_score(self, key: str, st: _ContextStats) -> float:
        age = max(0.0, float(self._step_counter - st.last_seen_step))
        half_life = max(1.0, float(self.context_half_life_steps))
        decay = math.exp(-math.log(2.0) * (age / half_life))
        freq_boost = 1.0 + self.context_frequency_weight * math.log1p(max(0, st.hit_count))
        return max(0.0, st.importance * decay * freq_boost)

    def _evict_contexts_if_needed(self) -> None:
        limit = max(1, int(self.max_context_states))
        evicted_count = 0
        stale_keys = [
            key
            for key, st in self._context_stats.items()
            if self._retention_score(key, st) < max(0.0, float(self.context_retention_floor))
        ]
        for victim in stale_keys:
            self._prev_assignment_by_context.pop(victim, None)
            self._ann_cache_by_context.pop(victim, None)
            self._context_stats.pop(victim, None)
            self._context_importance_override.pop(victim, None)
            evicted_count += 1
        while len(self._context_stats) > limit:
            victim = min(self._context_stats.keys(), key=lambda k: self._retention_score(k, self._context_stats[k]))
            self._prev_assignment_by_context.pop(victim, None)
            self._ann_cache_by_context.pop(victim, None)
            self._context_stats.pop(victim, None)
            self._context_importance_override.pop(victim, None)
            evicted_count += 1
        self._last_evicted_context_count = evicted_count

    def clear_context_state(self, context_id: str) -> None:
        key = context_id.strip()
        if not key:
            return
        self._prev_assignment_by_context.pop(key, None)
        self._ann_cache_by_context.pop(key, None)
        self._context_stats.pop(key, None)
        self._context_importance_override.pop(key, None)

    def set_context_importance(self, context_id: str, importance: float) -> None:
        key = context_id.strip()
        if not key:
            return
        self._context_importance_override[key] = max(0.01, float(importance))

    def _add_weight(self, g: nx.Graph, src: str, dst: str, w: float) -> None:
        if src == dst or w <= 0.0:
            return
        if g.has_edge(src, dst):
            g[src][dst]["weight"] += w
        else:
            g.add_edge(src, dst, weight=w)

    def _assignment(self, clusters: list[set[str]]) -> dict[str, str]:
        out: dict[str, str] = {}
        for i, cluster in enumerate(clusters):
            cid = f"cluster_{i}"
            for node in cluster:
                out[node] = cid
        return out

    def _cluster_graph(
        self,
        graph: LayeredGraph,
        assignment: dict[str, str],
        source_graph: nx.Graph,
    ) -> LayeredGraph:
        out = LayeredGraph(graph_id=f"{graph.graph_id}:cluster", schema_version=graph.schema_version)
        now = datetime.now(timezone.utc)

        clusters = sorted(set(assignment.values()))
        for cid in clusters:
            out.actants[cid] = Actant(actant_id=cid, kind="cluster", label=cid)

        pair_weight: dict[tuple[str, str], float] = {}
        idx = 0
        for src, dst, data in source_graph.edges(data=True):
            c1 = assignment.get(src)
            c2 = assignment.get(dst)
            if not c1 or not c2 or c1 == c2:
                continue
            key = tuple(sorted((c1, c2)))
            pair_weight[key] = pair_weight.get(key, 0.0) + float(data.get("weight", 0.0))

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
        values_by_cluster: dict[str, list[float]] = {}
        counts: dict[str, int] = {}
        for node, cid in assignment.items():
            values_by_cluster.setdefault(cid, []).append(float(node_impact.get(node, 0.0)))
            counts[cid] = counts.get(cid, 0) + 1
        out: dict[str, float] = {}
        floor = max(0.0, min(1.0, float(self.cluster_impact_coherence_floor)))
        for cid, values in values_by_cluster.items():
            n = max(1, counts.get(cid, len(values)))
            signed_sum = sum(values)
            abs_sum = sum(abs(v) for v in values)
            mean_abs = abs_sum / n
            if abs_sum <= 1e-12:
                out[cid] = 0.0
                continue
            coherence = abs(signed_sum) / abs_sum
            if abs(signed_sum) > 1e-12:
                direction = 1.0 if signed_sum > 0.0 else -1.0
            else:
                pivot = max(values, key=lambda x: abs(x))
                direction = 1.0 if pivot >= 0.0 else -1.0
            magnitude = mean_abs * (floor + (1.0 - floor) * coherence)
            out[cid] = direction * magnitude
        return out

    def _cross_scale_consistency(self, assignment: dict[str, str], node_impact: dict[str, float]) -> float:
        by_cluster: dict[str, list[float]] = {}
        for node, cid in assignment.items():
            by_cluster.setdefault(cid, []).append(float(node_impact.get(node, 0.0)))

        if not by_cluster:
            return 0.0

        sign_w = max(0.0, min(1.0, float(self.cross_scale_sign_weight)))
        scores: list[float] = []
        weights: list[float] = []
        for values in by_cluster.values():
            if not values:
                continue
            n = len(values)
            abs_sum = sum(abs(v) for v in values)
            if abs_sum <= 1e-12:
                sign_consistency = 1.0
                magnitude_stability = 1.0
            else:
                signed_sum = sum(values)
                sign_consistency = max(0.0, min(1.0, abs(signed_sum) / abs_sum))
                mean_v = signed_sum / n
                var = sum((v - mean_v) ** 2 for v in values) / n
                mean_abs = abs_sum / n
                norm_var = var / (mean_abs * mean_abs + 1e-9)
                magnitude_stability = 1.0 / (1.0 + norm_var)
            score = sign_w * sign_consistency + (1.0 - sign_w) * magnitude_stability
            scores.append(float(np.clip(score, 0.0, 1.0)))
            weights.append(float(n))

        if not scores:
            return 0.0
        weight_sum = sum(weights)
        if weight_sum <= 1e-12:
            return sum(scores) / len(scores)
        return sum(s * w for s, w in zip(scores, weights)) / weight_sum

    def _project_cluster_control(self, base_impact: float, cluster_delta: float) -> float:
        clip = max(0.1, float(self.coarse_projection_clip))
        value = float(base_impact) + float(cluster_delta)
        return float(np.clip(value, -clip, clip))
