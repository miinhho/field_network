from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations
from typing import Iterable

import numpy as np

from ..models import Interaction, LayeredGraph


@dataclass(slots=True)
class GraphAdjustmentResult:
    adjusted_graph: LayeredGraph
    strengthened_edges: int
    weakened_edges: int
    unchanged_edges: int
    mean_weight_shift: float
    adjustment_objective_score: float
    selected_adjustment_scale: float
    selected_planner_horizon: int
    selected_edit_budget: int
    graph_density: float
    impact_noise: float
    coupling_penalty: float
    objective_terms: dict[str, float]
    applied_new_edges: int
    applied_drop_edges: int
    blocked_drop_edges: int
    suggested_new_edges: list[tuple[str, str]]
    suggested_drop_edges: list[tuple[str, str]]


@dataclass(slots=True)
class AdjustmentPlannerConfig:
    # Objective term base weights
    churn_weight_base: float = 0.38
    churn_weight_density_gain: float = 0.22
    volatility_weight_base: float = 0.28
    volatility_weight_noise_gain: float = 0.28
    rewiring_weight_base: float = 0.03
    rewiring_weight_density_gain: float = 0.08
    risk_weight_base: float = 0.28
    risk_weight_gain: float = 0.22
    coupling_weight_base: float = 0.2
    coupling_weight_noise_gain: float = 0.2
    # Rollout dynamics
    rollout_discount: float = 0.72
    under_adjust_penalty: float = 0.8
    over_adjust_penalty: float = 0.6
    # Candidate sets and thresholds
    default_scale_candidates: tuple[float, ...] = (0.5, 0.8, 1.0, 1.2)
    high_risk_scale_candidates: tuple[float, ...] = (0.4, 0.55, 0.7, 0.85)
    sparse_scale_candidates: tuple[float, ...] = (0.7, 1.0, 1.25, 1.5)
    dense_or_noisy_scale_candidates: tuple[float, ...] = (0.45, 0.65, 0.85, 1.0)
    high_risk_threshold: float = 0.75
    dense_threshold: float = 0.65
    sparse_threshold: float = 0.2
    noisy_threshold: float = 0.7
    very_noisy_threshold: float = 0.75


class DynamicGraphAdjuster:
    """Adjusts graph connectivity based on flow impact and dynamic state signals."""

    def __init__(
        self,
        learning_rate: float = 0.25,
        min_weight: float = 0.05,
        max_weight: float = 5.0,
        max_structural_edits: int = 2,
        apply_structural_edits: bool = True,
        planner_config: AdjustmentPlannerConfig | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.max_structural_edits = max(0, max_structural_edits)
        self.apply_structural_edits = apply_structural_edits
        self.planner_config = planner_config or AdjustmentPlannerConfig()

    def adjust(
        self,
        graph: LayeredGraph,
        impact_by_actant: dict[str, float],
        state: dict[str, float],
        phase_context: dict[str, float] | None = None,
        control_context: dict[str, float] | None = None,
    ) -> GraphAdjustmentResult:
        adjusted = LayeredGraph(
            graph_id=f"{graph.graph_id}:adjusted",
            schema_version=graph.schema_version,
            actants=dict(graph.actants),
            interactions=[],
        )

        viscosity = self._estimate_viscosity(state)
        instability = self._estimate_instability(state)
        phase_risk = self._phase_risk(phase_context)
        slowing = self._phase_slowing(phase_context)
        hysteresis = self._phase_hysteresis(phase_context)
        density, impact_noise = self._graph_profile(graph, impact_by_actant)
        coupling = self._control_coupling_penalty(control_context)
        lr_scale = 1.0 - 0.6 * phase_risk
        planner_horizon = self._planner_horizon(density, impact_noise, phase_risk, coupling, slowing, hysteresis)
        plan_scale, plan_budget = self._select_adjustment_plan(
            graph=graph,
            impact_by_actant=impact_by_actant,
            instability=instability,
            viscosity=viscosity,
            phase_risk=phase_risk,
            slowing=slowing,
            hysteresis=hysteresis,
            density=density,
            impact_noise=impact_noise,
            coupling_penalty=coupling,
            planner_horizon=planner_horizon,
            lr_scale=lr_scale,
        )

        strengthened = 0
        weakened = 0
        unchanged = 0
        shifts: list[float] = []

        for edge in graph.interactions:
            src_impact = float(impact_by_actant.get(edge.source_id, 0.0))
            dst_impact = float(impact_by_actant.get(edge.target_id, 0.0))
            pressure = 0.5 * (src_impact + dst_impact)

            delta = self.learning_rate * lr_scale * plan_scale * (
                (0.22 * pressure) + (0.10 * instability) - (0.08 * viscosity) - (0.10 * phase_risk)
            )
            if pressure < 0.08:
                delta -= self.learning_rate * lr_scale * plan_scale * 0.04

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

        suggested_new = self._suggest_new_edges(
            graph,
            impact_by_actant,
            phase_risk=phase_risk,
            density=density,
        )
        suggested_drop = self._suggest_drop_edges(
            adjusted,
            impact_by_actant,
            phase_risk=phase_risk,
            density=density,
        )
        applied_new, applied_drop = self._apply_structural_edits(
            adjusted=adjusted,
            suggested_new=suggested_new,
            suggested_drop=suggested_drop,
            phase_risk=phase_risk,
            density=density,
            slowing=slowing,
            requested_budget=plan_budget,
        )
        blocked_drop = self._estimate_blocked_drop_count(
            adjusted=adjusted,
            suggested_drop=suggested_drop,
            requested_drop=max(0, min(plan_budget // 2 if plan_budget > 1 else 0, len(suggested_drop))),
        )
        mean_shift = sum(shifts) / len(shifts) if shifts else 0.0
        mean_abs_shift = (sum(abs(v) for v in shifts) / len(shifts)) if shifts else 0.0
        objective = self._adjustment_objective(
            mean_abs_shift=mean_abs_shift,
            instability=instability,
            viscosity=viscosity,
            phase_risk=phase_risk,
            density=density,
            impact_noise=impact_noise,
            coupling_penalty=coupling,
            new_edge_count=applied_new,
            drop_edge_count=applied_drop,
        )
        objective_terms = self._objective_terms(
            mean_abs_shift=mean_abs_shift,
            instability=instability,
            viscosity=viscosity,
            phase_risk=phase_risk,
            density=density,
            impact_noise=impact_noise,
            coupling_penalty=coupling,
            new_edge_count=applied_new,
            drop_edge_count=applied_drop,
        )

        return GraphAdjustmentResult(
            adjusted_graph=adjusted,
            strengthened_edges=strengthened,
            weakened_edges=weakened,
            unchanged_edges=unchanged,
            mean_weight_shift=round(mean_shift, 6),
            adjustment_objective_score=round(objective, 6),
            selected_adjustment_scale=round(plan_scale, 6),
            selected_planner_horizon=planner_horizon,
            selected_edit_budget=plan_budget,
            graph_density=round(density, 6),
            impact_noise=round(impact_noise, 6),
            coupling_penalty=round(coupling, 6),
            objective_terms=objective_terms,
            applied_new_edges=applied_new,
            applied_drop_edges=applied_drop,
            blocked_drop_edges=blocked_drop,
            suggested_new_edges=suggested_new,
            suggested_drop_edges=suggested_drop,
        )

    def _suggest_new_edges(
        self,
        graph: LayeredGraph,
        impact_by_actant: dict[str, float],
        max_edges: int = 3,
        phase_risk: float = 0.0,
        density: float = 0.0,
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
            ia = float(impact_by_actant.get(a, 0.0))
            ib = float(impact_by_actant.get(b, 0.0))
            stability = 1.0 - (abs(ia - ib) / max(1e-6, ia + ib + 1e-6))
            low_risk_score = 0.55 * sim + 0.45 * bridge
            high_risk_score = 0.7 * sim + 0.3 * stability
            score = (1.0 - phase_risk) * low_risk_score + phase_risk * high_risk_score
            scored_pairs.append((score, (a, b)))

        scored_pairs.sort(key=lambda item: item[0], reverse=True)
        density_boost = 1.0 + 0.4 * (1.0 - max(0.0, min(1.0, density)))
        limit = max(
            0,
            int(round(max_edges * (1.0 - 0.75 * max(0.0, min(1.0, phase_risk))) * density_boost)),
        )
        return [pair for _, pair in scored_pairs[:limit]]

    def _suggest_drop_edges(
        self,
        graph: LayeredGraph,
        impact_by_actant: dict[str, float],
        max_edges: int = 3,
        phase_risk: float = 0.0,
        density: float = 0.0,
    ) -> list[tuple[str, str]]:
        scored: list[tuple[float, tuple[str, str]]] = []
        for edge in graph.interactions:
            src_impact = float(impact_by_actant.get(edge.source_id, 0.0))
            dst_impact = float(impact_by_actant.get(edge.target_id, 0.0))
            impact_scale = 0.2 + 0.35 * max(0.0, min(1.0, phase_risk))
            risk_gap = abs(src_impact - dst_impact)
            score = edge.weight + impact_scale * (src_impact + dst_impact) - 0.2 * phase_risk * risk_gap
            scored.append((score, (edge.source_id, edge.target_id)))

        scored.sort(key=lambda item: item[0])
        density_bias = 0.7 + 0.6 * max(0.0, min(1.0, density))
        limit = max(
            0,
            int(round(max_edges * (1.0 - 0.7 * max(0.0, min(1.0, phase_risk))) * density_bias)),
        )
        return [edge for _, edge in scored[:limit]]

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

    def _phase_risk(self, phase_context: dict[str, float] | None) -> float:
        if not phase_context:
            return 0.0
        critical = float(phase_context.get("critical_transition_score", 0.0))
        warning = float(phase_context.get("early_warning_score", 0.0))
        coherence = float(phase_context.get("coherence_break_score", 0.0))
        risk = 0.55 * critical + 0.3 * warning + 0.15 * coherence
        return max(0.0, min(1.0, risk))

    def _adjustment_objective(
        self,
        mean_abs_shift: float,
        instability: float,
        viscosity: float,
        phase_risk: float,
        density: float,
        impact_noise: float,
        coupling_penalty: float,
        new_edge_count: int,
        drop_edge_count: int,
    ) -> float:
        parts = self._objective_terms(
            mean_abs_shift=mean_abs_shift,
            instability=instability,
            viscosity=viscosity,
            phase_risk=phase_risk,
            density=density,
            impact_noise=impact_noise,
            coupling_penalty=coupling_penalty,
            new_edge_count=new_edge_count,
            drop_edge_count=drop_edge_count,
        )
        return parts["total"]

    def _objective_terms(
        self,
        mean_abs_shift: float,
        instability: float,
        viscosity: float,
        phase_risk: float,
        density: float,
        impact_noise: float,
        coupling_penalty: float,
        new_edge_count: int,
        drop_edge_count: int,
    ) -> dict[str, float]:
        # Lower is better: small structural churn with stable dynamics under risk.
        churn = mean_abs_shift
        volatility = max(0.0, instability - viscosity)
        cfg = self.planner_config
        churn_w = cfg.churn_weight_base + cfg.churn_weight_density_gain * max(0.0, min(1.0, density))
        volatility_w = cfg.volatility_weight_base + cfg.volatility_weight_noise_gain * max(0.0, min(1.0, impact_noise))
        rewiring_w = cfg.rewiring_weight_base + cfg.rewiring_weight_density_gain * max(0.0, min(1.0, density))
        risk_w = cfg.risk_weight_base + cfg.risk_weight_gain * max(0.0, min(1.0, phase_risk))
        coupling_w = cfg.coupling_weight_base + cfg.coupling_weight_noise_gain * max(0.0, min(1.0, impact_noise))
        rewiring_cost = rewiring_w * (new_edge_count + drop_edge_count)
        risk_penalty = risk_w * phase_risk
        churn_term = churn_w * churn
        volatility_term = volatility_w * volatility
        coupling_term = coupling_w * coupling_penalty
        total = churn_term + volatility_term + rewiring_cost + risk_penalty + coupling_term
        return {
            "churn": round(churn_term, 6),
            "volatility": round(volatility_term, 6),
            "rewiring": round(rewiring_cost, 6),
            "risk": round(risk_penalty, 6),
            "coupling": round(coupling_term, 6),
            "total": round(total, 6),
        }

    def _select_adjustment_plan(
        self,
        graph: LayeredGraph,
        impact_by_actant: dict[str, float],
        instability: float,
        viscosity: float,
        phase_risk: float,
        slowing: float,
        hysteresis: float,
        density: float,
        impact_noise: float,
        coupling_penalty: float,
        planner_horizon: int,
        lr_scale: float,
    ) -> tuple[float, int]:
        scale_candidates = self._planner_candidates(density, impact_noise, phase_risk)
        budget_candidates = self._planner_budget_candidates(phase_risk, slowing, hysteresis, density)
        best_scale = 1.0
        best_budget = 0
        best_obj = float("inf")
        for scale in scale_candidates:
            for budget in budget_candidates:
                obj = self._plan_objective_rollout(
                    graph=graph,
                    impact_by_actant=impact_by_actant,
                    instability=instability,
                    viscosity=viscosity,
                    phase_risk=phase_risk,
                    slowing=slowing,
                    hysteresis=hysteresis,
                    density=density,
                    impact_noise=impact_noise,
                    coupling_penalty=coupling_penalty,
                    lr_scale=lr_scale,
                    scale=scale,
                    horizon=planner_horizon,
                    edit_budget=budget,
                )
                if obj < best_obj:
                    best_obj = obj
                    best_scale = scale
                    best_budget = budget
        return best_scale, best_budget

    def _plan_objective_rollout(
        self,
        graph: LayeredGraph,
        impact_by_actant: dict[str, float],
        instability: float,
        viscosity: float,
        phase_risk: float,
        slowing: float,
        hysteresis: float,
        density: float,
        impact_noise: float,
        coupling_penalty: float,
        lr_scale: float,
        scale: float,
        horizon: int,
        edit_budget: int,
    ) -> float:
        base_shifts = self._simulate_shifts(
            graph=graph,
            impact_by_actant=impact_by_actant,
            instability=instability,
            viscosity=viscosity,
            phase_risk=phase_risk,
            lr_scale=lr_scale,
            scale=scale,
        )
        if not base_shifts:
            return 0.0
        mean_abs_shift = sum(abs(v) for v in base_shifts) / len(base_shifts)
        desired_shift = 0.06 + 0.1 * max(0.0, instability - 0.5 * viscosity) * (1.0 - 0.6 * phase_risk)
        desired_shift *= max(0.5, 1.0 - 0.35 * slowing - 0.25 * hysteresis)
        under_adjust = max(0.0, desired_shift - mean_abs_shift)
        over_adjust = max(0.0, mean_abs_shift - (0.24 - 0.12 * phase_risk))

        total = 0.0
        discount = 1.0
        inst = instability
        risk = phase_risk
        for _ in range(max(2, horizon)):
            volatility = max(0.0, inst - viscosity)
            dynamic_budget = max(0, int(round(edit_budget * discount)))
            est_new = dynamic_budget // 2
            est_drop = dynamic_budget - est_new
            step_obj = self._adjustment_objective(
                mean_abs_shift=mean_abs_shift,
                instability=inst,
                viscosity=viscosity,
                phase_risk=risk,
                density=density,
                impact_noise=impact_noise,
                coupling_penalty=coupling_penalty,
                new_edge_count=est_new,
                drop_edge_count=est_drop,
            )
            step_obj += self.planner_config.under_adjust_penalty * under_adjust
            step_obj += self.planner_config.over_adjust_penalty * over_adjust
            total += discount * step_obj
            # crude forecast: adjustment reduces instability a bit over horizon
            damping = max(0.05, 1.0 - viscosity)
            response = mean_abs_shift * damping * (1.0 - 0.45 * coupling_penalty)
            rebound = 0.08 * risk * impact_noise + 0.04 * slowing
            inst = max(0.0, inst - 0.28 * response - 0.06 * volatility + rebound)
            risk = max(0.0, risk - 0.10 * response + 0.05 * hysteresis)
            discount *= self.planner_config.rollout_discount
        return total

    def _simulate_shifts(
        self,
        graph: LayeredGraph,
        impact_by_actant: dict[str, float],
        instability: float,
        viscosity: float,
        phase_risk: float,
        lr_scale: float,
        scale: float,
    ) -> list[float]:
        shifts: list[float] = []
        for edge in graph.interactions:
            src_impact = float(impact_by_actant.get(edge.source_id, 0.0))
            dst_impact = float(impact_by_actant.get(edge.target_id, 0.0))
            pressure = 0.5 * (src_impact + dst_impact)
            delta = self.learning_rate * lr_scale * scale * (
                (0.22 * pressure) + (0.10 * instability) - (0.08 * viscosity) - (0.10 * phase_risk)
            )
            if pressure < 0.08:
                delta -= self.learning_rate * lr_scale * scale * 0.04
            new_weight = self._clip(edge.weight + delta)
            shifts.append(new_weight - edge.weight)
        return shifts

    def _graph_profile(
        self,
        graph: LayeredGraph,
        impact_by_actant: dict[str, float],
    ) -> tuple[float, float]:
        n = max(0, len(graph.actants))
        m = max(0, len(graph.interactions))
        possible = max(1, n * (n - 1) / 2)
        density = max(0.0, min(1.0, m / possible))
        vals = [float(v) for v in impact_by_actant.values()]
        if not vals:
            return density, 0.0
        mean_v = sum(vals) / len(vals)
        if mean_v <= 1e-9:
            return density, 0.0
        var = sum((v - mean_v) ** 2 for v in vals) / len(vals)
        std = var ** 0.5
        noise = max(0.0, min(1.0, std / (abs(mean_v) + 1e-9)))
        return density, noise

    def _planner_candidates(self, density: float, impact_noise: float, phase_risk: float) -> tuple[float, ...]:
        cfg = self.planner_config
        if phase_risk > cfg.high_risk_threshold:
            return cfg.high_risk_scale_candidates
        if density < cfg.sparse_threshold and impact_noise < 0.55:
            return cfg.sparse_scale_candidates
        if density > cfg.dense_threshold or impact_noise > cfg.noisy_threshold:
            return cfg.dense_or_noisy_scale_candidates
        return cfg.default_scale_candidates

    def _planner_horizon(
        self,
        density: float,
        impact_noise: float,
        phase_risk: float,
        coupling_penalty: float,
        slowing: float,
        hysteresis: float,
    ) -> int:
        cfg = self.planner_config
        if phase_risk > cfg.high_risk_threshold or coupling_penalty > 0.8 or slowing > 0.7:
            return 5
        if density < 0.2 and impact_noise < 0.45:
            return 4
        if impact_noise > cfg.very_noisy_threshold or hysteresis > 0.65:
            return 5
        if density > 0.7:
            return 3
        return 4

    def _planner_budget_candidates(self, phase_risk: float, slowing: float, hysteresis: float, density: float) -> tuple[int, ...]:
        if self.max_structural_edits <= 0:
            return (0,)
        if phase_risk > 0.8 or slowing > 0.75 or hysteresis > 0.7:
            return (0, 1)
        upper = max(1, self.max_structural_edits)
        if density < 0.2:
            return tuple(sorted(set([0, 1, min(upper, 2), upper])))
        return tuple(sorted(set([0, 1, min(upper, 2)])))

    def _control_coupling_penalty(self, control_context: dict[str, float] | None) -> float:
        if not control_context:
            return 0.0
        residual = max(0.0, float(control_context.get("residual_ratio", 0.0)))
        div_after = max(0.0, float(control_context.get("divergence_norm_after", 0.0)))
        energy = max(0.0, float(control_context.get("control_energy", 0.0)))
        div_term = min(1.0, div_after / 2.0)
        energy_term = min(1.0, energy / 3.0)
        return max(0.0, min(1.0, 0.45 * residual + 0.3 * div_term + 0.25 * energy_term))

    def _apply_structural_edits(
        self,
        adjusted: LayeredGraph,
        suggested_new: list[tuple[str, str]],
        suggested_drop: list[tuple[str, str]],
        phase_risk: float,
        density: float,
        slowing: float,
        requested_budget: int,
    ) -> tuple[int, int]:
        if not self.apply_structural_edits or self.max_structural_edits <= 0:
            return 0, 0
        budget_scale = (
            (1.0 - 0.7 * max(0.0, min(1.0, phase_risk)))
            * (0.8 + 0.4 * max(0.0, min(1.0, density)))
            * (1.0 - 0.5 * max(0.0, min(1.0, slowing)))
        )
        allowed = max(0, int(round(self.max_structural_edits * budget_scale)))
        budget = max(0, min(int(requested_budget), allowed))
        if budget <= 0:
            return 0, 0

        drop_budget = min(len(suggested_drop), budget // 2 if budget > 1 else 0)
        new_budget = min(len(suggested_new), budget - drop_budget)

        dropped = 0
        for pair in suggested_drop[:drop_budget]:
            key = tuple(sorted(pair))
            idx = self._find_interaction_index(adjusted.interactions, key)
            if idx is not None:
                if self._can_drop_edge(adjusted, idx):
                    adjusted.interactions.pop(idx)
                    dropped += 1

        existing = {tuple(sorted((e.source_id, e.target_id))) for e in adjusted.interactions}
        added = 0
        ts = self._base_timestamp(adjusted.interactions)
        for i, (a, b) in enumerate(suggested_new[:new_budget]):
            key = tuple(sorted((a, b)))
            if a == b or key in existing:
                continue
            adjusted.interactions.append(
                Interaction(
                    interaction_id=f"auto_add_{len(adjusted.interactions)}_{i}",
                    timestamp=ts + timedelta(seconds=i + 1),
                    source_id=a,
                    target_id=b,
                    layer="adaptive",
                    weight=round(self.min_weight * 1.4, 6),
                )
            )
            existing.add(key)
            added += 1
        return added, dropped

    def _estimate_blocked_drop_count(
        self,
        adjusted: LayeredGraph,
        suggested_drop: list[tuple[str, str]],
        requested_drop: int,
    ) -> int:
        if requested_drop <= 0:
            return 0
        blocked = 0
        temp_graph = LayeredGraph(
            graph_id=adjusted.graph_id,
            schema_version=adjusted.schema_version,
            actants=dict(adjusted.actants),
            interactions=list(adjusted.interactions),
        )
        for pair in suggested_drop[:requested_drop]:
            key = tuple(sorted(pair))
            idx = self._find_interaction_index(temp_graph.interactions, key)
            if idx is None:
                continue
            if self._can_drop_edge(temp_graph, idx):
                temp_graph.interactions.pop(idx)
            else:
                blocked += 1
        return blocked

    def _find_interaction_index(self, interactions: list[Interaction], pair_key: tuple[str, str]) -> int | None:
        best_idx: int | None = None
        best_weight = float("inf")
        for i, edge in enumerate(interactions):
            key = tuple(sorted((edge.source_id, edge.target_id)))
            if key != pair_key:
                continue
            if edge.weight < best_weight:
                best_weight = edge.weight
                best_idx = i
        return best_idx

    def _base_timestamp(self, interactions: list[Interaction]) -> datetime:
        if not interactions:
            return datetime.utcnow()
        return max(edge.timestamp for edge in interactions)

    def _can_drop_edge(self, graph: LayeredGraph, edge_idx: int) -> bool:
        interactions = graph.interactions
        if edge_idx < 0 or edge_idx >= len(interactions):
            return False
        edge = interactions[edge_idx]
        if edge.source_id == edge.target_id:
            return True

        # Keep at least one incident edge per endpoint when possible.
        deg = self._degree_map(interactions)
        if deg.get(edge.source_id, 0) <= 1 or deg.get(edge.target_id, 0) <= 1:
            return False

        before_cc = self._component_count(graph.actants.keys(), interactions)
        after = interactions[:edge_idx] + interactions[edge_idx + 1 :]
        after_cc = self._component_count(graph.actants.keys(), after)
        return after_cc <= before_cc

    def _degree_map(self, interactions: list[Interaction]) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in interactions:
            if e.source_id == e.target_id:
                out[e.source_id] = out.get(e.source_id, 0) + 1
                continue
            out[e.source_id] = out.get(e.source_id, 0) + 1
            out[e.target_id] = out.get(e.target_id, 0) + 1
        return out

    def _component_count(self, nodes: Iterable[str], interactions: list[Interaction]) -> int:
        adjacency: dict[str, set[str]] = {n: set() for n in nodes}
        for e in interactions:
            if e.source_id == e.target_id:
                continue
            adjacency.setdefault(e.source_id, set()).add(e.target_id)
            adjacency.setdefault(e.target_id, set()).add(e.source_id)

        visited: set[str] = set()
        comps = 0
        for n in adjacency.keys():
            if n in visited:
                continue
            comps += 1
            stack = [n]
            visited.add(n)
            while stack:
                cur = stack.pop()
                for nxt in adjacency.get(cur, set()):
                    if nxt in visited:
                        continue
                    visited.add(nxt)
                    stack.append(nxt)
        return comps

    def _phase_slowing(self, phase_context: dict[str, float] | None) -> float:
        if not phase_context:
            return 0.0
        return max(0.0, min(1.0, float(phase_context.get("critical_slowing_score", 0.0))))

    def _phase_hysteresis(self, phase_context: dict[str, float] | None) -> float:
        if not phase_context:
            return 0.0
        return max(0.0, min(1.0, float(phase_context.get("hysteresis_proxy_score", 0.0))))

    def _clip(self, value: float) -> float:
        return max(self.min_weight, min(self.max_weight, value))
