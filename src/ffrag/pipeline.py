from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .compose import compose_answer
from .flow import (
    ClusterFlowController,
    DynamicGraphAdjuster,
    FlowAnalyzerConfig,
    FlowDynamicsAnalyzer,
    FlowFieldDynamics,
    PhaseTransitionAnalyzer,
    FlowSimulator,
    StateVectorBuilder,
    TopologicalFlowController,
)
from .models import Answer, LayeredGraph, Perturbation, Query
from .retrieval import GraphRetriever
from .router import QueryRouter


@dataclass(slots=True)
class _CoreCycleSummary:
    final_graph: LayeredGraph
    final_impact: dict[str, float]
    trajectory: list[dict[str, float]]
    distance_series: list[float]
    total_strengthened: int
    total_weakened: int
    mean_weight_shift: float
    mean_adjustment_objective: float
    mean_adjustment_scale: float
    mean_planner_horizon: float
    mean_graph_density: float
    mean_impact_noise: float
    mean_coupling_penalty: float
    mean_applied_new_edges: float
    mean_applied_drop_edges: float
    mean_suggested_new_edges: float
    mean_suggested_drop_edges: float
    mean_control_energy: float
    mean_residual_ratio: float
    mean_divergence_reduction: float
    mean_saturation_ratio: float
    mean_cycle_pressure: float
    oscillation_index: float
    converged: bool
    cycles_executed: int
    mean_objective_score: float
    objective_improvement: float
    mean_curl_ratio: float
    mean_harmonic_ratio: float
    mean_higher_order_pressure: float
    mean_simplex_density: float
    mean_topological_tension: float
    critical_transition_score: float
    early_warning_score: float
    regime_switch_count: int
    regime_persistence_score: float
    coherence_break_score: float
    critical_slowing_score: float
    hysteresis_proxy_score: float
    dominant_regime: str
    mean_cluster_objective: float
    mean_cross_scale_consistency: float
    mean_micro_refinement_gain: float


class FlowGraphRAG:
    """PoC orchestration for describe/predict/intervene queries."""

    def __init__(self, analyzer_config: FlowAnalyzerConfig | None = None) -> None:
        self.router = QueryRouter()
        self.retriever = GraphRetriever()
        self.state_builder = StateVectorBuilder()
        self.simulator = FlowSimulator()
        self.dynamics = FlowFieldDynamics()
        self.analyzer = FlowDynamicsAnalyzer(config=analyzer_config)
        self.phase_analyzer = PhaseTransitionAnalyzer()
        self.adjuster = DynamicGraphAdjuster()
        self.controller = TopologicalFlowController()
        self.cluster_controller = ClusterFlowController()

    def run(self, graph: LayeredGraph, query: Query, perturbation: Perturbation | None = None) -> Answer:
        query_type = self.router.classify(query)
        if query_type == QueryRouter.DESCRIBE:
            return self._describe(graph, query)
        if query_type == QueryRouter.PREDICT:
            return self._predict(graph, query, perturbation)
        if query_type == QueryRouter.INTERVENE:
            return self._intervene(graph, query, perturbation)
        raise ValueError(f"Unknown query type: {query_type}")

    def _describe(self, graph: LayeredGraph, query: Query) -> Answer:
        local = self.retriever.retrieve_local(graph, query.text)
        if not local:
            local = self.retriever.retrieve_global(graph)

        evidence_ids = [edge.interaction_id for edge in local]
        layer_count = len({edge.layer for edge in local})
        claims = [
            f"Retrieved {len(local)} evidence edges across {layer_count} layers.",
            "Structure summary is based on graph connectivity, not subjective intent.",
        ]
        metrics = {
            "evidence_edge_count": float(len(local)),
            "layer_coverage": float(layer_count),
        }
        return compose_answer("describe", claims, evidence_ids, metrics, uncertainty=0.2)

    def _predict(self, graph: LayeredGraph, query: Query, perturbation: Perturbation | None) -> Answer:
        p = perturbation or self._default_perturbation(graph)
        cycle = self._run_core_cycle(graph, p, cycles=3, shock_kind="base")
        top_impacts = sorted(cycle.final_impact.items(), key=lambda item: item[1], reverse=True)[:3]
        final_distance = cycle.distance_series[-1] if cycle.distance_series else 0.0
        shock_vec = self._shock_vector(p)
        trans = self.analyzer.transition_analysis(cycle.trajectory, shock=shock_vec)
        resil = self.analyzer.resilience_analysis(cycle.distance_series)
        dominant_transition = self._dominant_transition_prob(trans.transition_matrix)

        claims = [
            f"Core cycle reached {len(cycle.final_impact)} actants after iterative co-evolution.",
            f"Top impacted actants: {', '.join(actant for actant, _ in top_impacts) if top_impacts else 'none'}.",
            f"Co-evolution produced {len(cycle.trajectory)} state updates; final attractor distance={final_distance:.4f}.",
            f"State trajectory classes: {', '.join(trans.states) if trans.states else 'none'}.",
            f"Attractor basin state: {trans.attractor_basin_state or 'none'}; triggers={len(trans.transition_triggers)}.",
            f"Trigger confidence avg={trans.avg_trigger_confidence:.3f}; basin occupancy={trans.basin_occupancy:.3f}.",
            f"Graph adjusted cumulatively: +{cycle.total_strengthened}/-{cycle.total_weakened} edges, mean shift={cycle.mean_weight_shift:.4f}.",
            f"Adjustment objective={cycle.mean_adjustment_objective:.4f} (lower is better).",
            f"Adjustment scale={cycle.mean_adjustment_scale:.3f}.",
            f"Planner horizon={cycle.mean_planner_horizon:.2f}.",
            f"Graph density/noise={cycle.mean_graph_density:.3f}/{cycle.mean_impact_noise:.3f}.",
            f"Control-adjust coupling penalty={cycle.mean_coupling_penalty:.3f}.",
            f"Applied structural edits new/drop={cycle.mean_applied_new_edges:.2f}/{cycle.mean_applied_drop_edges:.2f}.",
            f"Topological control energy={cycle.mean_control_energy:.4f}, residual ratio={cycle.mean_residual_ratio:.4f}.",
            f"Phase regime={cycle.dominant_regime}; critical score={cycle.critical_transition_score:.3f}.",
        ]
        metrics = {
            "hops_executed": 0.0,
            "affected_actants": float(len(cycle.final_impact)),
            "stabilized": 1.0 if len(cycle.distance_series) >= 2 and abs(cycle.distance_series[-1] - cycle.distance_series[-2]) < 0.02 else 0.0,
            "dynamics_steps": float(len(cycle.trajectory)),
            "final_attractor_distance": float(final_distance),
            "dynamics_stabilized": 1.0 if len(cycle.distance_series) >= 2 and abs(cycle.distance_series[-1] - cycle.distance_series[-2]) < 0.02 else 0.0,
            "state_class_count": float(len(set(trans.states))),
            "dominant_transition_prob": float(dominant_transition),
            "recovery_rate": float(resil.recovery_rate),
            "overshoot_index": float(resil.overshoot_index),
            "settling_time": float(resil.settling_time),
            "path_efficiency": float(resil.path_efficiency),
            "transition_trigger_count": float(len(trans.transition_triggers)),
            "avg_trigger_confidence": float(trans.avg_trigger_confidence),
            "attractor_basin_radius": float(trans.attractor_basin_radius),
            "basin_occupancy": float(trans.basin_occupancy),
            "causal_alignment_score": float(trans.causal_alignment_score),
            "strengthened_edges": float(cycle.total_strengthened),
            "weakened_edges": float(cycle.total_weakened),
            "mean_weight_shift": float(cycle.mean_weight_shift),
            "adjustment_objective_score": float(cycle.mean_adjustment_objective),
            "adjustment_scale": float(cycle.mean_adjustment_scale),
            "planner_horizon": float(cycle.mean_planner_horizon),
            "graph_density": float(cycle.mean_graph_density),
            "impact_noise": float(cycle.mean_impact_noise),
            "coupling_penalty": float(cycle.mean_coupling_penalty),
            "applied_new_edges": float(cycle.mean_applied_new_edges),
            "applied_drop_edges": float(cycle.mean_applied_drop_edges),
            "suggested_new_edges": float(cycle.mean_suggested_new_edges),
            "suggested_drop_edges": float(cycle.mean_suggested_drop_edges),
            "control_energy": float(cycle.mean_control_energy),
            "residual_ratio": float(cycle.mean_residual_ratio),
            "divergence_reduction": float(cycle.mean_divergence_reduction),
            "saturation_ratio": float(cycle.mean_saturation_ratio),
            "cycle_pressure": float(cycle.mean_cycle_pressure),
            "oscillation_index": float(cycle.oscillation_index),
            "converged": 1.0 if cycle.converged else 0.0,
            "cycles_executed": float(cycle.cycles_executed),
            "objective_score": float(cycle.mean_objective_score),
            "objective_improvement": float(cycle.objective_improvement),
            "curl_ratio": float(cycle.mean_curl_ratio),
            "harmonic_ratio": float(cycle.mean_harmonic_ratio),
            "higher_order_pressure": float(cycle.mean_higher_order_pressure),
            "simplex_density": float(cycle.mean_simplex_density),
            "topological_tension": float(cycle.mean_topological_tension),
            "cluster_objective": float(cycle.mean_cluster_objective),
            "cross_scale_consistency": float(cycle.mean_cross_scale_consistency),
            "micro_refinement_gain": float(cycle.mean_micro_refinement_gain),
            "critical_transition_score": float(cycle.critical_transition_score),
            "early_warning_score": float(cycle.early_warning_score),
            "regime_switch_count": float(cycle.regime_switch_count),
            "regime_persistence_score": float(cycle.regime_persistence_score),
            "coherence_break_score": float(cycle.coherence_break_score),
            "critical_slowing_score": float(cycle.critical_slowing_score),
            "hysteresis_proxy_score": float(cycle.hysteresis_proxy_score),
        }
        evidence_ids = [f"perturbation:{p.perturbation_id}"]
        return compose_answer("predict", claims, evidence_ids, metrics, uncertainty=0.35)

    def _intervene(self, graph: LayeredGraph, query: Query, perturbation: Perturbation | None) -> Answer:
        p = perturbation or self._default_perturbation(graph)
        baseline_cycle = self._run_core_cycle(graph, p, cycles=3, shock_kind="base")
        intervention_cycle = self._run_core_cycle(graph, p, cycles=3, shock_kind="counter")
        rewires = self.simulator.propagate(graph, p).rewired_edges[:3]
        rewire_text = ", ".join(f"{src}->{dst}" for src, dst in rewires) if rewires else "no rewiring suggested"
        base_dist = baseline_cycle.distance_series[-1] if baseline_cycle.distance_series else 0.0
        intervention_dist = intervention_cycle.distance_series[-1] if intervention_cycle.distance_series else 0.0
        improvement = base_dist - intervention_dist
        shock_vec = self._shock_vector(p)
        counter_shock_vec = self._counter_shock_vector(p)
        base_trans = self.analyzer.transition_analysis(baseline_cycle.trajectory, shock=shock_vec)
        int_trans = self.analyzer.transition_analysis(intervention_cycle.trajectory, shock=counter_shock_vec)
        int_resil = self.analyzer.resilience_analysis(
            intervention_cycle.distance_series,
            baseline_distances=baseline_cycle.distance_series,
        )
        structural_gain = baseline_cycle.mean_weight_shift - intervention_cycle.mean_weight_shift

        claims = [
            f"Suggested rewiring candidates: {rewire_text}.",
            "Intervention proposal is based on structural impact propagation.",
            f"Counter-shock simulation changed attractor distance by {improvement:.4f} (positive is better).",
            f"Intervention hysteresis index={int_resil.hysteresis_index:.4f}.",
            f"Intervention basin state: {int_trans.attractor_basin_state or 'none'}; triggers={len(int_trans.transition_triggers)}.",
            f"Intervention trigger confidence avg={int_trans.avg_trigger_confidence:.3f}; basin occupancy={int_trans.basin_occupancy:.3f}.",
            f"Structural adjustment gain={structural_gain:.4f} (lower post-intervention edge shift is better).",
            f"Control energy baseline/intervention={baseline_cycle.mean_control_energy:.4f}/{intervention_cycle.mean_control_energy:.4f}.",
            f"Critical transition score baseline/intervention={baseline_cycle.critical_transition_score:.3f}/{intervention_cycle.critical_transition_score:.3f}.",
        ]
        metrics = {
            "rewire_candidates": float(len(rewires)),
            "affected_actants": float(len(baseline_cycle.final_impact)),
            "baseline_attractor_distance": float(base_dist),
            "intervention_attractor_distance": float(intervention_dist),
            "intervention_improvement": float(improvement),
            "baseline_state_class_count": float(len(set(base_trans.states))),
            "intervention_state_class_count": float(len(set(int_trans.states))),
            "intervention_recovery_rate": float(int_resil.recovery_rate),
            "intervention_hysteresis_index": float(int_resil.hysteresis_index),
            "intervention_overshoot_index": float(int_resil.overshoot_index),
            "intervention_settling_time": float(int_resil.settling_time),
            "intervention_path_efficiency": float(int_resil.path_efficiency),
            "intervention_trigger_count": float(len(int_trans.transition_triggers)),
            "intervention_avg_trigger_confidence": float(int_trans.avg_trigger_confidence),
            "intervention_basin_radius": float(int_trans.attractor_basin_radius),
            "intervention_basin_occupancy": float(int_trans.basin_occupancy),
            "intervention_causal_alignment_score": float(int_trans.causal_alignment_score),
            "intervention_structural_gain": float(structural_gain),
            "baseline_mean_weight_shift": float(baseline_cycle.mean_weight_shift),
            "intervention_mean_weight_shift": float(intervention_cycle.mean_weight_shift),
            "baseline_adjustment_objective_score": float(baseline_cycle.mean_adjustment_objective),
            "intervention_adjustment_objective_score": float(intervention_cycle.mean_adjustment_objective),
            "baseline_adjustment_scale": float(baseline_cycle.mean_adjustment_scale),
            "intervention_adjustment_scale": float(intervention_cycle.mean_adjustment_scale),
            "baseline_planner_horizon": float(baseline_cycle.mean_planner_horizon),
            "intervention_planner_horizon": float(intervention_cycle.mean_planner_horizon),
            "baseline_graph_density": float(baseline_cycle.mean_graph_density),
            "intervention_graph_density": float(intervention_cycle.mean_graph_density),
            "baseline_impact_noise": float(baseline_cycle.mean_impact_noise),
            "intervention_impact_noise": float(intervention_cycle.mean_impact_noise),
            "baseline_coupling_penalty": float(baseline_cycle.mean_coupling_penalty),
            "intervention_coupling_penalty": float(intervention_cycle.mean_coupling_penalty),
            "baseline_applied_new_edges": float(baseline_cycle.mean_applied_new_edges),
            "intervention_applied_new_edges": float(intervention_cycle.mean_applied_new_edges),
            "baseline_applied_drop_edges": float(baseline_cycle.mean_applied_drop_edges),
            "intervention_applied_drop_edges": float(intervention_cycle.mean_applied_drop_edges),
            "baseline_strengthened_edges": float(baseline_cycle.total_strengthened),
            "intervention_strengthened_edges": float(intervention_cycle.total_strengthened),
            "baseline_control_energy": float(baseline_cycle.mean_control_energy),
            "intervention_control_energy": float(intervention_cycle.mean_control_energy),
            "baseline_residual_ratio": float(baseline_cycle.mean_residual_ratio),
            "intervention_residual_ratio": float(intervention_cycle.mean_residual_ratio),
            "baseline_oscillation_index": float(baseline_cycle.oscillation_index),
            "intervention_oscillation_index": float(intervention_cycle.oscillation_index),
            "baseline_converged": 1.0 if baseline_cycle.converged else 0.0,
            "intervention_converged": 1.0 if intervention_cycle.converged else 0.0,
            "baseline_objective_score": float(baseline_cycle.mean_objective_score),
            "intervention_objective_score": float(intervention_cycle.mean_objective_score),
            "baseline_curl_ratio": float(baseline_cycle.mean_curl_ratio),
            "intervention_curl_ratio": float(intervention_cycle.mean_curl_ratio),
            "baseline_harmonic_ratio": float(baseline_cycle.mean_harmonic_ratio),
            "intervention_harmonic_ratio": float(intervention_cycle.mean_harmonic_ratio),
            "baseline_higher_order_pressure": float(baseline_cycle.mean_higher_order_pressure),
            "intervention_higher_order_pressure": float(intervention_cycle.mean_higher_order_pressure),
            "baseline_simplex_density": float(baseline_cycle.mean_simplex_density),
            "intervention_simplex_density": float(intervention_cycle.mean_simplex_density),
            "baseline_topological_tension": float(baseline_cycle.mean_topological_tension),
            "intervention_topological_tension": float(intervention_cycle.mean_topological_tension),
            "baseline_cluster_objective": float(baseline_cycle.mean_cluster_objective),
            "intervention_cluster_objective": float(intervention_cycle.mean_cluster_objective),
            "baseline_cross_scale_consistency": float(baseline_cycle.mean_cross_scale_consistency),
            "intervention_cross_scale_consistency": float(intervention_cycle.mean_cross_scale_consistency),
            "baseline_micro_refinement_gain": float(baseline_cycle.mean_micro_refinement_gain),
            "intervention_micro_refinement_gain": float(intervention_cycle.mean_micro_refinement_gain),
            "baseline_critical_transition_score": float(baseline_cycle.critical_transition_score),
            "intervention_critical_transition_score": float(intervention_cycle.critical_transition_score),
            "baseline_early_warning_score": float(baseline_cycle.early_warning_score),
            "intervention_early_warning_score": float(intervention_cycle.early_warning_score),
            "baseline_regime_switch_count": float(baseline_cycle.regime_switch_count),
            "intervention_regime_switch_count": float(intervention_cycle.regime_switch_count),
            "baseline_regime_persistence_score": float(baseline_cycle.regime_persistence_score),
            "intervention_regime_persistence_score": float(intervention_cycle.regime_persistence_score),
            "baseline_coherence_break_score": float(baseline_cycle.coherence_break_score),
            "intervention_coherence_break_score": float(intervention_cycle.coherence_break_score),
            "baseline_critical_slowing_score": float(baseline_cycle.critical_slowing_score),
            "intervention_critical_slowing_score": float(intervention_cycle.critical_slowing_score),
            "baseline_hysteresis_proxy_score": float(baseline_cycle.hysteresis_proxy_score),
            "intervention_hysteresis_proxy_score": float(intervention_cycle.hysteresis_proxy_score),
        }
        evidence_ids = [f"perturbation:{p.perturbation_id}"]
        return compose_answer("intervene", claims, evidence_ids, metrics, uncertainty=0.4)

    def build_state_vector(self, graph: LayeredGraph, entity_id: str) -> dict[str, float]:
        return self.state_builder.build(graph, entity_id, timestamp=datetime.now(timezone.utc)).values

    def _default_perturbation(self, graph: LayeredGraph) -> Perturbation:
        target = next(iter(graph.actants), "")
        return Perturbation(
            perturbation_id="auto-default",
            timestamp=datetime.now(timezone.utc),
            targets=[target] if target else [],
            intensity=1.0,
            kind="default",
        )

    def _history_states(
        self,
        graph: LayeredGraph,
        target: str,
        impact_by_actant: dict[str, float],
    ) -> list[dict[str, float]]:
        if not target:
            return []
        ranked = sorted(impact_by_actant.items(), key=lambda item: item[1], reverse=True)
        candidates = [target] + [node for node, _ in ranked if node != target][:2]
        history: list[dict[str, float]] = []
        for node in candidates:
            if node in graph.actants:
                history.append(self.state_builder.build(graph, node).values)
        return history

    def _shock_vector(self, perturbation: Perturbation) -> dict[str, float]:
        magnitude = max(0.0, min(2.0, perturbation.intensity))
        return {
            "social_entropy": 0.1 * magnitude,
            "temporal_regularity": -0.12 * magnitude,
            "spatial_range": 0.05 * magnitude,
            "schedule_density": 0.18 * magnitude,
            "network_centrality": 0.08 * magnitude,
            "transition_speed": 0.22 * magnitude,
        }

    def _counter_shock_vector(self, perturbation: Perturbation) -> dict[str, float]:
        base = self._shock_vector(perturbation)
        return {key: -0.7 * value for key, value in base.items()}

    def _dominant_transition_prob(self, matrix: dict[str, dict[str, float]]) -> float:
        best = 0.0
        for row in matrix.values():
            for prob in row.values():
                if prob > best:
                    best = prob
        return round(best, 6)

    def _run_core_cycle(
        self,
        graph: LayeredGraph,
        perturbation: Perturbation,
        cycles: int,
        shock_kind: str,
    ) -> _CoreCycleSummary:
        current_graph = graph
        trajectory: list[dict[str, float]] = []
        distance_series: list[float] = []
        total_strengthened = 0
        total_weakened = 0
        mean_shifts: list[float] = []
        adjustment_objectives: list[float] = []
        adjustment_scales: list[float] = []
        planner_horizons: list[int] = []
        graph_densities: list[float] = []
        impact_noises: list[float] = []
        coupling_penalties: list[float] = []
        applied_new_counts: list[int] = []
        applied_drop_counts: list[int] = []
        new_edge_counts: list[int] = []
        drop_edge_counts: list[int] = []
        control_energies: list[float] = []
        residual_ratios: list[float] = []
        divergence_reductions: list[float] = []
        saturation_ratios: list[float] = []
        cycle_pressures: list[float] = []
        objective_scores: list[float] = []
        curl_ratios: list[float] = []
        harmonic_ratios: list[float] = []
        higher_pressures: list[float] = []
        simplex_densities: list[float] = []
        topological_tensions: list[float] = []
        cluster_objectives: list[float] = []
        cross_scale_consistency: list[float] = []
        micro_refinement_gain: list[float] = []
        converged = False
        stable_streak = 0
        last_impact: dict[str, float] = {}
        prev_phase_context: dict[str, float] = {}

        for _ in range(max(1, cycles)):
            propagation = self.simulator.propagate(current_graph, perturbation)
            last_impact = propagation.impact_by_actant
            target = perturbation.targets[0] if perturbation.targets else next(iter(current_graph.actants), "")
            initial_state = self.state_builder.build(current_graph, target).values if target else {}
            history = self._history_states(current_graph, target, propagation.impact_by_actant)
            shock_vec = self._shock_vector(perturbation) if shock_kind == "base" else self._counter_shock_vector(perturbation)
            dyn = self.dynamics.simulate(initial_state=initial_state, history=history, shock=shock_vec)
            final_state = dyn.snapshots[-1].state if dyn.snapshots else initial_state
            final_distance = dyn.snapshots[-1].attractor_distance if dyn.snapshots else 0.0
            trajectory.append(final_state)
            distance_series.append(final_distance)

            phase_signal = float(prev_phase_context.get("critical_transition_score", 0.0))
            cluster_plan = self.cluster_controller.plan(
                current_graph,
                propagation.impact_by_actant,
                final_state,
                phase_signal=phase_signal,
            )
            control = self.controller.compute(
                current_graph,
                cluster_plan.coarse_controlled_impact,
                final_state,
                phase_signal=phase_signal,
            )
            phase_context = self.phase_analyzer.analyze(
                trajectory=trajectory,
                attractor_distances=distance_series,
                objective_scores=objective_scores + [control.objective_score],
                topological_tensions=topological_tensions + [control.topological_tension],
            )
            phase_context_map = {
                "critical_transition_score": phase_context.critical_transition_score,
                "early_warning_score": phase_context.early_warning_score,
                "coherence_break_score": phase_context.coherence_break_score,
            }
            adjustment = self.adjuster.adjust(
                current_graph,
                control.controlled_impact,
                final_state,
                phase_context=phase_context_map,
                control_context={
                    "residual_ratio": control.residual_ratio,
                    "divergence_norm_after": control.divergence_norm_after,
                    "control_energy": control.control_energy,
                },
            )
            prev_phase_context = phase_context_map
            total_strengthened += adjustment.strengthened_edges
            total_weakened += adjustment.weakened_edges
            mean_shifts.append(adjustment.mean_weight_shift)
            adjustment_objectives.append(adjustment.adjustment_objective_score)
            adjustment_scales.append(adjustment.selected_adjustment_scale)
            planner_horizons.append(adjustment.selected_planner_horizon)
            graph_densities.append(adjustment.graph_density)
            impact_noises.append(adjustment.impact_noise)
            coupling_penalties.append(adjustment.coupling_penalty)
            applied_new_counts.append(adjustment.applied_new_edges)
            applied_drop_counts.append(adjustment.applied_drop_edges)
            new_edge_counts.append(len(adjustment.suggested_new_edges))
            drop_edge_counts.append(len(adjustment.suggested_drop_edges))
            control_energies.append(control.control_energy)
            residual_ratios.append(control.residual_ratio)
            divergence_reductions.append(max(0.0, control.divergence_norm_before - control.divergence_norm_after))
            saturation_ratios.append(control.saturation_ratio)
            cycle_pressures.append(control.cycle_pressure_mean)
            objective_scores.append(control.objective_score)
            curl_ratios.append(control.curl_ratio)
            harmonic_ratios.append(control.harmonic_ratio)
            higher_pressures.append(control.higher_order_pressure_mean)
            simplex_densities.append(control.simplex_density)
            topological_tensions.append(control.topological_tension)
            cluster_objectives.append(cluster_plan.cluster_objective)
            cross_scale_consistency.append(cluster_plan.cross_scale_consistency)
            micro_refinement_gain.append(max(0.0, cluster_plan.cluster_objective - control.objective_score))
            current_graph = adjustment.adjusted_graph

            if len(distance_series) >= 2:
                dist_delta = abs(distance_series[-1] - distance_series[-2])
                stable_dist = dist_delta < 0.015
                stable_div = control.divergence_norm_after <= max(0.12, 0.8 * control.divergence_norm_before)
                stable_shift = abs(adjustment.mean_weight_shift) < 0.02
                objective_flat = self._objective_flat(objective_scores, window=2, tol=0.01)
                stable_cross_scale = (
                    bool(cross_scale_consistency) and cross_scale_consistency[-1] >= 0.55
                )
                if stable_dist and stable_div and stable_shift and objective_flat and stable_cross_scale:
                    stable_streak += 1
                else:
                    stable_streak = 0
                if stable_streak >= 2:
                    converged = True
                    break

        mean_shift = sum(mean_shifts) / len(mean_shifts) if mean_shifts else 0.0
        mean_adjustment_objective = (
            sum(adjustment_objectives) / len(adjustment_objectives) if adjustment_objectives else 0.0
        )
        mean_adjustment_scale = (
            sum(adjustment_scales) / len(adjustment_scales) if adjustment_scales else 1.0
        )
        mean_planner_horizon = (
            sum(planner_horizons) / len(planner_horizons) if planner_horizons else 3.0
        )
        mean_graph_density = sum(graph_densities) / len(graph_densities) if graph_densities else 0.0
        mean_impact_noise = sum(impact_noises) / len(impact_noises) if impact_noises else 0.0
        mean_coupling_penalty = sum(coupling_penalties) / len(coupling_penalties) if coupling_penalties else 0.0
        mean_applied_new = sum(applied_new_counts) / len(applied_new_counts) if applied_new_counts else 0.0
        mean_applied_drop = sum(applied_drop_counts) / len(applied_drop_counts) if applied_drop_counts else 0.0
        mean_new = sum(new_edge_counts) / len(new_edge_counts) if new_edge_counts else 0.0
        mean_drop = sum(drop_edge_counts) / len(drop_edge_counts) if drop_edge_counts else 0.0
        mean_energy = sum(control_energies) / len(control_energies) if control_energies else 0.0
        mean_residual = sum(residual_ratios) / len(residual_ratios) if residual_ratios else 0.0
        mean_sat = sum(saturation_ratios) / len(saturation_ratios) if saturation_ratios else 0.0
        mean_cycle_pressure = sum(cycle_pressures) / len(cycle_pressures) if cycle_pressures else 0.0
        mean_objective = sum(objective_scores) / len(objective_scores) if objective_scores else 0.0
        mean_curl_ratio = sum(curl_ratios) / len(curl_ratios) if curl_ratios else 0.0
        mean_harmonic_ratio = sum(harmonic_ratios) / len(harmonic_ratios) if harmonic_ratios else 0.0
        mean_higher_pressure = sum(higher_pressures) / len(higher_pressures) if higher_pressures else 0.0
        mean_simplex_density = sum(simplex_densities) / len(simplex_densities) if simplex_densities else 0.0
        mean_topological_tension = (
            sum(topological_tensions) / len(topological_tensions) if topological_tensions else 0.0
        )
        mean_cluster_obj = sum(cluster_objectives) / len(cluster_objectives) if cluster_objectives else 0.0
        mean_consistency = (
            sum(cross_scale_consistency) / len(cross_scale_consistency) if cross_scale_consistency else 0.0
        )
        mean_refinement_gain = (
            sum(micro_refinement_gain) / len(micro_refinement_gain) if micro_refinement_gain else 0.0
        )
        phase = self.phase_analyzer.analyze(
            trajectory=trajectory,
            attractor_distances=distance_series,
            objective_scores=objective_scores,
            topological_tensions=topological_tensions,
        )
        objective_improvement = (
            (objective_scores[0] - objective_scores[-1]) if len(objective_scores) >= 2 else 0.0
        )
        mean_div_reduction = (
            sum(divergence_reductions) / len(divergence_reductions) if divergence_reductions else 0.0
        )
        oscillation_index = self._oscillation_index(distance_series)
        return _CoreCycleSummary(
            final_graph=current_graph,
            final_impact=last_impact,
            trajectory=trajectory,
            distance_series=distance_series,
            total_strengthened=total_strengthened,
            total_weakened=total_weakened,
            mean_weight_shift=mean_shift,
            mean_adjustment_objective=mean_adjustment_objective,
            mean_adjustment_scale=mean_adjustment_scale,
            mean_planner_horizon=mean_planner_horizon,
            mean_graph_density=mean_graph_density,
            mean_impact_noise=mean_impact_noise,
            mean_coupling_penalty=mean_coupling_penalty,
            mean_applied_new_edges=mean_applied_new,
            mean_applied_drop_edges=mean_applied_drop,
            mean_suggested_new_edges=mean_new,
            mean_suggested_drop_edges=mean_drop,
            mean_control_energy=mean_energy,
            mean_residual_ratio=mean_residual,
            mean_divergence_reduction=mean_div_reduction,
            mean_saturation_ratio=mean_sat,
            mean_cycle_pressure=mean_cycle_pressure,
            oscillation_index=oscillation_index,
            converged=converged,
            cycles_executed=len(distance_series),
            mean_objective_score=mean_objective,
            objective_improvement=objective_improvement,
            mean_curl_ratio=mean_curl_ratio,
            mean_harmonic_ratio=mean_harmonic_ratio,
            mean_higher_order_pressure=mean_higher_pressure,
            mean_simplex_density=mean_simplex_density,
            mean_topological_tension=mean_topological_tension,
            critical_transition_score=phase.critical_transition_score,
            early_warning_score=phase.early_warning_score,
            regime_switch_count=phase.regime_switch_count,
            regime_persistence_score=phase.regime_persistence_score,
            coherence_break_score=phase.coherence_break_score,
            critical_slowing_score=phase.critical_slowing_score,
            hysteresis_proxy_score=phase.hysteresis_proxy_score,
            dominant_regime=phase.dominant_regime,
            mean_cluster_objective=mean_cluster_obj,
            mean_cross_scale_consistency=mean_consistency,
            mean_micro_refinement_gain=mean_refinement_gain,
        )

    def _oscillation_index(self, values: list[float]) -> float:
        if len(values) < 3:
            return 0.0
        sign_changes = 0
        prev_sign = 0
        for i in range(1, len(values)):
            d = values[i] - values[i - 1]
            sign = 1 if d > 0 else (-1 if d < 0 else 0)
            if sign != 0 and prev_sign != 0 and sign != prev_sign:
                sign_changes += 1
            if sign != 0:
                prev_sign = sign
        return sign_changes / max(1, len(values) - 2)

    def _objective_flat(self, scores: list[float], window: int, tol: float) -> bool:
        if len(scores) < window + 1:
            return False
        recent = scores[-(window + 1) :]
        diffs = [abs(recent[i + 1] - recent[i]) for i in range(len(recent) - 1)]
        return all(d <= tol for d in diffs)
