from __future__ import annotations

from datetime import datetime, timezone

from .compose import compose_answer
from .flow import FlowDynamicsAnalyzer, FlowFieldDynamics, FlowSimulator, StateVectorBuilder
from .models import Answer, LayeredGraph, Perturbation, Query
from .retrieval import GraphRetriever
from .router import QueryRouter


class FlowGraphRAG:
    """PoC orchestration for describe/predict/intervene queries."""

    def __init__(self) -> None:
        self.router = QueryRouter()
        self.retriever = GraphRetriever()
        self.state_builder = StateVectorBuilder()
        self.simulator = FlowSimulator()
        self.dynamics = FlowFieldDynamics()
        self.analyzer = FlowDynamicsAnalyzer()

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
        result = self.simulator.propagate(graph, p)
        target = p.targets[0] if p.targets else next(iter(graph.actants), "")
        initial_state = self.state_builder.build(graph, target).values if target else {}
        history = self._history_states(graph, target, result.impact_by_actant)
        dyn = self.dynamics.simulate(initial_state=initial_state, history=history, shock=self._shock_vector(p))
        top_impacts = sorted(result.impact_by_actant.items(), key=lambda item: item[1], reverse=True)[:3]
        final_distance = dyn.snapshots[-1].attractor_distance if dyn.snapshots else 0.0
        trajectory = [snap.state for snap in dyn.snapshots]
        distances = [snap.attractor_distance for snap in dyn.snapshots]
        trans = self.analyzer.transition_analysis(trajectory)
        resil = self.analyzer.resilience_analysis(distances)
        dominant_transition = self._dominant_transition_prob(trans.transition_matrix)

        claims = [
            f"Predicted propagation reached {len(result.impact_by_actant)} actants in {result.hops_executed} hops.",
            f"Top impacted actants: {', '.join(actant for actant, _ in top_impacts) if top_impacts else 'none'}.",
            f"Dynamics simulation produced {len(dyn.snapshots)} state updates; final attractor distance={final_distance:.4f}.",
            f"State trajectory classes: {', '.join(trans.states) if trans.states else 'none'}.",
            f"Attractor basin state: {trans.attractor_basin_state or 'none'}; triggers={len(trans.transition_triggers)}.",
        ]
        metrics = {
            "hops_executed": float(result.hops_executed),
            "affected_actants": float(len(result.impact_by_actant)),
            "stabilized": 1.0 if result.stabilized else 0.0,
            "dynamics_steps": float(len(dyn.snapshots)),
            "final_attractor_distance": float(final_distance),
            "dynamics_stabilized": 1.0 if dyn.stabilized else 0.0,
            "state_class_count": float(len(set(trans.states))),
            "dominant_transition_prob": float(dominant_transition),
            "recovery_rate": float(resil.recovery_rate),
            "overshoot_index": float(resil.overshoot_index),
            "settling_time": float(resil.settling_time),
            "path_efficiency": float(resil.path_efficiency),
            "transition_trigger_count": float(len(trans.transition_triggers)),
        }
        evidence_ids = [f"perturbation:{p.perturbation_id}"]
        return compose_answer("predict", claims, evidence_ids, metrics, uncertainty=0.35)

    def _intervene(self, graph: LayeredGraph, query: Query, perturbation: Perturbation | None) -> Answer:
        p = perturbation or self._default_perturbation(graph)
        result = self.simulator.propagate(graph, p)
        target = p.targets[0] if p.targets else next(iter(graph.actants), "")
        initial_state = self.state_builder.build(graph, target).values if target else {}
        history = self._history_states(graph, target, result.impact_by_actant)
        baseline = self.dynamics.simulate(initial_state=initial_state, history=history, shock=self._shock_vector(p))
        intervention = self.dynamics.simulate(
            initial_state=initial_state,
            history=history,
            shock=self._counter_shock_vector(p),
        )
        rewires = result.rewired_edges[:3]
        rewire_text = ", ".join(f"{src}->{dst}" for src, dst in rewires) if rewires else "no rewiring suggested"
        base_dist = baseline.snapshots[-1].attractor_distance if baseline.snapshots else 0.0
        intervention_dist = intervention.snapshots[-1].attractor_distance if intervention.snapshots else 0.0
        improvement = base_dist - intervention_dist
        base_traj = [snap.state for snap in baseline.snapshots]
        int_traj = [snap.state for snap in intervention.snapshots]
        base_distances = [snap.attractor_distance for snap in baseline.snapshots]
        int_distances = [snap.attractor_distance for snap in intervention.snapshots]
        base_trans = self.analyzer.transition_analysis(base_traj)
        int_trans = self.analyzer.transition_analysis(int_traj)
        int_resil = self.analyzer.resilience_analysis(int_distances, baseline_distances=base_distances)

        claims = [
            f"Suggested rewiring candidates: {rewire_text}.",
            "Intervention proposal is based on structural impact propagation.",
            f"Counter-shock simulation changed attractor distance by {improvement:.4f} (positive is better).",
            f"Intervention hysteresis index={int_resil.hysteresis_index:.4f}.",
            f"Intervention basin state: {int_trans.attractor_basin_state or 'none'}; triggers={len(int_trans.transition_triggers)}.",
        ]
        metrics = {
            "rewire_candidates": float(len(rewires)),
            "affected_actants": float(len(result.impact_by_actant)),
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
