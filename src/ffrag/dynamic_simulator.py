from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import networkx as nx

from .flow import (
    ClusterFlowController,
    DynamicGraphAdjuster,
    FlowFieldDynamics,
    FlowSimulator,
    PhaseTransitionAnalyzer,
    StateVectorBuilder,
    TopologicalFlowController,
)
from .models import Actant, Interaction, LayeredGraph, Perturbation


@dataclass(slots=True)
class EdgeDelta:
    source_id: str
    target_id: str
    kind: str
    old_weight: float
    new_weight: float


@dataclass(slots=True)
class EdgeState:
    source_id: str
    target_id: str
    weight: float


@dataclass(slots=True)
class SimulationFrame:
    step: int
    objective_score: float
    critical_transition_score: float
    early_warning_score: float
    adjustment_scale: float
    planner_horizon: int
    edit_budget: int
    node_positions: dict[str, tuple[float, float]]
    node_impacts: dict[str, float]
    node_controls: dict[str, float]
    node_final_values: dict[str, float]
    edges: list[EdgeState]
    top_impacts: list[tuple[str, float]]
    top_controls: list[tuple[str, float]]
    top_final_nodes: list[tuple[str, float]]
    edge_deltas: list[EdgeDelta]


@dataclass(slots=True)
class SimulationTrace:
    frames: list[SimulationFrame]
    final_graph: LayeredGraph
    layout: dict[str, tuple[float, float]]


class DynamicGraphSimulator:
    """Step-by-step simulator for observing dynamic node/edge updates."""

    def __init__(self) -> None:
        self.state_builder = StateVectorBuilder()
        self.simulator = FlowSimulator()
        self.dynamics = FlowFieldDynamics()
        self.phase_analyzer = PhaseTransitionAnalyzer()
        self.cluster_controller = ClusterFlowController()
        self.controller = TopologicalFlowController()
        self.adjuster = DynamicGraphAdjuster()

    def run(
        self,
        graph: LayeredGraph,
        perturbation: Perturbation,
        steps: int = 5,
        top_k: int = 5,
    ) -> SimulationTrace:
        current_graph = graph
        frames: list[SimulationFrame] = []
        trajectory: list[dict[str, float]] = []
        distance_series: list[float] = []
        objective_scores: list[float] = []
        topological_tensions: list[float] = []
        prev_phase_context: dict[str, float] = {}
        prev_layout: dict[str, tuple[float, float]] | None = None

        for step in range(1, max(1, steps) + 1):
            before_edges = self._edge_weights(current_graph)
            prop = self.simulator.propagate(current_graph, perturbation)

            target = perturbation.targets[0] if perturbation.targets else next(iter(current_graph.actants), "")
            initial_state = self.state_builder.build(current_graph, target).values if target else {}
            history = self._history_states(current_graph, target, prop.impact_by_actant)
            shock = self._shock_vector(perturbation)
            dyn = self.dynamics.simulate(initial_state=initial_state, history=history, shock=shock)
            final_state = dyn.snapshots[-1].state if dyn.snapshots else initial_state
            final_distance = dyn.snapshots[-1].attractor_distance if dyn.snapshots else 0.0

            trajectory.append(final_state)
            distance_series.append(final_distance)

            phase_signal = float(prev_phase_context.get("critical_transition_score", 0.0))
            cluster_plan = self.cluster_controller.plan(
                current_graph,
                prop.impact_by_actant,
                final_state,
                phase_signal=phase_signal,
            )
            control = self.controller.compute(
                current_graph,
                cluster_plan.coarse_controlled_impact,
                final_state,
                phase_signal=phase_signal,
            )
            phase = self.phase_analyzer.analyze(
                trajectory=trajectory,
                attractor_distances=distance_series,
                objective_scores=objective_scores + [control.objective_score],
                topological_tensions=topological_tensions + [control.topological_tension],
            )
            phase_map = {
                "critical_transition_score": phase.critical_transition_score,
                "early_warning_score": phase.early_warning_score,
                "coherence_break_score": phase.coherence_break_score,
                "critical_slowing_score": phase.critical_slowing_score,
                "hysteresis_proxy_score": phase.hysteresis_proxy_score,
            }
            adjust = self.adjuster.adjust(
                current_graph,
                control.controlled_impact,
                final_state,
                phase_context=phase_map,
                control_context={
                    "residual_ratio": control.residual_ratio,
                    "divergence_norm_after": control.divergence_norm_after,
                    "control_energy": control.control_energy,
                },
            )

            objective_scores.append(control.objective_score)
            topological_tensions.append(control.topological_tension)
            prev_phase_context = phase_map
            current_graph = adjust.adjusted_graph

            after_edges = self._edge_weights(current_graph)
            deltas = self._edge_deltas(before_edges, after_edges)
            top_impacts = self._topk(prop.impact_by_actant, top_k)
            top_controls = self._topk(control.node_control, top_k, by_abs=True)
            top_final = self._topk(control.controlled_impact, top_k)
            frame_layout = self._layout(current_graph, prev_layout)
            frame_layout = self._advect_layout(
                frame_layout,
                control.node_control,
                prop.impact_by_actant,
            )
            prev_layout = frame_layout

            frames.append(
                SimulationFrame(
                    step=step,
                    objective_score=float(control.objective_score),
                    critical_transition_score=float(phase.critical_transition_score),
                    early_warning_score=float(phase.early_warning_score),
                    adjustment_scale=float(adjust.selected_adjustment_scale),
                    planner_horizon=int(adjust.selected_planner_horizon),
                    edit_budget=int(adjust.selected_edit_budget),
                    node_positions=frame_layout,
                    node_impacts={k: float(round(v, 6)) for k, v in prop.impact_by_actant.items()},
                    node_controls={k: float(round(v, 6)) for k, v in control.node_control.items()},
                    node_final_values={k: float(round(v, 6)) for k, v in control.controlled_impact.items()},
                    edges=self._edge_states(current_graph),
                    top_impacts=top_impacts,
                    top_controls=top_controls,
                    top_final_nodes=top_final,
                    edge_deltas=deltas,
                )
            )

        layout = prev_layout or self._layout(current_graph)
        return SimulationTrace(frames=frames, final_graph=current_graph, layout=layout)

    def demo_graph(self) -> LayeredGraph:
        g = LayeredGraph(graph_id="sim-demo", schema_version="0.1")
        for nid in ["a", "b", "c", "d", "e", "hub"]:
            g.actants[nid] = Actant(actant_id=nid, kind="entity", label=nid.upper())
        base = datetime(2026, 2, 1, 9, 0, 0)
        edges = [
            ("hub", "a", "social", 1.1),
            ("hub", "b", "social", 1.0),
            ("hub", "c", "temporal", 0.9),
            ("a", "d", "spatial", 0.6),
            ("b", "e", "spatial", 0.7),
            ("c", "d", "temporal", 0.5),
            ("d", "e", "social", 0.4),
        ]
        for i, (s, t, layer, w) in enumerate(edges):
            g.interactions.append(
                Interaction(
                    interaction_id=f"sim_e{i}",
                    timestamp=base + timedelta(minutes=i),
                    source_id=s,
                    target_id=t,
                    layer=layer,
                    weight=w,
                )
            )
        return g

    def _history_states(
        self,
        graph: LayeredGraph,
        target: str,
        impact_by_actant: dict[str, float],
    ) -> list[dict[str, float]]:
        if not target:
            return []
        ranked = sorted(impact_by_actant.items(), key=lambda x: x[1], reverse=True)
        candidates = [target] + [node for node, _ in ranked if node != target][:2]
        out: list[dict[str, float]] = []
        for node in candidates:
            if node in graph.actants:
                out.append(self.state_builder.build(graph, node).values)
        return out

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

    def _topk(self, values: dict[str, float], k: int, by_abs: bool = False) -> list[tuple[str, float]]:
        if by_abs:
            ranked = sorted(values.items(), key=lambda x: abs(x[1]), reverse=True)
        else:
            ranked = sorted(values.items(), key=lambda x: x[1], reverse=True)
        return [(nid, float(round(v, 6))) for nid, v in ranked[: max(1, k)]]

    def _edge_weights(self, graph: LayeredGraph) -> dict[tuple[str, str], float]:
        out: dict[tuple[str, str], float] = {}
        for e in graph.interactions:
            key = tuple(sorted((e.source_id, e.target_id)))
            out[key] = out.get(key, 0.0) + float(e.weight)
        return out

    def _edge_deltas(
        self,
        before: dict[tuple[str, str], float],
        after: dict[tuple[str, str], float],
        top_k: int = 8,
    ) -> list[EdgeDelta]:
        keys = set(before.keys()) | set(after.keys())
        rows: list[EdgeDelta] = []
        for key in keys:
            old = float(before.get(key, 0.0))
            new = float(after.get(key, 0.0))
            if abs(new - old) <= 1e-9:
                continue
            if old <= 1e-9:
                kind = "added"
            elif new <= 1e-9:
                kind = "removed"
            else:
                kind = "weight"
            rows.append(
                EdgeDelta(
                    source_id=key[0],
                    target_id=key[1],
                    kind=kind,
                    old_weight=round(old, 6),
                    new_weight=round(new, 6),
                )
            )
        rows.sort(key=lambda r: abs(r.new_weight - r.old_weight), reverse=True)
        return rows[:top_k]

    def _edge_states(self, graph: LayeredGraph) -> list[EdgeState]:
        return [
            EdgeState(
                source_id=e.source_id,
                target_id=e.target_id,
                weight=float(round(e.weight, 6)),
            )
            for e in graph.interactions
        ]

    def _layout(
        self,
        graph: LayeredGraph,
        prev_layout: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, tuple[float, float]]:
        g = nx.Graph()
        for nid in graph.actants.keys():
            g.add_node(nid)
        for e in graph.interactions:
            g.add_edge(e.source_id, e.target_id, weight=max(0.01, float(e.weight)))
        if g.number_of_nodes() == 0:
            return {}
        pos_arg = None
        if prev_layout:
            pos_arg = {k: prev_layout[k] for k in g.nodes() if k in prev_layout}
        pos = nx.spring_layout(g, seed=7, weight="weight", pos=pos_arg, iterations=25)
        out: dict[str, tuple[float, float]] = {}
        for nid, xy in pos.items():
            out[nid] = (float(xy[0]), float(xy[1]))
        return out

    def _advect_layout(
        self,
        layout: dict[str, tuple[float, float]],
        node_control: dict[str, float],
        node_impact: dict[str, float],
    ) -> dict[str, tuple[float, float]]:
        if not layout:
            return {}
        max_ctrl = max((abs(v) for v in node_control.values()), default=1.0) or 1.0
        max_imp = max((abs(v) for v in node_impact.values()), default=1.0) or 1.0
        out: dict[str, tuple[float, float]] = {}
        for nid, (x, y) in layout.items():
            c = float(node_control.get(nid, 0.0)) / max_ctrl
            p = float(node_impact.get(nid, 0.0)) / max_imp
            nx_ = max(-1.2, min(1.2, x + 0.18 * c))
            ny_ = max(-1.2, min(1.2, y + 0.18 * p))
            out[nid] = (nx_, ny_)
        return out
