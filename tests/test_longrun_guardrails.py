from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys
import unittest

import numpy as np
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag import Actant, Interaction, LayeredGraph, Perturbation
from ffrag.flow import (
    ClusterFlowController,
    DynamicGraphAdjuster,
    FlowFieldDynamics,
    FlowSimulator,
    PhaseTransitionAnalyzer,
    StateVectorBuilder,
    SupervisoryControlState,
    SupervisoryMetricsAnalyzer,
    TopologicalFlowController,
)


class LongRunGuardrailIntegrationTests(unittest.TestCase):
    def _dense_graph(self, n: int = 9) -> LayeredGraph:
        g = LayeredGraph(graph_id="lr-dense", schema_version="0.1")
        nodes = [f"d{i}" for i in range(n)]
        for node in nodes:
            g.actants[node] = Actant(node, "entity", node.upper())
        base = datetime(2026, 2, 13, 9, 0, 0)
        idx = 0
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                g.interactions.append(
                    Interaction(
                        interaction_id=f"de{idx}",
                        timestamp=base + timedelta(seconds=idx),
                        source_id=nodes[i],
                        target_id=nodes[j],
                        layer="dense",
                        weight=1.0,
                    )
                )
                idx += 1
        return g

    def _fragmentation_prone_graph(self) -> LayeredGraph:
        g = LayeredGraph(graph_id="lr-frag", schema_version="0.1")
        for node in ["a", "b", "c", "x", "y", "z", "bridge"]:
            g.actants[node] = Actant(node, "entity", node.upper())
        base = datetime(2026, 2, 13, 10, 0, 0)
        edges = [
            ("a", "b", 1.0),
            ("b", "c", 1.0),
            ("a", "c", 0.9),
            ("x", "y", 1.0),
            ("y", "z", 1.0),
            ("x", "z", 0.9),
            ("c", "bridge", 0.5),
            ("bridge", "x", 0.5),
        ]
        for i, (s, t, w) in enumerate(edges):
            g.interactions.append(
                Interaction(
                    interaction_id=f"fe{i}",
                    timestamp=base + timedelta(minutes=i),
                    source_id=s,
                    target_id=t,
                    layer="frag",
                    weight=w,
                )
            )
        return g

    def _noisy_graph(self) -> LayeredGraph:
        g = LayeredGraph(graph_id="lr-noisy", schema_version="0.1")
        for i in range(10):
            node = f"n{i}"
            g.actants[node] = Actant(node, "entity", node.upper())
        base = datetime(2026, 2, 13, 11, 0, 0)
        rng = np.random.default_rng(42)
        idx = 0
        nodes = list(g.actants.keys())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if rng.random() < 0.27:
                    w = float(0.4 + 0.9 * rng.random())
                    g.interactions.append(
                        Interaction(
                            interaction_id=f"ne{idx}",
                            timestamp=base + timedelta(seconds=idx),
                            source_id=nodes[i],
                            target_id=nodes[j],
                            layer="noisy",
                            weight=w,
                        )
                    )
                    idx += 1
        return g

    def _run_long_cycle(
        self,
        graph: LayeredGraph,
        target: str,
        steps: int,
        intensity_base: float,
        noisy: bool = False,
        signed_centering: bool = False,
    ) -> dict[str, float]:
        simulator = FlowSimulator()
        state_builder = StateVectorBuilder()
        dynamics = FlowFieldDynamics()
        cluster = ClusterFlowController()
        control = TopologicalFlowController()
        phase = PhaseTransitionAnalyzer()
        adjuster = DynamicGraphAdjuster()
        supervisory = SupervisoryMetricsAnalyzer()
        supervisory_state: SupervisoryControlState | None = None

        current = graph
        traj: list[dict[str, float]] = []
        dists: list[float] = []
        objectives: list[float] = []
        tensions: list[float] = []
        churn_series: list[int] = []
        forgetting_series: list[float] = []
        confusion_series: list[float] = []
        critical_series: list[float] = []
        min_coarse_series: list[float] = []
        min_controlled_series: list[float] = []

        important_nodes = [target]
        important_nodes.extend([n for n in current.actants.keys() if n != target][:2])
        now = datetime(2026, 2, 14, 9, 0, 0)
        phase_signal = 0.0

        for step in range(steps):
            intensity = intensity_base
            step_target = target
            if noisy:
                intensity = max(0.2, float(intensity_base + 0.5 * np.sin(step * 1.3)))
                nodes = list(current.actants.keys())
                step_target = nodes[step % len(nodes)]
            p = Perturbation(
                perturbation_id=f"lr-{step}",
                timestamp=now + timedelta(minutes=step),
                targets=[step_target],
                intensity=intensity,
            )
            prop = simulator.propagate(current, p)
            impact_input = dict(prop.impact_by_actant)
            if signed_centering and impact_input:
                mean_impact = sum(impact_input.values()) / len(impact_input)
                impact_input = {k: float(v - mean_impact) for k, v in impact_input.items()}
            init = state_builder.build(current, step_target).values
            shock = {
                "social_entropy": 0.1 * intensity,
                "temporal_regularity": -0.12 * intensity,
                "spatial_range": 0.05 * intensity,
                "schedule_density": 0.18 * intensity,
                "network_centrality": 0.08 * intensity,
                "transition_speed": 0.22 * intensity,
            }
            dyn = dynamics.simulate(initial_state=init, history=[init], shock=shock, steps=3)
            final_state = dyn.snapshots[-1].state if dyn.snapshots else init
            final_dist = dyn.snapshots[-1].attractor_distance if dyn.snapshots else 0.0
            traj.append(final_state)
            dists.append(final_dist)

            coarse = cluster.plan(current, impact_input, final_state, phase_signal=phase_signal)
            controlled = control.compute(
                current,
                coarse.coarse_controlled_impact,
                final_state,
                phase_signal=phase_signal,
            )
            min_coarse_series.append(min(coarse.coarse_controlled_impact.values(), default=0.0))
            min_controlled_series.append(min(controlled.controlled_impact.values(), default=0.0))
            phase_out = phase.analyze(
                trajectory=traj,
                attractor_distances=dists,
                objective_scores=objectives + [controlled.objective_score],
                topological_tensions=tensions + [controlled.topological_tension],
            )
            phase_signal = phase_out.critical_transition_score
            super_metrics, supervisory_state = supervisory.analyze(
                current,
                controlled.controlled_impact,
                state=supervisory_state,
                important_nodes=important_nodes,
            )
            adj = adjuster.adjust(
                current,
                controlled.controlled_impact,
                final_state,
                phase_context={
                    "critical_transition_score": phase_out.critical_transition_score,
                    "early_warning_score": phase_out.early_warning_score,
                    "coherence_break_score": phase_out.coherence_break_score,
                    "critical_slowing_score": phase_out.critical_slowing_score,
                    "hysteresis_proxy_score": phase_out.hysteresis_proxy_score,
                },
                control_context={
                    "residual_ratio": controlled.residual_ratio,
                    "divergence_norm_after": controlled.divergence_norm_after,
                    "control_energy": controlled.control_energy,
                },
                supervisory_context={
                    "confusion_score": super_metrics.confusion_score,
                    "forgetting_score": super_metrics.forgetting_score,
                },
            )
            current = adj.adjusted_graph
            objectives.append(controlled.objective_score)
            tensions.append(controlled.topological_tension)
            churn_series.append(adj.applied_new_edges + adj.applied_drop_edges)
            forgetting_series.append(adj.supervisory_forgetting_score)
            confusion_series.append(adj.supervisory_confusion_score)
            critical_series.append(phase_out.critical_transition_score)

        retention = self._retention_floor(current, important_nodes)
        diversity = self._diversity_floor(current)
        return {
            "mean_churn": float(sum(churn_series) / max(1, len(churn_series))),
            "max_churn": float(max(churn_series) if churn_series else 0.0),
            "mean_forgetting": float(sum(forgetting_series) / max(1, len(forgetting_series))),
            "mean_confusion": float(sum(confusion_series) / max(1, len(confusion_series))),
            "max_critical": float(max(critical_series) if critical_series else 0.0),
            "retention_floor": retention,
            "diversity_floor": diversity,
            "min_coarse_impact": float(min(min_coarse_series) if min_coarse_series else 0.0),
            "min_controlled_impact": float(min(min_controlled_series) if min_controlled_series else 0.0),
        }

    def _retention_floor(self, graph: LayeredGraph, important_nodes: list[str]) -> float:
        if not important_nodes:
            return 1.0
        deg: dict[str, int] = {n: 0 for n in graph.actants.keys()}
        for e in graph.interactions:
            if e.source_id in deg:
                deg[e.source_id] += 1
            if e.target_id in deg and e.target_id != e.source_id:
                deg[e.target_id] += 1
        retained = sum(1 for n in important_nodes if deg.get(n, 0) > 0)
        return retained / len(important_nodes)

    def _diversity_floor(self, graph: LayeredGraph) -> float:
        if not graph.actants:
            return 0.0
        g = self._to_nx(graph)
        if g.number_of_nodes() == 0:
            return 0.0
        if g.number_of_edges() == 0:
            return 0.0
        components = list(nx.connected_components(g))
        largest = max((len(c) for c in components), default=0)
        largest_ratio = largest / g.number_of_nodes()
        # A simple "anti-collapse" proxy: maintain some degree spread.
        degrees = [float(g.degree(n)) for n in g.nodes()]
        mean_deg = sum(degrees) / len(degrees)
        if mean_deg <= 1e-9:
            spread = 0.0
        else:
            var = sum((d - mean_deg) ** 2 for d in degrees) / len(degrees)
            spread = min(1.0, (var ** 0.5) / mean_deg)
        return max(0.0, min(1.0, 0.6 * largest_ratio + 0.4 * spread))

    def _to_nx(self, graph: LayeredGraph):
        g = nx.Graph()
        for n in graph.actants.keys():
            g.add_node(n)
        for e in graph.interactions:
            if e.source_id == e.target_id:
                continue
            g.add_edge(e.source_id, e.target_id)
        return g

    def test_collapse_prone_longrun_keeps_bounded_churn(self) -> None:
        out = self._run_long_cycle(
            graph=self._dense_graph(),
            target="d0",
            steps=20,
            intensity_base=1.3,
            noisy=False,
        )
        self.assertLessEqual(out["mean_churn"], 1.6)
        self.assertLessEqual(out["max_churn"], 2.0)
        self.assertGreaterEqual(out["retention_floor"], 0.66)
        self.assertGreaterEqual(out["diversity_floor"], 0.35)

    def test_fragmentation_prone_longrun_keeps_retention_and_connectivity(self) -> None:
        out = self._run_long_cycle(
            graph=self._fragmentation_prone_graph(),
            target="bridge",
            steps=22,
            intensity_base=1.1,
            noisy=False,
        )
        self.assertLessEqual(out["mean_churn"], 1.5)
        self.assertGreaterEqual(out["retention_floor"], 0.66)
        self.assertGreaterEqual(out["diversity_floor"], 0.3)
        self.assertLessEqual(out["mean_forgetting"], 0.8)

    def test_noisy_longrun_keeps_guardrails_active(self) -> None:
        out = self._run_long_cycle(
            graph=self._noisy_graph(),
            target="n0",
            steps=24,
            intensity_base=1.0,
            noisy=True,
        )
        self.assertLessEqual(out["mean_churn"], 1.7)
        self.assertLessEqual(out["max_churn"], 2.0)
        self.assertGreaterEqual(out["retention_floor"], 0.5)
        self.assertGreaterEqual(out["diversity_floor"], 0.25)
        self.assertGreaterEqual(out["max_critical"], 0.0)
        self.assertLessEqual(out["max_critical"], 1.0)

    def test_signed_projection_longrun_remains_stable(self) -> None:
        out = self._run_long_cycle(
            graph=self._noisy_graph(),
            target="n0",
            steps=20,
            intensity_base=1.0,
            noisy=True,
            signed_centering=True,
        )
        self.assertLess(out["min_coarse_impact"], 0.0)
        self.assertLess(out["min_controlled_impact"], 0.0)
        self.assertLessEqual(out["mean_churn"], 2.2)
        self.assertLessEqual(out["max_churn"], 3.0)
        self.assertGreaterEqual(out["retention_floor"], 0.45)
        self.assertGreaterEqual(out["diversity_floor"], 0.2)


if __name__ == "__main__":
    unittest.main()
