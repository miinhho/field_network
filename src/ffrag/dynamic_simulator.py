from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math

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
        position_model: str = "hybrid",
        physics_substeps: int = 4,
        physics_dt: float = 0.06,
        physics_damping: float = 0.92,
        physics_spring_k: float = 0.85,
        physics_field_k: float = 0.60,
    ) -> SimulationTrace:
        current_graph = graph
        frames: list[SimulationFrame] = []
        trajectory: list[dict[str, float]] = []
        distance_series: list[float] = []
        objective_scores: list[float] = []
        topological_tensions: list[float] = []
        prev_phase_context: dict[str, float] = {}
        prev_layout: dict[str, tuple[float, float]] | None = None
        prev_velocity: dict[str, tuple[float, float]] = {}

        for step in range(1, max(1, steps) + 1):
            before_edges = self._edge_weights(current_graph)
            prop = self.simulator.propagate(current_graph, perturbation)

            target = perturbation.targets[0] if perturbation.targets else next(iter(current_graph.actants), "")
            if target and target not in current_graph.actants:
                target = next(iter(current_graph.actants), "")
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
            base_layout = self._layout(current_graph, prev_layout)
            if position_model == "physics":
                frame_layout, prev_velocity = self._physical_step(
                    current_graph,
                    base_layout,
                    prev_velocity,
                    control.node_control,
                    prop.impact_by_actant,
                    substeps=max(1, int(physics_substeps)),
                    dt=max(0.005, float(physics_dt)),
                    damping=max(0.5, min(0.999, float(physics_damping))),
                    spring_k=max(0.01, float(physics_spring_k)),
                    field_k=max(0.0, float(physics_field_k)),
                )
            else:
                frame_layout = self._advect_layout(
                    base_layout,
                    control.node_control,
                    prop.impact_by_actant,
                )
                prev_velocity = {}
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
        return self.generate_graph(node_count=6, avg_degree=2.3, seed=7)

    def generate_graph(
        self,
        node_count: int = 6,
        avg_degree: float = 2.3,
        seed: int = 7,
    ) -> LayeredGraph:
        n = max(2, int(node_count))
        if n <= 10:
            return self._small_demo_graph()
        return self._random_graph(node_count=n, avg_degree=max(0.5, float(avg_degree)), seed=seed)

    def _small_demo_graph(self) -> LayeredGraph:
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

    def _random_graph(self, node_count: int, avg_degree: float, seed: int) -> LayeredGraph:
        g = LayeredGraph(graph_id=f"sim-random-{node_count}", schema_version="0.1")
        node_ids = [f"n{i}" for i in range(node_count)]
        for nid in node_ids:
            g.actants[nid] = Actant(actant_id=nid, kind="entity", label=nid.upper())

        max_edges = node_count * (node_count - 1) // 2
        m = int(min(max_edges, max(node_count - 1, math.floor(node_count * avg_degree / 2.0))))
        rg = nx.gnm_random_graph(node_count, m, seed=seed)
        if not nx.is_connected(rg):
            comps = [list(c) for c in nx.connected_components(rg)]
            for i in range(len(comps) - 1):
                rg.add_edge(comps[i][0], comps[i + 1][0])

        base = datetime(2026, 2, 1, 9, 0, 0)
        layers = ("social", "temporal", "spatial")
        for i, (u, v) in enumerate(rg.edges()):
            layer = layers[(u + v + i) % len(layers)]
            weight = 0.35 + ((u * 17 + v * 13 + i * 7) % 100) / 100.0
            g.interactions.append(
                Interaction(
                    interaction_id=f"sim_re{i}",
                    timestamp=base + timedelta(seconds=i),
                    source_id=node_ids[u],
                    target_id=node_ids[v],
                    layer=layer,
                    weight=float(round(weight, 4)),
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
        iterations = 25 if g.number_of_nodes() <= 400 else 8
        try:
            pos = nx.spring_layout(g, seed=7, weight="weight", pos=pos_arg, iterations=iterations)
            out: dict[str, tuple[float, float]] = {}
            for nid, xy in pos.items():
                out[nid] = (float(xy[0]), float(xy[1]))
            return out
        except ModuleNotFoundError:
            return self._layout_no_scipy(g, prev_layout)

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

    def _layout_no_scipy(
        self,
        g: nx.Graph,
        prev_layout: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, tuple[float, float]]:
        nodes = list(g.nodes())
        n = max(1, len(nodes))
        pos: dict[str, tuple[float, float]] = {}

        if prev_layout:
            for nid in nodes:
                if nid in prev_layout:
                    pos[nid] = prev_layout[nid]
        if len(pos) != len(nodes):
            for i, nid in enumerate(nodes):
                if nid in pos:
                    continue
                angle = 2.0 * math.pi * (i / n)
                r = 0.55 + 0.35 * ((i * 31) % 100) / 100.0
                pos[nid] = (r * math.cos(angle), r * math.sin(angle))

        rounds = 2 if n <= 2000 else 1
        alpha = 0.22 if n <= 2000 else 0.14
        for _ in range(rounds):
            nxt: dict[str, tuple[float, float]] = {}
            for nid in nodes:
                nbrs = list(g.neighbors(nid))
                if not nbrs:
                    nxt[nid] = pos[nid]
                    continue
                sx = 0.0
                sy = 0.0
                sw = 0.0
                for nb in nbrs:
                    w = float(g[nid][nb].get("weight", 1.0))
                    px, py = pos[nb]
                    sx += w * px
                    sy += w * py
                    sw += w
                cx = sx / max(sw, 1e-9)
                cy = sy / max(sw, 1e-9)
                x, y = pos[nid]
                nx_ = (1.0 - alpha) * x + alpha * cx
                ny_ = (1.0 - alpha) * y + alpha * cy
                nxt[nid] = (max(-1.2, min(1.2, nx_)), max(-1.2, min(1.2, ny_)))
            pos = nxt
        return pos

    def _physical_step(
        self,
        graph: LayeredGraph,
        initial_pos: dict[str, tuple[float, float]],
        prev_velocity: dict[str, tuple[float, float]],
        node_control: dict[str, float],
        node_impact: dict[str, float],
        substeps: int,
        dt: float,
        damping: float,
        spring_k: float,
        field_k: float,
    ) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
        pos = dict(initial_pos)
        vel: dict[str, tuple[float, float]] = {
            nid: tuple(prev_velocity.get(nid, (0.0, 0.0))) for nid in graph.actants.keys()
        }
        max_ctrl = max((abs(v) for v in node_control.values()), default=1.0) or 1.0
        max_imp = max((abs(v) for v in node_impact.values()), default=1.0) or 1.0

        masses: dict[str, float] = {nid: 1.0 for nid in graph.actants.keys()}
        degs: dict[str, float] = {nid: 0.0 for nid in graph.actants.keys()}
        for e in graph.interactions:
            w = max(0.01, float(e.weight))
            degs[e.source_id] = degs.get(e.source_id, 0.0) + w
            degs[e.target_id] = degs.get(e.target_id, 0.0) + w
        for nid, d in degs.items():
            masses[nid] = 1.0 + math.log1p(max(0.0, d))

        for _ in range(substeps):
            forces: dict[str, list[float]] = {nid: [0.0, 0.0] for nid in graph.actants.keys()}

            # Edge spring + short-range collision force.
            for e in graph.interactions:
                a = e.source_id
                b = e.target_id
                if a not in pos or b not in pos:
                    continue
                ax, ay = pos[a]
                bx, by = pos[b]
                dx = bx - ax
                dy = by - ay
                dist = math.sqrt(dx * dx + dy * dy) + 1e-9
                ux = dx / dist
                uy = dy / dist
                w = max(0.01, float(e.weight))
                rest = max(0.05, 0.28 / (0.35 + w))
                fs = spring_k * w * (dist - rest)
                fx = fs * ux
                fy = fs * uy
                forces[a][0] += fx
                forces[a][1] += fy
                forces[b][0] -= fx
                forces[b][1] -= fy

                # Avoid node overlap under heavy attraction.
                if dist < 0.04:
                    rep = (0.04 - dist) * 2.0
                    rx = rep * ux
                    ry = rep * uy
                    forces[a][0] -= rx
                    forces[a][1] -= ry
                    forces[b][0] += rx
                    forces[b][1] += ry

            # External flow field force from control/impact.
            for nid in graph.actants.keys():
                c = float(node_control.get(nid, 0.0)) / max_ctrl
                p = float(node_impact.get(nid, 0.0)) / max_imp
                forces[nid][0] += field_k * c
                forces[nid][1] += field_k * p

            # Weak central restoring term to keep bounded domain.
            for nid in graph.actants.keys():
                x, y = pos.get(nid, (0.0, 0.0))
                forces[nid][0] += -0.10 * x
                forces[nid][1] += -0.10 * y

            # Semi-implicit Euler integration.
            for nid in graph.actants.keys():
                x, y = pos.get(nid, (0.0, 0.0))
                vx, vy = vel.get(nid, (0.0, 0.0))
                fx, fy = forces[nid]
                mass = max(0.1, masses.get(nid, 1.0))
                vx = (vx + dt * (fx / mass)) * damping
                vy = (vy + dt * (fy / mass)) * damping
                x = x + dt * vx
                y = y + dt * vy
                x = max(-1.2, min(1.2, x))
                y = max(-1.2, min(1.2, y))
                pos[nid] = (x, y)
                vel[nid] = (vx, vy)

        return pos, vel
