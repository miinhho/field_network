from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import LayeredGraph


@dataclass(slots=True)
class TopologicalControlResult:
    node_control: dict[str, float]
    controlled_impact: dict[str, float]
    control_energy: float
    residual_ratio: float
    divergence_norm_before: float
    divergence_norm_after: float
    saturation_ratio: float
    cycle_pressure_mean: float
    gain_k_div: float
    gain_residual_damping: float


class TopologicalFlowController:
    """Node-level topological flow control.

    Steps:
    1) Build edge flow from node impact differences
    2) Compute node divergence via incidence matrix
    3) Solve Laplacian potential and remove residual component
    4) Produce node control input u_i
    """

    def __init__(
        self,
        k_div: float = 0.35,
        k_phi: float = 0.15,
        k_speed: float = 0.2,
        k_cycle: float = 0.12,
        target_speed: float = 0.45,
        residual_damping: float = 0.5,
        control_clip: float = 1.0,
    ) -> None:
        self.k_div = k_div
        self.k_phi = k_phi
        self.k_speed = k_speed
        self.k_cycle = k_cycle
        self.target_speed = target_speed
        self.residual_damping = residual_damping
        self.control_clip = control_clip
        self._k_div_bounds = (0.1, 1.2)
        self._residual_bounds = (0.2, 0.9)

    def compute(
        self,
        graph: LayeredGraph,
        impact_by_actant: dict[str, float],
        state: dict[str, float],
    ) -> TopologicalControlResult:
        nodes = list(graph.actants.keys())
        if not nodes or not graph.interactions:
            return TopologicalControlResult({}, impact_by_actant, 0.0, 0.0, 0.0, 0.0)

        node_index = {node: i for i, node in enumerate(nodes)}
        edges = [(edge.source_id, edge.target_id, edge.weight) for edge in graph.interactions]

        B = self._incidence(nodes, edges, node_index)
        f = self._edge_flow(edges, impact_by_actant)

        div_before = B @ f
        phi = self._solve_potential(B, div_before)
        gradient_flow = B.T @ phi
        residual = f - gradient_flow
        f_controlled = f - self.residual_damping * residual
        div_after = B @ f_controlled

        speed_term = float(state.get("transition_speed", 0.0)) - self.target_speed
        cycle_pressure = self._cycle_pressure(nodes, edges, node_index, impact_by_actant)
        u = -self.k_div * div_before - self.k_phi * phi - self.k_speed * speed_term - self.k_cycle * cycle_pressure
        u = np.clip(u, -self.control_clip, self.control_clip)

        node_control = {node: float(round(u[node_index[node]], 6)) for node in nodes}
        controlled_impact = {
            node: float(max(0.0, impact_by_actant.get(node, 0.0) + node_control[node])) for node in nodes
        }

        residual_ratio = float(np.linalg.norm(residual) / (np.linalg.norm(f) + 1e-9))
        energy = float(np.sum(u * u))
        sat = float(np.mean(np.abs(u) >= (0.98 * self.control_clip)))
        cycle_pressure_mean = float(np.mean(np.abs(cycle_pressure))) if cycle_pressure.size else 0.0
        self._adapt_gains(residual_ratio, div_before, div_after, energy, sat)
        return TopologicalControlResult(
            node_control=node_control,
            controlled_impact=controlled_impact,
            control_energy=round(energy, 6),
            residual_ratio=round(residual_ratio, 6),
            divergence_norm_before=round(float(np.linalg.norm(div_before)), 6),
            divergence_norm_after=round(float(np.linalg.norm(div_after)), 6),
            saturation_ratio=round(sat, 6),
            cycle_pressure_mean=round(cycle_pressure_mean, 6),
            gain_k_div=round(self.k_div, 6),
            gain_residual_damping=round(self.residual_damping, 6),
        )

    def _adapt_gains(
        self,
        residual_ratio: float,
        div_before: np.ndarray,
        div_after: np.ndarray,
        control_energy: float,
        saturation_ratio: float,
    ) -> None:
        before = float(np.linalg.norm(div_before))
        after = float(np.linalg.norm(div_after))
        improved = after < before * 0.97

        if residual_ratio > 0.45 or not improved:
            self.k_div = min(self._k_div_bounds[1], self.k_div * 1.06)
            self.residual_damping = min(self._residual_bounds[1], self.residual_damping * 1.04)
        else:
            self.k_div = max(self._k_div_bounds[0], self.k_div * 0.995)
            self.residual_damping = max(self._residual_bounds[0], self.residual_damping * 0.995)

        if control_energy > 2.5 or saturation_ratio > 0.35:
            self.k_div = max(self._k_div_bounds[0], self.k_div * 0.97)
            self.residual_damping = max(self._residual_bounds[0], self.residual_damping * 0.985)

    def _cycle_pressure(
        self,
        nodes: list[str],
        edges: list[tuple[str, str, float]],
        node_index: dict[str, int],
        impact_by_actant: dict[str, float],
    ) -> np.ndarray:
        adjacency: dict[str, set[str]] = {node: set() for node in nodes}
        for src, dst, _ in edges:
            if src == dst:
                continue
            adjacency.setdefault(src, set()).add(dst)
            adjacency.setdefault(dst, set()).add(src)

        pressure = np.zeros((len(nodes),), dtype=np.float64)
        for node in nodes:
            neigh = list(adjacency.get(node, set()))
            if len(neigh) < 2:
                continue
            tri = 0
            total = 0
            for i in range(len(neigh)):
                for j in range(i + 1, len(neigh)):
                    total += 1
                    if neigh[j] in adjacency.get(neigh[i], set()):
                        tri += 1
            if total == 0:
                continue
            loop_density = tri / total
            pressure[node_index[node]] = loop_density * float(impact_by_actant.get(node, 0.0))
        return pressure

    def _incidence(self, nodes: list[str], edges: list[tuple[str, str, float]], idx: dict[str, int]) -> np.ndarray:
        B = np.zeros((len(nodes), len(edges)), dtype=np.float64)
        for j, (src, dst, _) in enumerate(edges):
            if src not in idx or dst not in idx or src == dst:
                continue
            B[idx[src], j] = -1.0
            B[idx[dst], j] = 1.0
        return B

    def _edge_flow(self, edges: list[tuple[str, str, float]], impact_by_actant: dict[str, float]) -> np.ndarray:
        f = np.zeros((len(edges),), dtype=np.float64)
        for j, (src, dst, w) in enumerate(edges):
            src_i = float(impact_by_actant.get(src, 0.0))
            dst_i = float(impact_by_actant.get(dst, 0.0))
            f[j] = (dst_i - src_i) + 0.05 * float(w)
        return f

    def _solve_potential(self, B: np.ndarray, divergence: np.ndarray) -> np.ndarray:
        if B.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)
        L = B @ B.T
        b = divergence.copy()

        # Fix gauge: pin first node potential to zero.
        L = L.astype(np.float64)
        L[0, :] = 0.0
        L[:, 0] = 0.0
        L[0, 0] = 1.0
        b[0] = 0.0

        eps = 1e-6
        return np.linalg.solve(L + eps * np.eye(L.shape[0]), b)
