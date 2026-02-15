from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import LayeredGraph
from .topology import SimplicialTopologyModel


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
    higher_order_pressure_mean: float
    gain_k_div: float
    gain_residual_damping: float
    gain_k_higher: float
    objective_score: float
    gradient_norm: float
    curl_norm: float
    harmonic_norm: float
    curl_ratio: float
    harmonic_ratio: float
    simplex_density: float
    topological_tension: float


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
        k_curl: float = 0.18,
        k_harmonic: float = 0.16,
        k_higher: float = 0.1,
        target_speed: float = 0.45,
        residual_damping: float = 0.5,
        control_clip: float = 1.0,
        objective_weights: tuple[float, float, float, float, float] = (1.0, 0.6, 0.25, 0.35, 0.2),
        objective_step_k_div: float = 0.02,
        objective_step_residual: float = 0.015,
        lookahead_horizon: int = 2,
        lookahead_decay: float = 0.7,
    ) -> None:
        self.k_div = k_div
        self.k_phi = k_phi
        self.k_speed = k_speed
        self.k_cycle = k_cycle
        self.k_curl = k_curl
        self.k_harmonic = k_harmonic
        self.k_higher = k_higher
        self.target_speed = target_speed
        self.residual_damping = residual_damping
        self.control_clip = control_clip
        self.objective_weights = objective_weights
        self.objective_step_k_div = objective_step_k_div
        self.objective_step_residual = objective_step_residual
        self.lookahead_horizon = max(1, lookahead_horizon)
        self.lookahead_decay = max(0.2, min(0.95, lookahead_decay))
        self._k_div_bounds = (0.1, 1.2)
        self._residual_bounds = (0.2, 0.9)
        self._k_higher_bounds = (0.02, 0.6)
        self._prev_objective: float | None = None
        self._direction_k_div: float = 1.0
        self._direction_residual: float = 1.0
        self.topology_model = SimplicialTopologyModel()

    def compute(
        self,
        graph: LayeredGraph,
        impact_by_actant: dict[str, float],
        state: dict[str, float],
        phase_signal: float = 0.0,
    ) -> TopologicalControlResult:
        nodes = list(graph.actants.keys())
        if not nodes or not graph.interactions:
            return TopologicalControlResult(
                node_control={},
                controlled_impact=impact_by_actant,
                control_energy=0.0,
                residual_ratio=0.0,
                divergence_norm_before=0.0,
                divergence_norm_after=0.0,
                saturation_ratio=0.0,
                cycle_pressure_mean=0.0,
                higher_order_pressure_mean=0.0,
                gain_k_div=float(self.k_div),
                gain_residual_damping=float(self.residual_damping),
                gain_k_higher=float(self.k_higher),
                objective_score=0.0,
                gradient_norm=0.0,
                curl_norm=0.0,
                harmonic_norm=0.0,
                curl_ratio=0.0,
                harmonic_ratio=0.0,
                simplex_density=0.0,
                topological_tension=0.0,
            )

        node_index = {node: i for i, node in enumerate(nodes)}
        edges = [(edge.source_id, edge.target_id, edge.weight) for edge in graph.interactions]

        B = self._incidence(nodes, edges, node_index)
        f = self._edge_flow(edges, impact_by_actant)

        div_before = B @ f
        phi = self._solve_potential(B, div_before)
        gradient_flow = B.T @ phi
        C = self._cycle_matrix(nodes, edges, node_index)
        curl_flow = self._solve_curl_component(C, f - gradient_flow)
        harmonic = f - gradient_flow - curl_flow
        speed_term = float(state.get("transition_speed", 0.0)) - self.target_speed
        cycle_pressure = self._cycle_pressure(nodes, edges, node_index, impact_by_actant)
        topo = self.topology_model.compute(graph, impact_by_actant)
        higher_pressure = np.array(
            [float(topo.node_pressure.get(node, 0.0)) for node in nodes],
            dtype=np.float64,
        )
        curl_pressure = self._curl_pressure(C, curl_flow, len(nodes), edges, node_index)
        harmonic_pressure = self._harmonic_pressure(harmonic, len(nodes), edges, node_index)
        cycle_pressure_mean = float(np.mean(np.abs(cycle_pressure))) if cycle_pressure.size else 0.0
        higher_pressure_mean = float(np.mean(np.abs(higher_pressure))) if higher_pressure.size else 0.0

        residual = curl_flow + harmonic
        k_div, residual_damping, k_higher = self._lookahead_select_gains(
            B=B,
            f=f,
            residual=residual,
            harmonic=harmonic,
            div_before=div_before,
            phi=phi,
            speed_term=speed_term,
            cycle_pressure=cycle_pressure,
            higher_pressure=higher_pressure,
            curl_pressure=curl_pressure,
            harmonic_pressure=harmonic_pressure,
            cycle_pressure_mean=cycle_pressure_mean,
            higher_pressure_mean=higher_pressure_mean,
            topological_tension=float(topo.topological_tension),
            flow_norm=float(np.linalg.norm(f) + 1e-9),
            curl_norm=float(np.linalg.norm(curl_flow)),
            harmonic_norm=float(np.linalg.norm(harmonic)),
        )
        self.k_div = k_div
        self.residual_damping = residual_damping
        self.k_higher = k_higher
        self._phase_safety_clamp(phase_signal)

        f_controlled = f - self.residual_damping * residual - self.k_harmonic * harmonic
        div_after = B @ f_controlled

        effective_clip = self._effective_clip(phase_signal)
        u = (
            -self.k_div * div_before
            - self.k_phi * phi
            - self.k_speed * speed_term
            - self.k_cycle * cycle_pressure
            - self.k_higher * higher_pressure
            - self.k_curl * curl_pressure
            - self.k_harmonic * harmonic_pressure
        )
        u = np.clip(u, -effective_clip, effective_clip)

        node_control = {node: float(round(u[node_index[node]], 6)) for node in nodes}
        controlled_impact = {
            node: float(max(0.0, impact_by_actant.get(node, 0.0) + node_control[node])) for node in nodes
        }

        residual_ratio = float(np.linalg.norm(residual) / (np.linalg.norm(f) + 1e-9))
        energy = float(np.sum(u * u))
        sat = float(np.mean(np.abs(u) >= (0.98 * effective_clip)))
        grad_norm = float(np.linalg.norm(gradient_flow))
        curl_norm = float(np.linalg.norm(curl_flow))
        harmonic_norm = float(np.linalg.norm(harmonic))
        flow_norm = float(np.linalg.norm(f) + 1e-9)
        curl_ratio = curl_norm / flow_norm
        harmonic_ratio = harmonic_norm / flow_norm
        objective = self._objective(
            div_after=float(np.linalg.norm(div_after)),
            residual_ratio=residual_ratio,
            control_energy=energy,
            saturation_ratio=sat,
            cycle_pressure=cycle_pressure_mean,
            curl_ratio=curl_ratio,
            harmonic_ratio=harmonic_ratio,
            higher_pressure=higher_pressure_mean,
            topological_tension=float(topo.topological_tension),
        )
        self._adapt_gains(objective, residual_ratio, div_before, div_after, energy, sat)
        self._phase_safety_clamp(phase_signal)
        return TopologicalControlResult(
            node_control=node_control,
            controlled_impact=controlled_impact,
            control_energy=round(energy, 6),
            residual_ratio=round(residual_ratio, 6),
            divergence_norm_before=round(float(np.linalg.norm(div_before)), 6),
            divergence_norm_after=round(float(np.linalg.norm(div_after)), 6),
            saturation_ratio=round(sat, 6),
            cycle_pressure_mean=round(cycle_pressure_mean, 6),
            higher_order_pressure_mean=round(higher_pressure_mean, 6),
            gain_k_div=round(self.k_div, 6),
            gain_residual_damping=round(self.residual_damping, 6),
            gain_k_higher=round(self.k_higher, 6),
            objective_score=round(objective, 6),
            gradient_norm=round(grad_norm, 6),
            curl_norm=round(curl_norm, 6),
            harmonic_norm=round(harmonic_norm, 6),
            curl_ratio=round(curl_ratio, 6),
            harmonic_ratio=round(harmonic_ratio, 6),
            simplex_density=round(float(topo.simplex_density), 6),
            topological_tension=round(float(topo.topological_tension), 6),
        )

    def _phase_safety_clamp(self, phase_signal: float) -> None:
        risk = max(0.0, min(1.0, float(phase_signal)))
        if risk <= 1e-6:
            return
        damp = 1.0 - (0.22 * risk)
        self.k_div = min(self._k_div_bounds[1], max(self._k_div_bounds[0], self.k_div * damp))
        self.residual_damping = min(
            self._residual_bounds[1],
            max(self._residual_bounds[0], self.residual_damping * (1.0 - 0.18 * risk)),
        )
        self.k_higher = min(
            self._k_higher_bounds[1],
            max(self._k_higher_bounds[0], self.k_higher * (1.0 - 0.2 * risk)),
        )

    def _effective_clip(self, phase_signal: float) -> float:
        risk = max(0.0, min(1.0, float(phase_signal)))
        return max(0.35, self.control_clip * (1.0 - 0.35 * risk))

    def _adapt_gains(
        self,
        objective: float,
        residual_ratio: float,
        div_before: np.ndarray,
        div_after: np.ndarray,
        control_energy: float,
        saturation_ratio: float,
    ) -> None:
        before = float(np.linalg.norm(div_before))
        after = float(np.linalg.norm(div_after))
        improved = after < before * 0.97

        if self._prev_objective is not None:
            objective_improved = objective < self._prev_objective
            if not objective_improved:
                self._direction_k_div *= -1.0
                self._direction_residual *= -1.0
            self.k_div += self._direction_k_div * self.objective_step_k_div
            self.residual_damping += self._direction_residual * self.objective_step_residual

        if residual_ratio > 0.45 or not improved:
            self.k_div = min(self._k_div_bounds[1], self.k_div * 1.03)
            self.residual_damping = min(self._residual_bounds[1], self.residual_damping * 1.02)
            self.k_higher = min(self._k_higher_bounds[1], self.k_higher * 1.01)
        else:
            self.k_div = max(self._k_div_bounds[0], self.k_div * 0.995)
            self.residual_damping = max(self._residual_bounds[0], self.residual_damping * 0.995)
            self.k_higher = max(self._k_higher_bounds[0], self.k_higher * 0.997)

        if control_energy > 2.5 or saturation_ratio > 0.35:
            self.k_div = max(self._k_div_bounds[0], self.k_div * 0.97)
            self.residual_damping = max(self._residual_bounds[0], self.residual_damping * 0.985)
            self.k_higher = max(self._k_higher_bounds[0], self.k_higher * 0.985)

        self.k_div = min(self._k_div_bounds[1], max(self._k_div_bounds[0], self.k_div))
        self.residual_damping = min(self._residual_bounds[1], max(self._residual_bounds[0], self.residual_damping))
        self.k_higher = min(self._k_higher_bounds[1], max(self._k_higher_bounds[0], self.k_higher))
        self._prev_objective = objective

    def _lookahead_select_gains(
        self,
        B: np.ndarray,
        f: np.ndarray,
        residual: np.ndarray,
        harmonic: np.ndarray,
        div_before: np.ndarray,
        phi: np.ndarray,
        speed_term: float,
        cycle_pressure: np.ndarray,
        higher_pressure: np.ndarray,
        curl_pressure: np.ndarray,
        harmonic_pressure: np.ndarray,
        cycle_pressure_mean: float,
        higher_pressure_mean: float,
        topological_tension: float,
        flow_norm: float,
        curl_norm: float,
        harmonic_norm: float,
    ) -> tuple[float, float, float]:
        k_div_candidates = (
            self.k_div - self.objective_step_k_div,
            self.k_div,
            self.k_div + self.objective_step_k_div,
        )
        residual_candidates = (
            self.residual_damping - self.objective_step_residual,
            self.residual_damping,
            self.residual_damping + self.objective_step_residual,
        )
        k_higher_candidates = (
            self.k_higher - 0.01,
            self.k_higher,
            self.k_higher + 0.01,
        )

        best = (self.k_div, self.residual_damping, self.k_higher)
        best_obj = float("inf")
        residual_ratio = float(np.linalg.norm(residual) / flow_norm)
        curl_ratio = curl_norm / flow_norm
        harmonic_ratio = harmonic_norm / flow_norm

        for kd in k_div_candidates:
            kd = min(self._k_div_bounds[1], max(self._k_div_bounds[0], kd))
            for rd in residual_candidates:
                rd = min(self._residual_bounds[1], max(self._residual_bounds[0], rd))
                for kh in k_higher_candidates:
                    kh = min(self._k_higher_bounds[1], max(self._k_higher_bounds[0], kh))
                    obj = self._multi_step_objective(
                        B=B,
                        f=f,
                        residual=residual,
                        harmonic=harmonic,
                        div_before=div_before,
                        phi=phi,
                        speed_term=speed_term,
                        cycle_pressure=cycle_pressure,
                        higher_pressure=higher_pressure,
                        curl_pressure=curl_pressure,
                        harmonic_pressure=harmonic_pressure,
                        cycle_pressure_mean=cycle_pressure_mean,
                        higher_pressure_mean=higher_pressure_mean,
                        topological_tension=topological_tension,
                        residual_ratio=residual_ratio,
                        curl_ratio=curl_ratio,
                        harmonic_ratio=harmonic_ratio,
                        kd=kd,
                        rd=rd,
                        kh=kh,
                    )
                    if obj < best_obj:
                        best_obj = obj
                        best = (kd, rd, kh)
        return best

    def _multi_step_objective(
        self,
        B: np.ndarray,
        f: np.ndarray,
        residual: np.ndarray,
        harmonic: np.ndarray,
        div_before: np.ndarray,
        phi: np.ndarray,
        speed_term: float,
        cycle_pressure: np.ndarray,
        higher_pressure: np.ndarray,
        curl_pressure: np.ndarray,
        harmonic_pressure: np.ndarray,
        cycle_pressure_mean: float,
        higher_pressure_mean: float,
        topological_tension: float,
        residual_ratio: float,
        curl_ratio: float,
        harmonic_ratio: float,
        kd: float,
        rd: float,
        kh: float,
    ) -> float:
        total = 0.0
        discount = 1.0
        f_curr = f.copy()
        residual_curr = residual.copy()
        div_before_curr = div_before.copy()
        speed_curr = speed_term
        for _ in range(self.lookahead_horizon):
            f_ctrl = f_curr - rd * residual_curr - self.k_harmonic * harmonic
            div_after = B @ f_ctrl
            u = (
                -kd * div_before_curr
                - self.k_phi * phi
                - self.k_speed * speed_curr
                - self.k_cycle * cycle_pressure
                - kh * higher_pressure
                - self.k_curl * curl_pressure
                - self.k_harmonic * harmonic_pressure
            )
            u = np.clip(u, -self.control_clip, self.control_clip)
            sat = float(np.mean(np.abs(u) >= (0.98 * self.control_clip)))
            energy = float(np.sum(u * u))
            obj = self._objective(
                div_after=float(np.linalg.norm(div_after)),
                residual_ratio=residual_ratio,
                control_energy=energy,
                saturation_ratio=sat,
                cycle_pressure=cycle_pressure_mean,
                curl_ratio=curl_ratio,
                harmonic_ratio=harmonic_ratio,
                higher_pressure=higher_pressure_mean,
                topological_tension=topological_tension,
            )
            total += discount * obj

            # Coarse rollout dynamics for next lookahead step.
            f_curr = f_ctrl
            residual_curr = residual_curr * rd
            div_before_curr = div_after
            speed_curr *= self.lookahead_decay
            discount *= self.lookahead_decay
        return total

    def _objective(
        self,
        div_after: float,
        residual_ratio: float,
        control_energy: float,
        saturation_ratio: float,
        cycle_pressure: float,
        curl_ratio: float,
        harmonic_ratio: float,
        higher_pressure: float,
        topological_tension: float,
    ) -> float:
        w_div, w_res, w_energy, w_sat, w_cycle = self.objective_weights
        return (
            w_div * div_after
            + w_res * residual_ratio
            + w_energy * control_energy
            + w_sat * saturation_ratio
            + w_cycle * cycle_pressure
            + 0.45 * curl_ratio
            + 0.35 * harmonic_ratio
            + 0.25 * higher_pressure
            + 0.15 * topological_tension
        )

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

    def _cycle_matrix(
        self,
        nodes: list[str],
        edges: list[tuple[str, str, float]],
        node_index: dict[str, int],
    ) -> np.ndarray:
        if not edges:
            return np.zeros((0, 0), dtype=np.float64)
        undirected_edges = [(src, dst) for src, dst, _ in edges]
        edge_idx = {tuple(sorted((src, dst))): i for i, (src, dst) in enumerate(undirected_edges)}
        adjacency: dict[str, set[str]] = {n: set() for n in nodes}
        for src, dst in undirected_edges:
            adjacency.setdefault(src, set()).add(dst)
            adjacency.setdefault(dst, set()).add(src)

        cycles: list[list[str]] = []
        visited: set[str] = set()
        for root in nodes:
            if root in visited:
                continue
            parent = {root: None}
            stack = [root]
            while stack:
                u = stack.pop()
                visited.add(u)
                for v in adjacency.get(u, set()):
                    if parent.get(u) == v:
                        continue
                    if v not in parent:
                        parent[v] = u
                        stack.append(v)
                    else:
                        cycle = self._extract_cycle(u, v, parent)
                        if len(cycle) >= 3:
                            cycles.append(cycle)

        if not cycles:
            return np.zeros((0, len(edges)), dtype=np.float64)

        C = np.zeros((len(cycles), len(edges)), dtype=np.float64)
        for i, cycle in enumerate(cycles):
            for j in range(len(cycle)):
                a = cycle[j]
                b = cycle[(j + 1) % len(cycle)]
                key = tuple(sorted((a, b)))
                eidx = edge_idx.get(key)
                if eidx is None:
                    continue
                src, dst, _ = edges[eidx]
                C[i, eidx] = 1.0 if (a == src and b == dst) else -1.0
        return C

    def _extract_cycle(self, u: str, v: str, parent: dict[str, str | None]) -> list[str]:
        path_u = []
        x = u
        while x is not None:
            path_u.append(x)
            x = parent.get(x)
        path_v = []
        y = v
        while y is not None:
            path_v.append(y)
            y = parent.get(y)
        set_u = set(path_u)
        lca = next((node for node in path_v if node in set_u), None)
        if lca is None:
            return []
        cycle = []
        x = u
        while x != lca and x is not None:
            cycle.append(x)
            x = parent.get(x)
        cycle.append(lca)
        rev = []
        y = v
        while y != lca and y is not None:
            rev.append(y)
            y = parent.get(y)
        cycle.extend(reversed(rev))
        if len(cycle) != len(set(cycle)):
            return []
        return cycle

    def _solve_curl_component(self, C: np.ndarray, residual: np.ndarray) -> np.ndarray:
        if C.size == 0:
            return np.zeros_like(residual)
        M = C @ C.T
        b = C @ residual
        eps = 1e-6
        psi = np.linalg.solve(M + eps * np.eye(M.shape[0]), b)
        return C.T @ psi

    def _curl_pressure(
        self,
        C: np.ndarray,
        curl_flow: np.ndarray,
        num_nodes: int,
        edges: list[tuple[str, str, float]],
        node_index: dict[str, int],
    ) -> np.ndarray:
        pressure = np.zeros((num_nodes,), dtype=np.float64)
        if C.size == 0 or curl_flow.size == 0:
            return pressure
        edge_abs = np.abs(curl_flow)
        for j, (src, dst, _) in enumerate(edges):
            val = edge_abs[j]
            pressure[node_index[src]] += val
            pressure[node_index[dst]] += val
        return pressure / max(1.0, np.max(pressure))

    def _harmonic_pressure(
        self,
        harmonic: np.ndarray,
        num_nodes: int,
        edges: list[tuple[str, str, float]],
        node_index: dict[str, int],
    ) -> np.ndarray:
        pressure = np.zeros((num_nodes,), dtype=np.float64)
        if harmonic.size == 0:
            return pressure
        edge_abs = np.abs(harmonic)
        for j, (src, dst, _) in enumerate(edges):
            val = edge_abs[j]
            pressure[node_index[src]] += val
            pressure[node_index[dst]] += val
        return pressure / max(1.0, np.max(pressure))
