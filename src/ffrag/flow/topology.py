from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from ..models import LayeredGraph


@dataclass(slots=True)
class SimplicialTopologyResult:
    node_pressure: dict[str, float]
    triangle_count: int
    tetra_count: int
    simplex_density: float
    topological_tension: float


class SimplicialTopologyModel:
    """Lightweight simplicial approximation over graph cliques.

    - 2-simplex: triangle clique (size=3)
    - 3-simplex: tetra clique (size=4)
    """

    def __init__(self, triangle_weight: float = 0.6, tetra_weight: float = 1.0) -> None:
        self.triangle_weight = triangle_weight
        self.tetra_weight = tetra_weight

    def compute(self, graph: LayeredGraph, impact_by_actant: dict[str, float]) -> SimplicialTopologyResult:
        g = nx.Graph()
        for node in graph.actants.keys():
            g.add_node(node)
        for edge in graph.interactions:
            if edge.source_id == edge.target_id:
                continue
            g.add_edge(edge.source_id, edge.target_id)

        if g.number_of_nodes() == 0:
            return SimplicialTopologyResult({}, 0, 0, 0.0, 0.0)

        triangles: list[tuple[str, str, str]] = []
        tetras: list[tuple[str, str, str, str]] = []
        for clique in nx.enumerate_all_cliques(g):
            if len(clique) == 3:
                triangles.append((clique[0], clique[1], clique[2]))
            elif len(clique) == 4:
                tetras.append((clique[0], clique[1], clique[2], clique[3]))
            elif len(clique) > 4:
                break

        pressure = {node: 0.0 for node in g.nodes()}
        participation = {node: 0.0 for node in g.nodes()}

        for tri in triangles:
            avg_impact = sum(float(impact_by_actant.get(n, 0.0)) for n in tri) / 3.0
            for n in tri:
                base = float(impact_by_actant.get(n, 0.0))
                contrib = self.triangle_weight * (0.5 * base + 0.5 * avg_impact)
                pressure[n] += contrib
                participation[n] += 1.0

        for tet in tetras:
            avg_impact = sum(float(impact_by_actant.get(n, 0.0)) for n in tet) / 4.0
            for n in tet:
                base = float(impact_by_actant.get(n, 0.0))
                contrib = self.tetra_weight * (0.5 * base + 0.5 * avg_impact)
                pressure[n] += contrib
                participation[n] += 1.5

        max_p = max(pressure.values()) if pressure else 0.0
        if max_p > 1e-9:
            pressure = {k: v / max_p for k, v in pressure.items()}

        simplex_density = (len(triangles) + 2.0 * len(tetras)) / max(1, g.number_of_nodes())
        vals = list(participation.values())
        if vals:
            mean_v = sum(vals) / len(vals)
            topological_tension = sum((v - mean_v) ** 2 for v in vals) / len(vals)
        else:
            topological_tension = 0.0

        return SimplicialTopologyResult(
            node_pressure=pressure,
            triangle_count=len(triangles),
            tetra_count=len(tetras),
            simplex_density=float(simplex_density),
            topological_tension=float(topological_tension),
        )
