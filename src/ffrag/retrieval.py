from __future__ import annotations

import networkx as nx

from .models import Interaction, LayeredGraph


class GraphRetriever:
    """Minimal GraphRAG-like evidence retriever over in-memory graph."""

    def retrieve_local(self, graph: LayeredGraph, text: str, limit: int = 8) -> list[Interaction]:
        terms = set(token.lower() for token in text.split())
        if not terms:
            return graph.interactions[:limit]

        scored: list[tuple[int, Interaction]] = []
        for edge in graph.interactions:
            hay = f"{edge.source_id} {edge.target_id} {edge.layer} {edge.metadata}".lower()
            score = sum(1 for term in terms if term in hay)
            if score:
                scored.append((score, edge))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [edge for _, edge in scored[:limit]]

    def retrieve_global(self, graph: LayeredGraph, limit: int = 8) -> list[Interaction]:
        g = nx.Graph()
        for edge in graph.interactions:
            g.add_edge(edge.source_id, edge.target_id, weight=edge.weight)

        degrees = dict(g.degree()) if g.number_of_nodes() > 0 else {}

        scored_edges = sorted(
            graph.interactions,
            key=lambda e: degrees[e.source_id] + degrees[e.target_id] + int(e.weight),
            reverse=True,
        )
        return scored_edges[:limit]
