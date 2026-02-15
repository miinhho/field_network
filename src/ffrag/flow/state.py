from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

import numpy as np

from ..models import LayeredGraph, StateVector


class StateVectorBuilder:
    """Builds simple dynamic state vectors from graph snapshots."""

    FEATURES = (
        "social_entropy",
        "temporal_regularity",
        "spatial_range",
        "schedule_density",
        "network_centrality",
        "transition_speed",
    )

    def build(self, graph: LayeredGraph, entity_id: str, timestamp: datetime | None = None) -> StateVector:
        now = timestamp or datetime.now(timezone.utc)
        edges = [e for e in graph.interactions if e.source_id == entity_id or e.target_id == entity_id]

        if not edges:
            values = {name: 0.0 for name in self.FEATURES}
            return StateVector(entity_id=entity_id, timestamp=now, values=values)

        neighbors = Counter()
        layers = Counter()
        for edge in edges:
            neighbor = edge.target_id if edge.source_id == entity_id else edge.source_id
            neighbors[neighbor] += 1
            layers[edge.layer] += 1

        total = sum(neighbors.values())
        probs = np.array([count / total for count in neighbors.values()], dtype=np.float64)
        entropy = float(-(probs * np.log(probs + 1e-9)).sum())

        spatial_layers = {"spatial", "location", "place"}
        spatial_edges = sum(count for layer, count in layers.items() if layer.lower() in spatial_layers)

        values = {
            "social_entropy": round(entropy, 4),
            "temporal_regularity": round(min(1.0, layers.get("temporal", 0) / len(edges)), 4),
            "spatial_range": float(spatial_edges),
            "schedule_density": float(len(edges)),
            "network_centrality": round(len(neighbors) / max(1, len(graph.actants) - 1), 4),
            "transition_speed": round(len(layers) / max(1, len(edges)), 4),
        }
        return StateVector(entity_id=entity_id, timestamp=now, values=values)
