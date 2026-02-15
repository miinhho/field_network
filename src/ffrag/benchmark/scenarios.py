from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import random

from ..models import Actant, Interaction, LayeredGraph, Perturbation


@dataclass(slots=True)
class PredictScenario:
    scenario_id: str
    graph: LayeredGraph
    perturbation: Perturbation
    expected_impacted: set[str]


def _truth_impacted(
    graph: LayeredGraph,
    seeds: list[str],
    max_hops: int = 3,
    attenuation: float = 0.6,
    top_k: int = 3,
) -> set[str]:
    adjacency: dict[str, set[str]] = {}
    for edge in graph.interactions:
        adjacency.setdefault(edge.source_id, set()).add(edge.target_id)
        adjacency.setdefault(edge.target_id, set()).add(edge.source_id)

    score: dict[str, float] = {seed: 1.0 for seed in seeds}
    frontier = {seed: 1.0 for seed in seeds}
    for _ in range(max_hops):
        nxt: dict[str, float] = {}
        for src, strength in frontier.items():
            next_strength = strength * attenuation
            for dst in adjacency.get(src, set()):
                score[dst] = score.get(dst, 0.0) + next_strength
                prev = nxt.get(dst, 0.0)
                if next_strength > prev:
                    nxt[dst] = next_strength
        frontier = {node: value for node, value in nxt.items() if value > 0.01}
        if not frontier:
            break
    ranked = sorted(score.items(), key=lambda item: item[1], reverse=True)
    return {node for node, _ in ranked[:top_k]}


def generate_scenarios(num_scenarios: int = 10, seed: int = 42) -> list[PredictScenario]:
    rng = random.Random(seed)
    scenarios: list[PredictScenario] = []

    for i in range(num_scenarios):
        graph = LayeredGraph(graph_id=f"g-{i}", schema_version="0.1")
        node_count = 8
        base = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)

        for j in range(node_count):
            node_id = f"n{j}"
            graph.actants[node_id] = Actant(actant_id=node_id, kind="entity", label=node_id)

        interaction_id = 0
        for j in range(node_count):
            src = f"n{j}"
            dst = f"n{(j + 1) % node_count}"
            graph.interactions.append(
                Interaction(
                    interaction_id=f"e{i}-{interaction_id}",
                    timestamp=base + timedelta(minutes=interaction_id),
                    source_id=src,
                    target_id=dst,
                    layer="social" if j % 2 == 0 else "temporal",
                    weight=1.0,
                )
            )
            interaction_id += 1

        for _ in range(4):
            src_idx = rng.randint(0, node_count - 1)
            dst_idx = rng.randint(0, node_count - 1)
            if src_idx == dst_idx:
                continue
            graph.interactions.append(
                Interaction(
                    interaction_id=f"e{i}-{interaction_id}",
                    timestamp=base + timedelta(minutes=interaction_id),
                    source_id=f"n{src_idx}",
                    target_id=f"n{dst_idx}",
                    layer="spatial",
                    weight=1.0,
                )
            )
            interaction_id += 1

        target = f"n{rng.randint(0, node_count - 1)}"
        perturbation = Perturbation(
            perturbation_id=f"p-{i}",
            timestamp=base,
            targets=[target],
            intensity=1.0,
            kind="synthetic",
        )
        expected = _truth_impacted(graph, perturbation.targets, max_hops=3, attenuation=0.6, top_k=3)
        scenarios.append(
            PredictScenario(
                scenario_id=f"s-{i}",
                graph=graph,
                perturbation=perturbation,
                expected_impacted=expected,
            )
        )

    return scenarios
