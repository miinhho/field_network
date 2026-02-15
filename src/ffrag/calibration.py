from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import random

from .flow import AdjustmentPlannerConfig
from .models import Actant, Interaction, LayeredGraph, Perturbation, Query
from .pipeline import FlowGraphRAG


@dataclass(slots=True)
class CalibrationRow:
    config_id: str
    score: float
    avg_adjustment_objective: float
    avg_critical_transition: float
    avg_coupling_penalty: float
    avg_applied_edits: float
    avg_converged: float


def candidate_configs() -> list[tuple[str, AdjustmentPlannerConfig]]:
    return [
        (
            "base",
            AdjustmentPlannerConfig(),
        ),
        (
            "conservative",
            AdjustmentPlannerConfig(
                default_scale_candidates=(0.4, 0.6, 0.8, 1.0),
                risk_weight_base=0.35,
                risk_weight_gain=0.28,
            ),
        ),
        (
            "explore_sparse",
            AdjustmentPlannerConfig(
                sparse_scale_candidates=(0.9, 1.15, 1.35, 1.6),
                rewiring_weight_base=0.02,
                rewiring_weight_density_gain=0.05,
            ),
        ),
        (
            "noise_robust",
            AdjustmentPlannerConfig(
                volatility_weight_base=0.4,
                volatility_weight_noise_gain=0.34,
                high_risk_threshold=0.7,
            ),
        ),
        (
            "risk_averse",
            AdjustmentPlannerConfig(
                high_risk_threshold=0.65,
                high_risk_scale_candidates=(0.35, 0.5, 0.65, 0.8),
                under_adjust_penalty=0.7,
                over_adjust_penalty=0.85,
            ),
        ),
    ]


def run_calibration(num_scenarios: int = 20, seed: int = 42) -> list[CalibrationRow]:
    rng = random.Random(seed)
    scenarios = [_scenario(rng, i) for i in range(max(1, num_scenarios))]
    rows: list[CalibrationRow] = []

    for config_id, cfg in candidate_configs():
        rag = FlowGraphRAG(adjustment_planner_config=cfg)
        sum_obj = 0.0
        sum_critical = 0.0
        sum_coupling = 0.0
        sum_edits = 0.0
        sum_conv = 0.0

        for graph, perturbation in scenarios:
            out = rag.run(graph, Query(text="predict calibration scenario"), perturbation=perturbation)
            m = out.metrics_used
            sum_obj += float(m.get("adjustment_objective_score", 0.0))
            sum_critical += float(m.get("critical_transition_score", 0.0))
            sum_coupling += float(m.get("coupling_penalty", 0.0))
            sum_edits += float(m.get("applied_new_edges", 0.0) + m.get("applied_drop_edges", 0.0))
            sum_conv += float(m.get("converged", 0.0))

        n = float(len(scenarios))
        avg_obj = sum_obj / n
        avg_critical = sum_critical / n
        avg_coupling = sum_coupling / n
        avg_edits = sum_edits / n
        avg_conv = sum_conv / n

        # Lower is better; reward convergence modestly.
        score = avg_obj + 0.55 * avg_critical + 0.3 * avg_coupling + 0.05 * avg_edits - 0.25 * avg_conv
        rows.append(
            CalibrationRow(
                config_id=config_id,
                score=round(score, 6),
                avg_adjustment_objective=round(avg_obj, 6),
                avg_critical_transition=round(avg_critical, 6),
                avg_coupling_penalty=round(avg_coupling, 6),
                avg_applied_edits=round(avg_edits, 6),
                avg_converged=round(avg_conv, 6),
            )
        )

    rows.sort(key=lambda r: r.score)
    return rows


def _scenario(rng: random.Random, idx: int) -> tuple[LayeredGraph, Perturbation]:
    node_count = rng.randint(6, 10)
    edge_prob = rng.uniform(0.2, 0.7)
    base = datetime(2026, 2, 1, 9, 0, 0) + timedelta(hours=idx)

    g = LayeredGraph(graph_id=f"calib-g{idx}", schema_version="0.1")
    nodes = [f"n{i}" for i in range(node_count)]
    for n in nodes:
        g.actants[n] = Actant(actant_id=n, kind="entity", label=n)

    eidx = 0
    for i in range(node_count):
        for j in range(i + 1, node_count):
            if rng.random() > edge_prob:
                continue
            g.interactions.append(
                Interaction(
                    interaction_id=f"ce{idx}_{eidx}",
                    timestamp=base + timedelta(seconds=eidx),
                    source_id=nodes[i],
                    target_id=nodes[j],
                    layer="calibration",
                    weight=round(rng.uniform(0.4, 1.3), 3),
                )
            )
            eidx += 1

    if not g.interactions and node_count >= 2:
        g.interactions.append(
            Interaction(
                interaction_id=f"ce{idx}_fallback",
                timestamp=base,
                source_id=nodes[0],
                target_id=nodes[1],
                layer="calibration",
                weight=0.6,
            )
        )

    target = rng.choice(nodes)
    perturbation = Perturbation(
        perturbation_id=f"cp{idx}",
        timestamp=base + timedelta(minutes=30),
        targets=[target],
        intensity=round(rng.uniform(0.6, 1.5), 3),
        kind="calibration",
    )
    return g, perturbation
