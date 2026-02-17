from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import random

from .flow import AdjustmentPlannerConfig, PlasticityConfig, DynamicGraphAdjuster
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
    avg_supervisory_confusion: float
    avg_supervisory_forgetting: float
    avg_longrun_churn: float
    avg_longrun_retention: float
    avg_longrun_diversity: float
    avg_cluster_ann_cache_hit_rate: float
    avg_cluster_active_contexts: float
    avg_cluster_evicted_contexts: float


@dataclass(slots=True)
class CalibrationSummary:
    batch: str
    candidate_count: int
    top_count: int
    eta_up_min: float
    eta_up_max: float
    eta_down_min: float
    eta_down_max: float
    theta_on_min: float
    theta_on_max: float
    theta_off_min: float
    theta_off_max: float
    hysteresis_dwell_min: int
    hysteresis_dwell_max: int
    risk_weight_base_min: float
    risk_weight_base_max: float
    volatility_weight_base_min: float
    volatility_weight_base_max: float


@dataclass(slots=True)
class CalibrationCandidate:
    config_id: str
    planner_config: AdjustmentPlannerConfig
    plasticity_config: PlasticityConfig


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


def candidate_profiles(batch: str = "default") -> list[CalibrationCandidate]:
    base_planners = candidate_configs()
    plasticity_variants = [
        ("plastic_base", PlasticityConfig()),
        (
            "plastic_fast",
            PlasticityConfig(
                eta_up=0.34,
                eta_down=0.08,
                theta_on=0.58,
                theta_off=0.36,
                hysteresis_dwell=2,
            ),
        ),
        (
            "plastic_guarded",
            PlasticityConfig(
                eta_up=0.20,
                eta_down=0.16,
                theta_on=0.72,
                theta_off=0.28,
                hysteresis_dwell=3,
            ),
        ),
    ]

    mode = batch.lower().strip()
    if mode not in ("default", "plasticity", "mixed"):
        mode = "default"

    out: list[CalibrationCandidate] = []
    if mode == "default":
        for pid, p in base_planners:
            out.append(CalibrationCandidate(config_id=pid, planner_config=p, plasticity_config=PlasticityConfig()))
        return out
    if mode == "plasticity":
        planner = AdjustmentPlannerConfig()
        for vid, pvc in plasticity_variants:
            out.append(CalibrationCandidate(config_id=vid, planner_config=planner, plasticity_config=pvc))
        return out

    # mixed mode: planner x plasticity grid.
    for pid, p in base_planners:
        for vid, pvc in plasticity_variants:
            out.append(CalibrationCandidate(config_id=f"{pid}__{vid}", planner_config=p, plasticity_config=pvc))
    return out


def run_calibration(num_scenarios: int = 20, seed: int = 42, batch: str = "default") -> list[CalibrationRow]:
    rows, _ = run_calibration_with_summary(num_scenarios=num_scenarios, seed=seed, batch=batch)
    return rows


def run_calibration_with_summary(
    num_scenarios: int = 20,
    seed: int = 42,
    batch: str = "default",
    top_fraction: float = 0.4,
) -> tuple[list[CalibrationRow], CalibrationSummary]:
    rng = random.Random(seed)
    scenarios = [_scenario(rng, i) for i in range(max(1, num_scenarios))]
    rows: list[CalibrationRow] = []
    candidates = candidate_profiles(batch=batch)

    for cand in candidates:
        rag = FlowGraphRAG(adjustment_planner_config=cand.planner_config)
        rag.adjuster = DynamicGraphAdjuster(
            planner_config=cand.planner_config,
            plasticity_config=cand.plasticity_config,
        )
        sum_obj = 0.0
        sum_critical = 0.0
        sum_coupling = 0.0
        sum_edits = 0.0
        sum_conv = 0.0
        sum_confusion = 0.0
        sum_forgetting = 0.0
        sum_longrun_churn = 0.0
        sum_longrun_retention = 0.0
        sum_longrun_diversity = 0.0
        sum_cluster_ann_cache_hit_rate = 0.0
        sum_cluster_active_contexts = 0.0
        sum_cluster_evicted_contexts = 0.0

        for graph, perturbation in scenarios:
            out = rag.run(graph, Query(text="predict calibration scenario"), perturbation=perturbation)
            m = out.metrics_used
            sum_obj += float(m.get("adjustment_objective_score", 0.0))
            sum_critical += float(m.get("critical_transition_score", 0.0))
            sum_coupling += float(m.get("coupling_penalty", 0.0))
            sum_edits += float(m.get("applied_new_edges", 0.0) + m.get("applied_drop_edges", 0.0))
            sum_conv += float(m.get("converged", 0.0))
            sum_confusion += float(m.get("supervisory_confusion_score", 0.0))
            sum_forgetting += float(m.get("supervisory_forgetting_score", 0.0))
            sum_cluster_ann_cache_hit_rate += float(m.get("cluster_ann_cache_hit_rate", 0.0))
            sum_cluster_active_contexts += float(m.get("cluster_active_contexts", 0.0))
            sum_cluster_evicted_contexts += float(m.get("cluster_evicted_contexts", 0.0))
            probe = _longrun_probe(rag=rag, graph=graph, perturbation=perturbation, steps=5)
            sum_longrun_churn += probe["avg_longrun_churn"]
            sum_longrun_retention += probe["avg_longrun_retention"]
            sum_longrun_diversity += probe["avg_longrun_diversity"]

        n = float(len(scenarios))
        avg_obj = sum_obj / n
        avg_critical = sum_critical / n
        avg_coupling = sum_coupling / n
        avg_edits = sum_edits / n
        avg_conv = sum_conv / n
        avg_confusion = sum_confusion / n
        avg_forgetting = sum_forgetting / n
        avg_longrun_churn = sum_longrun_churn / n
        avg_longrun_retention = sum_longrun_retention / n
        avg_longrun_diversity = sum_longrun_diversity / n
        avg_cluster_ann_cache_hit_rate = sum_cluster_ann_cache_hit_rate / n
        avg_cluster_active_contexts = sum_cluster_active_contexts / n
        avg_cluster_evicted_contexts = sum_cluster_evicted_contexts / n

        # Lower is better; reward convergence modestly.
        score = (
            avg_obj
            + 0.55 * avg_critical
            + 0.3 * avg_coupling
            + 0.05 * avg_edits
            + 0.15 * avg_confusion
            + 0.12 * avg_forgetting
            + 0.10 * avg_longrun_churn
            - 0.18 * avg_longrun_retention
            - 0.15 * avg_longrun_diversity
            - 0.25 * avg_conv
        )
        rows.append(
            CalibrationRow(
                config_id=cand.config_id,
                score=round(score, 6),
                avg_adjustment_objective=round(avg_obj, 6),
                avg_critical_transition=round(avg_critical, 6),
                avg_coupling_penalty=round(avg_coupling, 6),
                avg_applied_edits=round(avg_edits, 6),
                avg_converged=round(avg_conv, 6),
                avg_supervisory_confusion=round(avg_confusion, 6),
                avg_supervisory_forgetting=round(avg_forgetting, 6),
                avg_longrun_churn=round(avg_longrun_churn, 6),
                avg_longrun_retention=round(avg_longrun_retention, 6),
                avg_longrun_diversity=round(avg_longrun_diversity, 6),
                avg_cluster_ann_cache_hit_rate=round(avg_cluster_ann_cache_hit_rate, 6),
                avg_cluster_active_contexts=round(avg_cluster_active_contexts, 6),
                avg_cluster_evicted_contexts=round(avg_cluster_evicted_contexts, 6),
            )
        )

    rows.sort(key=lambda r: r.score)
    summary = _build_summary(rows=rows, candidates=candidates, batch=batch, top_fraction=top_fraction)
    return rows, summary


def _build_summary(
    rows: list[CalibrationRow],
    candidates: list[CalibrationCandidate],
    batch: str,
    top_fraction: float,
) -> CalibrationSummary:
    by_id = {c.config_id: c for c in candidates}
    if not rows:
        return CalibrationSummary(
            batch=batch,
            candidate_count=0,
            top_count=0,
            eta_up_min=0.0,
            eta_up_max=0.0,
            eta_down_min=0.0,
            eta_down_max=0.0,
            theta_on_min=0.0,
            theta_on_max=0.0,
            theta_off_min=0.0,
            theta_off_max=0.0,
            hysteresis_dwell_min=0,
            hysteresis_dwell_max=0,
            risk_weight_base_min=0.0,
            risk_weight_base_max=0.0,
            volatility_weight_base_min=0.0,
            volatility_weight_base_max=0.0,
        )

    k = max(1, int(round(len(rows) * max(0.1, min(0.9, top_fraction)))))
    top_rows = rows[:k]
    eta_ups: list[float] = []
    eta_downs: list[float] = []
    theta_ons: list[float] = []
    theta_offs: list[float] = []
    dwells: list[int] = []
    risk_ws: list[float] = []
    vol_ws: list[float] = []
    for row in top_rows:
        cand = by_id.get(row.config_id)
        if not cand:
            continue
        p = cand.plasticity_config
        a = cand.planner_config
        eta_ups.append(float(p.eta_up))
        eta_downs.append(float(p.eta_down))
        theta_ons.append(float(p.theta_on))
        theta_offs.append(float(p.theta_off))
        dwells.append(int(p.hysteresis_dwell))
        risk_ws.append(float(a.risk_weight_base))
        vol_ws.append(float(a.volatility_weight_base))

    def _minmax(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        return min(values), max(values)

    eta_up_min, eta_up_max = _minmax(eta_ups)
    eta_down_min, eta_down_max = _minmax(eta_downs)
    theta_on_min, theta_on_max = _minmax(theta_ons)
    theta_off_min, theta_off_max = _minmax(theta_offs)
    risk_min, risk_max = _minmax(risk_ws)
    vol_min, vol_max = _minmax(vol_ws)
    dwell_min = min(dwells) if dwells else 0
    dwell_max = max(dwells) if dwells else 0
    return CalibrationSummary(
        batch=batch,
        candidate_count=len(candidates),
        top_count=len(top_rows),
        eta_up_min=round(eta_up_min, 6),
        eta_up_max=round(eta_up_max, 6),
        eta_down_min=round(eta_down_min, 6),
        eta_down_max=round(eta_down_max, 6),
        theta_on_min=round(theta_on_min, 6),
        theta_on_max=round(theta_on_max, 6),
        theta_off_min=round(theta_off_min, 6),
        theta_off_max=round(theta_off_max, 6),
        hysteresis_dwell_min=dwell_min,
        hysteresis_dwell_max=dwell_max,
        risk_weight_base_min=round(risk_min, 6),
        risk_weight_base_max=round(risk_max, 6),
        volatility_weight_base_min=round(vol_min, 6),
        volatility_weight_base_max=round(vol_max, 6),
    )


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


def _longrun_probe(
    rag: FlowGraphRAG,
    graph: LayeredGraph,
    perturbation: Perturbation,
    steps: int = 5,
) -> dict[str, float]:
    current = graph
    churn_series: list[float] = []
    retention_series: list[float] = []
    diversity_series: list[float] = []
    p = perturbation

    for _ in range(max(1, steps)):
        cycle = rag._run_core_cycle(current, p, cycles=3, shock_kind="base")
        churn = float(cycle.mean_applied_new_edges + cycle.mean_applied_drop_edges)
        retention = max(0.0, min(1.0, 1.0 - float(cycle.mean_supervisory_forgetting)))
        diversity = max(0.0, min(1.0, float(cycle.mean_cross_scale_consistency)))
        churn_series.append(churn)
        retention_series.append(retention)
        diversity_series.append(diversity)
        current = cycle.final_graph

    return {
        "avg_longrun_churn": sum(churn_series) / max(1, len(churn_series)),
        "avg_longrun_retention": sum(retention_series) / max(1, len(retention_series)),
        "avg_longrun_diversity": sum(diversity_series) / max(1, len(diversity_series)),
    }
