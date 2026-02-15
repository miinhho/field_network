from datetime import datetime
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag import Actant, FlowGraphRAG, Interaction, LayeredGraph, Perturbation, Query


def _graph() -> LayeredGraph:
    g = LayeredGraph(graph_id="core-ready-g1", schema_version="0.1")
    g.actants = {
        "n1": Actant("n1", "entity", "N1"),
        "n2": Actant("n2", "entity", "N2"),
        "n3": Actant("n3", "entity", "N3"),
        "n4": Actant("n4", "entity", "N4"),
    }
    g.interactions = [
        Interaction("e1", datetime(2026, 2, 1, 9), "n1", "n2", "social", 1.0),
        Interaction("e2", datetime(2026, 2, 1, 10), "n2", "n3", "social", 0.9),
        Interaction("e3", datetime(2026, 2, 1, 11), "n3", "n4", "temporal", 0.8),
        Interaction("e4", datetime(2026, 2, 1, 12), "n1", "n3", "spatial", 0.7),
    ]
    return g


class CoreReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rag = FlowGraphRAG()
        self.graph = _graph()
        self.perturbation = Perturbation(
            perturbation_id="core-ready-p1",
            timestamp=datetime(2026, 2, 2, 9, 0, 0),
            targets=["n1"],
            intensity=1.1,
        )

    def test_predict_exposes_core_diagnostic_metrics(self) -> None:
        out = self.rag.run(self.graph, Query(text="predict core readiness"), perturbation=self.perturbation)
        self.assertEqual(out.query_type, "predict")
        required = {
            "adjustment_objective_score",
            "adjustment_scale",
            "planner_horizon",
            "edit_budget",
            "graph_density",
            "impact_noise",
            "coupling_penalty",
            "applied_new_edges",
            "applied_drop_edges",
            "blocked_drop_edges",
            "adjustment_term_churn",
            "adjustment_term_volatility",
            "adjustment_term_rewiring",
            "adjustment_term_risk",
            "adjustment_term_coupling",
            "critical_transition_score",
            "critical_slowing_score",
            "hysteresis_proxy_score",
        }
        self.assertTrue(required.issubset(set(out.metrics_used.keys())))

    def test_intervene_exposes_baseline_intervention_pairs(self) -> None:
        out = self.rag.run(self.graph, Query(text="intervene core readiness"), perturbation=self.perturbation)
        self.assertEqual(out.query_type, "intervene")
        required = {
            "baseline_adjustment_scale",
            "intervention_adjustment_scale",
            "baseline_planner_horizon",
            "intervention_planner_horizon",
            "baseline_edit_budget",
            "intervention_edit_budget",
            "baseline_applied_drop_edges",
            "intervention_applied_drop_edges",
            "baseline_blocked_drop_edges",
            "intervention_blocked_drop_edges",
            "baseline_critical_slowing_score",
            "intervention_critical_slowing_score",
            "baseline_hysteresis_proxy_score",
            "intervention_hysteresis_proxy_score",
        }
        self.assertTrue(required.issubset(set(out.metrics_used.keys())))

    def test_core_metric_ranges_are_bounded(self) -> None:
        out = self.rag.run(self.graph, Query(text="predict bounded metrics"), perturbation=self.perturbation)
        m = out.metrics_used
        for key in ["graph_density", "impact_noise", "coupling_penalty", "critical_slowing_score", "hysteresis_proxy_score"]:
            self.assertGreaterEqual(m[key], 0.0)
            self.assertLessEqual(m[key], 1.0)
        self.assertGreaterEqual(m["planner_horizon"], 2.0)
        self.assertLessEqual(m["planner_horizon"], 5.0)
        self.assertGreaterEqual(m["edit_budget"], 0.0)
        self.assertLessEqual(m["edit_budget"], 2.0)


if __name__ == "__main__":
    unittest.main()
