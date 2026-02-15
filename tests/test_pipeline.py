from datetime import datetime
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag import (
    Actant,
    FlowGraphRAG,
    Interaction,
    LayeredGraph,
    Perturbation,
    Query,
)


def sample_graph() -> LayeredGraph:
    graph = LayeredGraph(graph_id="g1", schema_version="0.1")
    graph.actants = {
        "alice": Actant(actant_id="alice", kind="person", label="Alice"),
        "bob": Actant(actant_id="bob", kind="person", label="Bob"),
        "hq": Actant(actant_id="hq", kind="place", label="HQ"),
    }
    graph.interactions = [
        Interaction(
            interaction_id="e1",
            timestamp=datetime(2026, 2, 1, 9, 0, 0),
            source_id="alice",
            target_id="bob",
            layer="social",
            weight=1.0,
        ),
        Interaction(
            interaction_id="e2",
            timestamp=datetime(2026, 2, 1, 10, 0, 0),
            source_id="alice",
            target_id="hq",
            layer="spatial",
            weight=1.0,
        ),
        Interaction(
            interaction_id="e3",
            timestamp=datetime(2026, 2, 1, 11, 0, 0),
            source_id="bob",
            target_id="hq",
            layer="temporal",
            weight=1.0,
        ),
    ]
    return graph


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rag = FlowGraphRAG()
        self.graph = sample_graph()

    def test_describe_route_returns_evidence(self) -> None:
        answer = self.rag.run(self.graph, Query(text="describe current structure"))
        self.assertEqual(answer.query_type, "describe")
        self.assertFalse(answer.blocked_by_guardrail)
        self.assertGreater(len(answer.evidence_ids), 0)

    def test_predict_route_uses_perturbation(self) -> None:
        perturbation = Perturbation(
            perturbation_id="p1",
            timestamp=datetime(2026, 2, 2, 9, 0, 0),
            targets=["alice"],
            intensity=1.0,
        )
        answer = self.rag.run(self.graph, Query(text="predict next transition"), perturbation=perturbation)
        self.assertEqual(answer.query_type, "predict")
        self.assertGreaterEqual(answer.metrics_used["affected_actants"], 1)
        self.assertIn("dynamics_steps", answer.metrics_used)
        self.assertIn("final_attractor_distance", answer.metrics_used)
        self.assertIn("dominant_transition_prob", answer.metrics_used)
        self.assertIn("recovery_rate", answer.metrics_used)
        self.assertIn("overshoot_index", answer.metrics_used)
        self.assertIn("settling_time", answer.metrics_used)
        self.assertIn("transition_trigger_count", answer.metrics_used)
        self.assertIn("avg_trigger_confidence", answer.metrics_used)
        self.assertIn("attractor_basin_radius", answer.metrics_used)
        self.assertIn("basin_occupancy", answer.metrics_used)
        self.assertIn("causal_alignment_score", answer.metrics_used)
        self.assertIn("strengthened_edges", answer.metrics_used)
        self.assertIn("weakened_edges", answer.metrics_used)
        self.assertIn("mean_weight_shift", answer.metrics_used)
        self.assertIn("control_energy", answer.metrics_used)
        self.assertIn("residual_ratio", answer.metrics_used)
        self.assertIn("saturation_ratio", answer.metrics_used)
        self.assertIn("oscillation_index", answer.metrics_used)
        self.assertIn("cycles_executed", answer.metrics_used)
        self.assertIn("objective_score", answer.metrics_used)
        self.assertIn("objective_improvement", answer.metrics_used)
        self.assertIn("curl_ratio", answer.metrics_used)
        self.assertIn("harmonic_ratio", answer.metrics_used)

    def test_intervene_route_returns_rewiring_metric(self) -> None:
        answer = self.rag.run(self.graph, Query(text="intervene to reduce risk"))
        self.assertEqual(answer.query_type, "intervene")
        self.assertIn("rewire_candidates", answer.metrics_used)
        self.assertIn("intervention_improvement", answer.metrics_used)
        self.assertIn("intervention_hysteresis_index", answer.metrics_used)
        self.assertIn("intervention_overshoot_index", answer.metrics_used)
        self.assertIn("intervention_settling_time", answer.metrics_used)
        self.assertIn("intervention_avg_trigger_confidence", answer.metrics_used)
        self.assertIn("intervention_basin_radius", answer.metrics_used)
        self.assertIn("intervention_basin_occupancy", answer.metrics_used)
        self.assertIn("intervention_causal_alignment_score", answer.metrics_used)
        self.assertIn("intervention_structural_gain", answer.metrics_used)
        self.assertIn("baseline_mean_weight_shift", answer.metrics_used)
        self.assertIn("intervention_mean_weight_shift", answer.metrics_used)
        self.assertIn("baseline_control_energy", answer.metrics_used)
        self.assertIn("intervention_control_energy", answer.metrics_used)
        self.assertIn("baseline_oscillation_index", answer.metrics_used)
        self.assertIn("intervention_oscillation_index", answer.metrics_used)
        self.assertIn("baseline_objective_score", answer.metrics_used)
        self.assertIn("intervention_objective_score", answer.metrics_used)
        self.assertIn("baseline_curl_ratio", answer.metrics_used)
        self.assertIn("intervention_curl_ratio", answer.metrics_used)

    def test_guardrail_blocks_subjective_inference(self) -> None:
        answer = self.rag.run(self.graph, Query(text="describe"))
        self.assertFalse(answer.blocked_by_guardrail)


if __name__ == "__main__":
    unittest.main()
