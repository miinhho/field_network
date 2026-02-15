from datetime import datetime, timedelta
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag import Actant, FlowGraphRAG, Interaction, LayeredGraph, Perturbation, Query


class EndToEndIntegrationTests(unittest.TestCase):
    def _build_graph(self) -> LayeredGraph:
        g = LayeredGraph(graph_id="integration-g1", schema_version="0.1")
        for node_id, kind in [
            ("u1", "person"),
            ("u2", "person"),
            ("u3", "person"),
            ("loc_a", "place"),
            ("loc_b", "place"),
            ("task_x", "activity"),
        ]:
            g.actants[node_id] = Actant(actant_id=node_id, kind=kind, label=node_id)

        base = datetime(2026, 2, 10, 9, 0, 0)
        edges = [
            ("u1", "u2", "social"),
            ("u2", "u3", "social"),
            ("u1", "loc_a", "spatial"),
            ("u2", "loc_b", "spatial"),
            ("u3", "task_x", "temporal"),
            ("task_x", "loc_a", "temporal"),
            ("u1", "task_x", "temporal"),
            ("u2", "loc_a", "temporal"),
        ]
        for i, (src, dst, layer) in enumerate(edges):
            g.interactions.append(
                Interaction(
                    interaction_id=f"ie-{i}",
                    timestamp=base + timedelta(minutes=i),
                    source_id=src,
                    target_id=dst,
                    layer=layer,
                    weight=1.0,
                )
            )
        return g

    def test_end_to_end_query_cycle(self) -> None:
        rag = FlowGraphRAG()
        graph = self._build_graph()
        perturbation = Perturbation(
            perturbation_id="integration-p1",
            timestamp=datetime(2026, 2, 10, 12, 0, 0),
            targets=["u1"],
            intensity=1.2,
            kind="external_shock",
        )

        describe_answer = rag.run(graph, Query(text="describe current structure"))
        predict_answer = rag.run(graph, Query(text="predict transition after shock"), perturbation=perturbation)
        intervene_answer = rag.run(graph, Query(text="intervene to stabilize"), perturbation=perturbation)

        self.assertEqual(describe_answer.query_type, "describe")
        self.assertFalse(describe_answer.blocked_by_guardrail)
        self.assertGreater(len(describe_answer.evidence_ids), 0)

        self.assertEqual(predict_answer.query_type, "predict")
        self.assertFalse(predict_answer.blocked_by_guardrail)
        self.assertIn("causal_alignment_score", predict_answer.metrics_used)
        self.assertIn("attractor_basin_radius", predict_answer.metrics_used)
        self.assertIn("transition_trigger_count", predict_answer.metrics_used)

        self.assertEqual(intervene_answer.query_type, "intervene")
        self.assertFalse(intervene_answer.blocked_by_guardrail)
        self.assertIn("intervention_causal_alignment_score", intervene_answer.metrics_used)
        self.assertIn("intervention_basin_radius", intervene_answer.metrics_used)
        self.assertIn("intervention_improvement", intervene_answer.metrics_used)

    def test_predict_without_explicit_perturbation_uses_default(self) -> None:
        rag = FlowGraphRAG()
        graph = self._build_graph()

        answer = rag.run(graph, Query(text="predict what happens next"))
        self.assertEqual(answer.query_type, "predict")
        self.assertGreaterEqual(answer.metrics_used["affected_actants"], 1.0)


if __name__ == "__main__":
    unittest.main()
