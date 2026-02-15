from datetime import datetime, timedelta
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag import Actant, FlowGraphRAG, Interaction, LayeredGraph, Perturbation, Query


class ExtremeScenarioIntegrationTests(unittest.TestCase):
    def _build_sparse_graph(self) -> LayeredGraph:
        g = LayeredGraph(graph_id="extreme-sparse", schema_version="0.1")
        for i in range(8):
            nid = f"s{i}"
            g.actants[nid] = Actant(actant_id=nid, kind="entity", label=nid)
        base = datetime(2026, 2, 12, 9, 0, 0)
        edges = [("s0", "s1"), ("s2", "s3"), ("s4", "s5")]
        for i, (a, b) in enumerate(edges):
            g.interactions.append(
                Interaction(
                    interaction_id=f"se{i}",
                    timestamp=base + timedelta(minutes=i),
                    source_id=a,
                    target_id=b,
                    layer="sparse",
                    weight=0.8,
                )
            )
        return g

    def _build_dense_graph(self) -> LayeredGraph:
        g = LayeredGraph(graph_id="extreme-dense", schema_version="0.1")
        nodes = [f"d{i}" for i in range(7)]
        for nid in nodes:
            g.actants[nid] = Actant(actant_id=nid, kind="entity", label=nid)
        base = datetime(2026, 2, 12, 10, 0, 0)
        idx = 0
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                g.interactions.append(
                    Interaction(
                        interaction_id=f"de{idx}",
                        timestamp=base + timedelta(seconds=idx),
                        source_id=nodes[i],
                        target_id=nodes[j],
                        layer="dense",
                        weight=1.0,
                    )
                )
                idx += 1
        return g

    def test_sparse_graph_predict_stays_bounded(self) -> None:
        rag = FlowGraphRAG()
        g = self._build_sparse_graph()
        p = Perturbation(
            perturbation_id="ext-sparse-p",
            timestamp=datetime(2026, 2, 12, 12, 0, 0),
            targets=["s0"],
            intensity=1.0,
        )
        out = rag.run(g, Query(text="predict sparse transition"), perturbation=p)
        self.assertEqual(out.query_type, "predict")
        self.assertGreaterEqual(out.metrics_used["graph_density"], 0.0)
        self.assertLessEqual(out.metrics_used["graph_density"], 1.0)
        self.assertGreaterEqual(out.metrics_used["impact_noise"], 0.0)
        self.assertLessEqual(out.metrics_used["impact_noise"], 1.0)
        self.assertGreaterEqual(out.metrics_used["planner_horizon"], 2.0)
        self.assertLessEqual(out.metrics_used["planner_horizon"], 5.0)

    def test_dense_graph_intervention_stays_bounded(self) -> None:
        rag = FlowGraphRAG()
        g = self._build_dense_graph()
        p = Perturbation(
            perturbation_id="ext-dense-p",
            timestamp=datetime(2026, 2, 12, 13, 0, 0),
            targets=["d0"],
            intensity=1.3,
        )
        out = rag.run(g, Query(text="intervene dense system"), perturbation=p)
        self.assertEqual(out.query_type, "intervene")
        self.assertGreaterEqual(out.metrics_used["baseline_graph_density"], 0.0)
        self.assertLessEqual(out.metrics_used["baseline_graph_density"], 1.0)
        self.assertGreaterEqual(out.metrics_used["baseline_edit_budget"], 0.0)
        self.assertLessEqual(out.metrics_used["baseline_edit_budget"], 2.0)
        self.assertGreaterEqual(out.metrics_used["baseline_planner_horizon"], 2.0)
        self.assertLessEqual(out.metrics_used["baseline_planner_horizon"], 5.0)


if __name__ == "__main__":
    unittest.main()
