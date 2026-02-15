from datetime import datetime
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag import Actant, Interaction, LayeredGraph
from ffrag.flow import ClusterFlowController


class MultiscaleTests(unittest.TestCase):
    def _graph(self) -> LayeredGraph:
        g = LayeredGraph(graph_id="ms1", schema_version="0.1")
        for node in ["a", "b", "c", "d", "e"]:
            g.actants[node] = Actant(node, "entity", node)
        edges = [
            ("a", "b"),
            ("b", "c"),
            ("a", "c"),
            ("d", "e"),
            ("c", "d"),
        ]
        for i, (s, t) in enumerate(edges):
            g.interactions.append(
                Interaction(
                    interaction_id=f"e{i}",
                    timestamp=datetime(2026, 2, 1, 9, i),
                    source_id=s,
                    target_id=t,
                    layer="social",
                    weight=1.0,
                )
            )
        return g

    def test_cluster_plan_outputs(self) -> None:
        controller = ClusterFlowController()
        out = controller.plan(
            self._graph(),
            impact_by_actant={"a": 1.0, "b": 0.9, "c": 0.7, "d": 0.2, "e": 0.1},
            state={"transition_speed": 0.6, "temporal_regularity": 0.4, "schedule_density": 3.0},
        )
        self.assertGreaterEqual(len(out.cluster_assignment), 5)
        self.assertGreaterEqual(out.cluster_objective, 0.0)
        self.assertGreaterEqual(out.cross_scale_consistency, 0.0)
        self.assertLessEqual(out.cross_scale_consistency, 1.0)


if __name__ == "__main__":
    unittest.main()
