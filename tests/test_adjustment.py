from datetime import datetime
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag import Actant, Interaction, LayeredGraph
from ffrag.flow import DynamicGraphAdjuster


class AdjustmentTests(unittest.TestCase):
    def _graph(self) -> LayeredGraph:
        g = LayeredGraph(graph_id="ga1", schema_version="0.1")
        g.actants = {
            "a": Actant("a", "person", "A"),
            "b": Actant("b", "person", "B"),
            "c": Actant("c", "person", "C"),
        }
        g.interactions = [
            Interaction("e1", datetime(2026, 2, 1, 9), "a", "b", "social", 1.0),
            Interaction("e2", datetime(2026, 2, 1, 10), "b", "c", "social", 0.8),
        ]
        return g

    def test_adjust_returns_weighted_graph(self) -> None:
        adjuster = DynamicGraphAdjuster()
        out = adjuster.adjust(
            self._graph(),
            impact_by_actant={"a": 1.0, "b": 0.8, "c": 0.2},
            state={"transition_speed": 0.7, "temporal_regularity": 0.3, "schedule_density": 4.0},
        )
        self.assertEqual(len(out.adjusted_graph.interactions), 2)
        self.assertGreaterEqual(out.strengthened_edges + out.weakened_edges + out.unchanged_edges, 2)
        self.assertGreaterEqual(len(out.suggested_drop_edges), 0)
        self.assertGreaterEqual(len(out.suggested_new_edges), 0)


if __name__ == "__main__":
    unittest.main()
