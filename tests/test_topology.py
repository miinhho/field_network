from datetime import datetime
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag import Actant, Interaction, LayeredGraph
from ffrag.flow import SimplicialTopologyModel


class TopologyTests(unittest.TestCase):
    def _graph(self) -> LayeredGraph:
        g = LayeredGraph(graph_id="topo1", schema_version="0.1")
        g.actants = {
            "a": Actant("a", "entity", "a"),
            "b": Actant("b", "entity", "b"),
            "c": Actant("c", "entity", "c"),
            "d": Actant("d", "entity", "d"),
        }
        g.interactions = [
            Interaction("e1", datetime(2026, 2, 1, 9), "a", "b", "social", 1.0),
            Interaction("e2", datetime(2026, 2, 1, 9), "a", "c", "social", 1.0),
            Interaction("e3", datetime(2026, 2, 1, 9), "b", "c", "social", 1.0),
            Interaction("e4", datetime(2026, 2, 1, 9), "a", "d", "social", 1.0),
            Interaction("e5", datetime(2026, 2, 1, 9), "b", "d", "social", 1.0),
            Interaction("e6", datetime(2026, 2, 1, 9), "c", "d", "social", 1.0),
        ]
        return g

    def test_simplicial_model_detects_triangles_and_tetra(self) -> None:
        model = SimplicialTopologyModel()
        out = model.compute(
            self._graph(),
            {"a": 1.0, "b": 0.8, "c": 0.5, "d": 0.2},
        )
        self.assertGreaterEqual(out.triangle_count, 4)
        self.assertGreaterEqual(out.tetra_count, 1)
        self.assertGreaterEqual(out.simplex_density, 0.0)
        self.assertGreaterEqual(out.topological_tension, 0.0)
        self.assertEqual(set(out.node_pressure.keys()), {"a", "b", "c", "d"})


if __name__ == "__main__":
    unittest.main()
