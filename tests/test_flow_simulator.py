from datetime import datetime
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag import Actant, Interaction, LayeredGraph, Perturbation
from ffrag.flow import FlowSimulator


class FlowSimulatorTests(unittest.TestCase):
    def _graph(self) -> LayeredGraph:
        g = LayeredGraph(graph_id="fs1", schema_version="0.1")
        for node in ["a", "b", "c"]:
            g.actants[node] = Actant(node, "entity", node)
        g.interactions = [
            Interaction("e1", datetime(2026, 2, 4, 9), "a", "b", "social", 1.0),
            Interaction("e2", datetime(2026, 2, 4, 10), "b", "c", "social", 1.0),
        ]
        return g

    def test_default_propagation_stays_nonnegative(self) -> None:
        sim = FlowSimulator()
        out = sim.propagate(
            self._graph(),
            Perturbation(
                perturbation_id="p1",
                timestamp=datetime(2026, 2, 4, 11),
                targets=["a"],
                intensity=1.0,
            ),
        )
        self.assertGreaterEqual(out.impact_by_actant.get("a", 0.0), 0.0)
        self.assertGreaterEqual(out.impact_by_actant.get("b", 0.0), 0.0)

    def test_target_weights_enable_signed_seed(self) -> None:
        sim = FlowSimulator()
        out = sim.propagate(
            self._graph(),
            Perturbation(
                perturbation_id="p2",
                timestamp=datetime(2026, 2, 4, 11, 5),
                targets=["a", "c"],
                intensity=1.0,
                metadata={"target_weights": {"a": 1.0, "c": -0.8}},
            ),
        )
        self.assertGreater(out.impact_by_actant.get("a", 0.0), 0.0)
        self.assertLess(out.impact_by_actant.get("c", 0.0), 0.0)

    def test_negative_edge_polarity_flips_neighbor_sign(self) -> None:
        g = LayeredGraph(graph_id="fs2", schema_version="0.1")
        for node in ["a", "b"]:
            g.actants[node] = Actant(node, "entity", node)
        g.interactions = [
            Interaction(
                "e1",
                datetime(2026, 2, 4, 12),
                "a",
                "b",
                "social",
                1.0,
                metadata={"polarity": -1.0},
            )
        ]
        sim = FlowSimulator()
        out = sim.propagate(
            g,
            Perturbation(
                perturbation_id="p3",
                timestamp=datetime(2026, 2, 4, 12, 5),
                targets=["a"],
                intensity=1.0,
            ),
        )
        self.assertGreater(out.impact_by_actant.get("a", 0.0), 0.0)
        self.assertLess(out.impact_by_actant.get("b", 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
