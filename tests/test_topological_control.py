from datetime import datetime
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag import Actant, Interaction, LayeredGraph
from ffrag.flow import TopologicalFlowController


class TopologicalControlTests(unittest.TestCase):
    def _graph(self) -> LayeredGraph:
        g = LayeredGraph(graph_id="tc1", schema_version="0.1")
        g.actants = {
            "n1": Actant("n1", "entity", "n1"),
            "n2": Actant("n2", "entity", "n2"),
            "n3": Actant("n3", "entity", "n3"),
        }
        g.interactions = [
            Interaction("e1", datetime(2026, 2, 1, 9), "n1", "n2", "social", 1.0),
            Interaction("e2", datetime(2026, 2, 1, 10), "n2", "n3", "social", 1.0),
            Interaction("e3", datetime(2026, 2, 1, 11), "n1", "n3", "social", 1.0),
        ]
        return g

    def test_controller_returns_node_control_and_reduced_divergence(self) -> None:
        controller = TopologicalFlowController()
        out = controller.compute(
            self._graph(),
            impact_by_actant={"n1": 1.2, "n2": 0.4, "n3": 0.1},
            state={"transition_speed": 0.8, "temporal_regularity": 0.2, "schedule_density": 4.0},
        )
        self.assertEqual(len(out.node_control), 3)
        self.assertGreaterEqual(out.control_energy, 0.0)
        self.assertGreaterEqual(out.residual_ratio, 0.0)
        self.assertLessEqual(out.residual_ratio, 10.0)
        self.assertLessEqual(out.divergence_norm_after, out.divergence_norm_before + 1e-6)
        self.assertGreaterEqual(out.saturation_ratio, 0.0)
        self.assertLessEqual(out.saturation_ratio, 1.0)
        self.assertGreaterEqual(out.gain_k_div, 0.1)
        self.assertGreaterEqual(out.objective_score, 0.0)
        self.assertGreaterEqual(out.gradient_norm, 0.0)
        self.assertGreaterEqual(out.curl_norm, 0.0)
        self.assertGreaterEqual(out.harmonic_norm, 0.0)
        self.assertGreaterEqual(out.curl_ratio, 0.0)
        self.assertGreaterEqual(out.harmonic_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
