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

    def _dense_graph(self) -> LayeredGraph:
        g = LayeredGraph(graph_id="ga_dense", schema_version="0.1")
        for node in ["a", "b", "c", "d", "e"]:
            g.actants[node] = Actant(node, "person", node.upper())
        idx = 0
        nodes = list(g.actants.keys())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                g.interactions.append(
                    Interaction(f"de{idx}", datetime(2026, 2, 1, 9), nodes[i], nodes[j], "social", 1.0)
                )
                idx += 1
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
        self.assertGreaterEqual(out.adjustment_objective_score, 0.0)
        self.assertGreaterEqual(out.selected_adjustment_scale, 0.4)
        self.assertLessEqual(out.selected_adjustment_scale, 1.5)
        self.assertGreaterEqual(out.graph_density, 0.0)
        self.assertLessEqual(out.graph_density, 1.0)
        self.assertGreaterEqual(out.impact_noise, 0.0)
        self.assertLessEqual(out.impact_noise, 1.0)

    def test_phase_aware_adjustment_becomes_conservative(self) -> None:
        adjuster = DynamicGraphAdjuster()
        base = adjuster.adjust(
            self._graph(),
            impact_by_actant={"a": 1.0, "b": 0.8, "c": 0.2},
            state={"transition_speed": 0.7, "temporal_regularity": 0.3, "schedule_density": 4.0},
        )
        risky = adjuster.adjust(
            self._graph(),
            impact_by_actant={"a": 1.0, "b": 0.8, "c": 0.2},
            state={"transition_speed": 0.7, "temporal_regularity": 0.3, "schedule_density": 4.0},
            phase_context={
                "critical_transition_score": 0.9,
                "early_warning_score": 0.8,
                "coherence_break_score": 0.7,
            },
        )
        self.assertLessEqual(abs(risky.mean_weight_shift), abs(base.mean_weight_shift) + 1e-6)
        self.assertLessEqual(len(risky.suggested_new_edges), len(base.suggested_new_edges))
        self.assertLessEqual(len(risky.suggested_drop_edges), len(base.suggested_drop_edges))
        self.assertGreaterEqual(risky.adjustment_objective_score, base.adjustment_objective_score - 1e-6)
        self.assertLessEqual(risky.selected_adjustment_scale, base.selected_adjustment_scale + 1e-6)

    def test_sparse_vs_dense_adaptive_profile(self) -> None:
        adjuster = DynamicGraphAdjuster()
        sparse = adjuster.adjust(
            self._graph(),
            impact_by_actant={"a": 1.0, "b": 0.8, "c": 0.2},
            state={"transition_speed": 0.4, "temporal_regularity": 0.5, "schedule_density": 2.0},
        )
        dense = adjuster.adjust(
            self._dense_graph(),
            impact_by_actant={"a": 1.0, "b": 0.9, "c": 0.8, "d": 0.7, "e": 0.6},
            state={"transition_speed": 0.4, "temporal_regularity": 0.5, "schedule_density": 2.0},
        )
        self.assertLessEqual(sparse.graph_density, dense.graph_density)
        self.assertGreaterEqual(len(sparse.suggested_new_edges), len(dense.suggested_new_edges))


if __name__ == "__main__":
    unittest.main()
