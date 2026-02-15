from datetime import datetime
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag import Actant, Interaction, LayeredGraph
from ffrag.flow import SupervisoryControlState, SupervisoryMetricsAnalyzer


class SupervisoryMetricsTests(unittest.TestCase):
    def _base_graph(self) -> LayeredGraph:
        g = LayeredGraph(graph_id="sv-base", schema_version="0.1")
        for nid in ["a", "b", "c", "d", "e", "f"]:
            g.actants[nid] = Actant(nid, "entity", nid.upper())
        edges = [
            ("a", "b", 1.2),
            ("a", "c", 1.1),
            ("b", "c", 1.0),
            ("d", "e", 0.6),
            ("d", "f", 0.5),
            ("e", "f", 0.5),
        ]
        for i, (s, t, w) in enumerate(edges):
            g.interactions.append(
                Interaction(f"e{i}", datetime(2026, 2, 1, 9, i), s, t, "social", w)
            )
        return g

    def _mixed_graph(self) -> LayeredGraph:
        g = self._base_graph()
        bridge_edges = [
            ("a", "d", 1.1),
            ("b", "e", 1.0),
            ("c", "f", 1.0),
        ]
        for i, (s, t, w) in enumerate(bridge_edges):
            g.interactions.append(
                Interaction(f"x{i}", datetime(2026, 2, 1, 10, i), s, t, "social", w)
            )
        return g

    def test_metrics_are_bounded(self) -> None:
        analyzer = SupervisoryMetricsAnalyzer()
        metrics, state = analyzer.analyze(
            self._base_graph(),
            impact_by_actant={"a": 1.0, "b": 0.8, "c": 0.7, "d": 0.3, "e": 0.2, "f": 0.1},
        )
        self.assertIsInstance(state, SupervisoryControlState)
        self.assertGreaterEqual(metrics.confusion_score, 0.0)
        self.assertLessEqual(metrics.confusion_score, 1.0)
        self.assertGreaterEqual(metrics.forgetting_score, 0.0)
        self.assertLessEqual(metrics.forgetting_score, 1.0)
        self.assertGreaterEqual(metrics.cluster_margin, 0.0)
        self.assertLessEqual(metrics.cluster_margin, 1.0)
        self.assertGreaterEqual(metrics.mixing_entropy, 0.0)
        self.assertLessEqual(metrics.mixing_entropy, 1.0)
        self.assertGreaterEqual(metrics.retention_loss, 0.0)
        self.assertLessEqual(metrics.retention_loss, 1.0)
        self.assertGreaterEqual(metrics.connectivity_loss, 0.0)
        self.assertLessEqual(metrics.connectivity_loss, 1.0)

    def test_confusion_increases_with_low_margin_and_high_mixing(self) -> None:
        analyzer = SupervisoryMetricsAnalyzer()
        clear_metrics, _ = analyzer.analyze(
            self._base_graph(),
            impact_by_actant={"a": 1.1, "b": 1.0, "c": 0.9, "d": 0.2, "e": 0.2, "f": 0.1},
        )
        mixed_metrics, _ = analyzer.analyze(
            self._mixed_graph(),
            impact_by_actant={"a": 0.7, "b": 0.7, "c": 0.7, "d": 0.6, "e": 0.6, "f": 0.6},
        )
        self.assertGreaterEqual(mixed_metrics.mixing_entropy, clear_metrics.mixing_entropy)
        self.assertLessEqual(mixed_metrics.cluster_margin, clear_metrics.cluster_margin)
        self.assertGreaterEqual(mixed_metrics.confusion_score, clear_metrics.confusion_score)

    def test_forgetting_increases_when_important_nodes_disconnect(self) -> None:
        analyzer = SupervisoryMetricsAnalyzer()
        important = ["a", "b", "c"]
        baseline_graph = self._base_graph()
        baseline_metrics, state = analyzer.analyze(
            baseline_graph,
            impact_by_actant={"a": 1.0, "b": 0.9, "c": 0.8, "d": 0.2, "e": 0.2, "f": 0.1},
            important_nodes=important,
        )

        degraded = LayeredGraph(graph_id="sv-degraded", schema_version="0.1")
        degraded.actants = dict(baseline_graph.actants)
        degraded.interactions = [
            Interaction("z1", datetime(2026, 2, 1, 11, 0), "d", "e", "social", 0.6),
            Interaction("z2", datetime(2026, 2, 1, 11, 1), "e", "f", "social", 0.6),
        ]
        degraded_metrics, _ = analyzer.analyze(
            degraded,
            impact_by_actant={"a": 1.0, "b": 0.9, "c": 0.8, "d": 0.2, "e": 0.2, "f": 0.1},
            state=state,
            important_nodes=important,
        )
        self.assertLessEqual(baseline_metrics.forgetting_score, degraded_metrics.forgetting_score)
        self.assertLessEqual(baseline_metrics.retention_loss, degraded_metrics.retention_loss)
        self.assertLessEqual(baseline_metrics.connectivity_loss, degraded_metrics.connectivity_loss)


if __name__ == "__main__":
    unittest.main()
