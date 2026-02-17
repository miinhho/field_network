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
        self.assertIn(controller.last_ann_backend, {"faiss", "exact"})
        self.assertGreaterEqual(len(out.cluster_assignment), 5)
        self.assertGreaterEqual(out.cluster_objective, 0.0)
        self.assertGreaterEqual(out.cross_scale_consistency, 0.0)
        self.assertLessEqual(out.cross_scale_consistency, 1.0)
        self.assertGreaterEqual(out.ann_cache_hit, 0.0)
        self.assertLessEqual(out.ann_cache_hit, 1.0)
        self.assertGreaterEqual(out.active_context_count, 1)

    def test_hybrid_knn_clusters_without_structure_edges(self) -> None:
        g = LayeredGraph(graph_id="ms-empty", schema_version="0.1")
        for node in ["n1", "n2", "n3", "n4"]:
            g.actants[node] = Actant(node, "entity", node)
        controller = ClusterFlowController()
        out = controller.plan(
            g,
            impact_by_actant={"n1": 1.0, "n2": 0.9, "n3": 0.8, "n4": 0.7},
            state={"transition_speed": 0.3, "temporal_regularity": 0.5, "schedule_density": 0.1},
        )
        unique_clusters = set(out.cluster_assignment.values())
        self.assertLess(len(unique_clusters), len(g.actants))

    def test_hybrid_knn_large_mode_uses_soft_separation(self) -> None:
        g = LayeredGraph(graph_id="ms-large", schema_version="0.1")
        for i in range(40):
            nid = f"n{i}"
            g.actants[nid] = Actant(nid, "entity", nid)
        impact = {f"n{i}": 1.0 - (i * 0.01) for i in range(40)}
        controller = ClusterFlowController()
        controller.hybrid_ann_backend = "exact"
        out = controller.plan(
            g,
            impact_by_actant=impact,
            state={"transition_speed": 0.4, "temporal_regularity": 0.5, "schedule_density": 0.2},
        )
        self.assertEqual(len(out.cluster_assignment), 40)
        self.assertEqual(controller.last_ann_backend, "exact")
        self.assertGreaterEqual(out.cluster_objective, 0.0)

    def test_context_state_lifecycle_eviction(self) -> None:
        controller = ClusterFlowController()
        controller.max_context_states = 2
        g = self._graph()
        impact = {"a": 1.0, "b": 0.9, "c": 0.7, "d": 0.2, "e": 0.1}
        state = {"transition_speed": 0.6, "temporal_regularity": 0.4, "schedule_density": 3.0}
        controller.plan(g, impact, state, context_id="ctx1")
        controller.plan(g, impact, state, context_id="ctx2")
        controller.plan(g, impact, state, context_id="ctx3")
        self.assertNotIn("ctx1", controller._prev_assignment_by_context)
        self.assertIn("ctx2", controller._prev_assignment_by_context)
        self.assertIn("ctx3", controller._prev_assignment_by_context)

    def test_ann_cache_reused_for_same_context_and_features(self) -> None:
        controller = ClusterFlowController()
        controller.hybrid_ann_backend = "exact"
        g = self._graph()
        impact = {"a": 1.0, "b": 0.9, "c": 0.7, "d": 0.2, "e": 0.1}
        state = {"transition_speed": 0.6, "temporal_regularity": 0.4, "schedule_density": 3.0}
        controller.plan(g, impact, state, context_id="same")
        first = controller._ann_cache_by_context.get("same")
        controller.plan(g, impact, state, context_id="same")
        second = controller._ann_cache_by_context.get("same")
        self.assertIsNotNone(first)
        self.assertIs(first.index, second.index)

    def test_importance_override_protects_context_from_eviction(self) -> None:
        controller = ClusterFlowController()
        controller.hybrid_ann_backend = "exact"
        controller.max_context_states = 2
        controller.context_half_life_steps = 1.0
        g = self._graph()
        impact = {"a": 1.0, "b": 0.9, "c": 0.7, "d": 0.2, "e": 0.1}
        state = {"transition_speed": 0.6, "temporal_regularity": 0.4, "schedule_density": 3.0}
        controller.set_context_importance("important", 50.0)
        controller.plan(g, impact, state, context_id="important")
        controller.plan(g, impact, state, context_id="regular1")
        controller.plan(g, impact, state, context_id="regular2")
        self.assertIn("important", controller._prev_assignment_by_context)

    def test_retention_floor_evicts_stale_context_without_capacity_pressure(self) -> None:
        controller = ClusterFlowController()
        controller.hybrid_ann_backend = "exact"
        controller.max_context_states = 10
        controller.context_half_life_steps = 1.0
        controller.context_frequency_weight = 0.0
        controller.context_retention_floor = 0.2
        g = self._graph()
        impact = {"a": 1.0, "b": 0.9, "c": 0.7, "d": 0.2, "e": 0.1}
        state = {"transition_speed": 0.6, "temporal_regularity": 0.4, "schedule_density": 3.0}

        controller.plan(g, impact, state, context_id="stale")
        controller.plan(g, impact, state, context_id="fresh1")
        controller.plan(g, impact, state, context_id="fresh2")
        controller.plan(g, impact, state, context_id="fresh3")

        self.assertNotIn("stale", controller._prev_assignment_by_context)
        self.assertIn("fresh3", controller._prev_assignment_by_context)

    def test_cluster_impact_reduces_cancellation_loss(self) -> None:
        controller = ClusterFlowController()
        assignment = {"a": "cluster_0", "b": "cluster_0", "c": "cluster_0"}
        node_impact = {"a": 1.0, "b": -0.95, "c": 0.1}
        out = controller._cluster_impact(assignment, node_impact)
        self.assertIn("cluster_0", out)
        self.assertGreater(out["cluster_0"], 0.2)

    def test_coarse_projection_preserves_signed_signal(self) -> None:
        controller = ClusterFlowController()
        val = controller._project_cluster_control(base_impact=0.1, cluster_delta=-0.8)
        self.assertLess(val, 0.0)

    def test_cross_scale_consistency_penalizes_sign_mixing(self) -> None:
        controller = ClusterFlowController()
        assignment = {"a": "c0", "b": "c0", "d": "c1", "e": "c1"}
        same_sign = {"a": 0.8, "b": 0.7, "d": -0.6, "e": -0.5}
        mixed_sign = {"a": 0.8, "b": -0.7, "d": -0.6, "e": 0.5}
        score_same = controller._cross_scale_consistency(assignment, same_sign)
        score_mixed = controller._cross_scale_consistency(assignment, mixed_sign)
        self.assertGreater(score_same, score_mixed)

    def test_cross_scale_consistency_penalizes_high_variance(self) -> None:
        controller = ClusterFlowController()
        controller.cross_scale_sign_weight = 0.2
        assignment = {"a": "c0", "b": "c0", "c": "c0"}
        stable = {"a": 0.6, "b": 0.62, "c": 0.58}
        volatile = {"a": 1.2, "b": 0.1, "c": 0.5}
        score_stable = controller._cross_scale_consistency(assignment, stable)
        score_volatile = controller._cross_scale_consistency(assignment, volatile)
        self.assertGreater(score_stable, score_volatile)


if __name__ == "__main__":
    unittest.main()
