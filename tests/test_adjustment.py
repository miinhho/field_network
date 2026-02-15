from datetime import datetime
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag import Actant, Interaction, LayeredGraph
from ffrag.flow import AdjustmentPlannerConfig, DynamicGraphAdjuster


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
        self.assertGreaterEqual(out.selected_planner_horizon, 2)
        self.assertLessEqual(out.selected_planner_horizon, 5)
        self.assertGreaterEqual(out.selected_edit_budget, 0)
        self.assertLessEqual(out.selected_edit_budget, 2)
        self.assertGreaterEqual(out.graph_density, 0.0)
        self.assertLessEqual(out.graph_density, 1.0)
        self.assertGreaterEqual(out.impact_noise, 0.0)
        self.assertLessEqual(out.impact_noise, 1.0)
        self.assertGreaterEqual(out.coupling_penalty, 0.0)
        self.assertLessEqual(out.coupling_penalty, 1.0)
        self.assertGreaterEqual(out.applied_new_edges, 0)
        self.assertGreaterEqual(out.applied_drop_edges, 0)
        self.assertGreaterEqual(out.blocked_drop_edges, 0)
        self.assertLessEqual(out.applied_new_edges, len(out.suggested_new_edges))
        self.assertLessEqual(out.applied_drop_edges, len(out.suggested_drop_edges))

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
        self.assertGreaterEqual(risky.selected_planner_horizon, base.selected_planner_horizon)
        self.assertLessEqual(risky.selected_edit_budget, base.selected_edit_budget)
        self.assertLessEqual(risky.applied_new_edges + risky.applied_drop_edges, base.applied_new_edges + base.applied_drop_edges)

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

    def test_control_coupling_increases_objective_when_control_is_unstable(self) -> None:
        adjuster = DynamicGraphAdjuster()
        base = adjuster.adjust(
            self._graph(),
            impact_by_actant={"a": 1.0, "b": 0.8, "c": 0.2},
            state={"transition_speed": 0.5, "temporal_regularity": 0.4, "schedule_density": 3.0},
            control_context={"residual_ratio": 0.05, "divergence_norm_after": 0.08, "control_energy": 0.2},
        )
        unstable = adjuster.adjust(
            self._graph(),
            impact_by_actant={"a": 1.0, "b": 0.8, "c": 0.2},
            state={"transition_speed": 0.5, "temporal_regularity": 0.4, "schedule_density": 3.0},
            control_context={"residual_ratio": 0.9, "divergence_norm_after": 1.6, "control_energy": 2.8},
        )
        self.assertGreaterEqual(unstable.coupling_penalty, base.coupling_penalty)
        self.assertGreaterEqual(unstable.adjustment_objective_score, base.adjustment_objective_score - 1e-6)

    def test_phase_slowing_reduces_edit_budget(self) -> None:
        adjuster = DynamicGraphAdjuster()
        base = adjuster.adjust(
            self._dense_graph(),
            impact_by_actant={"a": 1.0, "b": 0.9, "c": 0.8, "d": 0.7, "e": 0.6},
            state={"transition_speed": 0.5, "temporal_regularity": 0.4, "schedule_density": 3.0},
            phase_context={
                "critical_transition_score": 0.35,
                "early_warning_score": 0.3,
                "coherence_break_score": 0.2,
                "critical_slowing_score": 0.1,
                "hysteresis_proxy_score": 0.1,
            },
        )
        slowed = adjuster.adjust(
            self._dense_graph(),
            impact_by_actant={"a": 1.0, "b": 0.9, "c": 0.8, "d": 0.7, "e": 0.6},
            state={"transition_speed": 0.5, "temporal_regularity": 0.4, "schedule_density": 3.0},
            phase_context={
                "critical_transition_score": 0.35,
                "early_warning_score": 0.3,
                "coherence_break_score": 0.2,
                "critical_slowing_score": 0.9,
                "hysteresis_proxy_score": 0.8,
            },
        )
        self.assertLessEqual(slowed.selected_edit_budget, base.selected_edit_budget)

    def test_bridge_edge_drop_is_blocked_by_safety_constraint(self) -> None:
        g = LayeredGraph(graph_id="ga_bridge", schema_version="0.1")
        g.actants = {
            "x": Actant("x", "person", "X"),
            "y": Actant("y", "person", "Y"),
            "z": Actant("z", "person", "Z"),
        }
        g.interactions = [
            Interaction("b1", datetime(2026, 2, 2, 9), "x", "y", "social", 0.2),
            Interaction("b2", datetime(2026, 2, 2, 10), "y", "z", "social", 0.2),
        ]
        adjuster = DynamicGraphAdjuster(max_structural_edits=2, apply_structural_edits=True)
        # Either edge in a 3-node chain is a bridge and should be protected.
        self.assertFalse(adjuster._can_drop_edge(g, 0))
        self.assertFalse(adjuster._can_drop_edge(g, 1))

    def test_planner_config_hook_changes_objective_terms(self) -> None:
        default_adjuster = DynamicGraphAdjuster()
        tuned_adjuster = DynamicGraphAdjuster(
            planner_config=AdjustmentPlannerConfig(
                churn_weight_base=0.2,
                churn_weight_density_gain=0.1,
                volatility_weight_base=0.6,
                volatility_weight_noise_gain=0.2,
            )
        )
        state = {"transition_speed": 0.6, "temporal_regularity": 0.3, "schedule_density": 3.0}
        impacts = {"a": 1.0, "b": 0.8, "c": 0.2}
        out_default = default_adjuster.adjust(self._graph(), impacts, state)
        out_tuned = tuned_adjuster.adjust(self._graph(), impacts, state)
        self.assertNotEqual(
            out_default.objective_terms["volatility"],
            out_tuned.objective_terms["volatility"],
        )

    def test_affinity_plasticity_suggests_emergent_link(self) -> None:
        g = self._graph()
        adjuster = DynamicGraphAdjuster(max_structural_edits=2, apply_structural_edits=False)
        impacts = {"a": 1.0, "b": 0.05, "c": 1.0}
        state = {"transition_speed": 0.55, "temporal_regularity": 0.35, "schedule_density": 2.8}

        out = None
        for _ in range(20):
            out = adjuster.adjust(g, impact_by_actant=impacts, state=state)
            g = out.adjusted_graph
        self.assertIsNotNone(out)
        assert out is not None
        self.assertGreaterEqual(out.affinity_suggested_new_edges, 1)
        self.assertIn(("a", "c"), out.suggested_new_edges)

    def test_plasticity_budget_is_bounded(self) -> None:
        g = self._dense_graph()
        adjuster = DynamicGraphAdjuster(max_structural_edits=2, apply_structural_edits=True)
        impacts = {"a": 1.0, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.5}
        state = {"transition_speed": 0.6, "temporal_regularity": 0.2, "schedule_density": 4.2}
        out = None
        for _ in range(5):
            out = adjuster.adjust(g, impact_by_actant=impacts, state=state)
            g = out.adjusted_graph
        self.assertIsNotNone(out)
        assert out is not None
        self.assertGreaterEqual(out.mean_plasticity_budget, 0.0)
        self.assertLessEqual(out.mean_plasticity_budget, 1.0)
        self.assertGreaterEqual(out.tracked_affinity_pairs, 1)

    def test_supervisory_confusion_reduces_merge_prone_rewiring(self) -> None:
        adjuster = DynamicGraphAdjuster(max_structural_edits=2, apply_structural_edits=False)
        g = self._dense_graph()
        impacts = {"a": 1.0, "b": 0.9, "c": 0.85, "d": 0.7, "e": 0.65}
        state = {"transition_speed": 0.6, "temporal_regularity": 0.3, "schedule_density": 4.0}

        base = adjuster.adjust(g, impact_by_actant=impacts, state=state)
        confused = adjuster.adjust(
            g,
            impact_by_actant=impacts,
            state=state,
            supervisory_context={"confusion_score": 0.9, "forgetting_score": 0.1},
        )

        self.assertLessEqual(
            confused.supervisory_policy_trace["new_edge_bias"],
            base.supervisory_policy_trace["new_edge_bias"],
        )
        self.assertGreaterEqual(
            confused.supervisory_policy_trace["theta_on"],
            base.supervisory_policy_trace["theta_on"],
        )
        self.assertLessEqual(
            confused.supervisory_policy_trace["eta_up"],
            base.supervisory_policy_trace["eta_up"],
        )

    def test_supervisory_forgetting_reduces_aggressive_pruning(self) -> None:
        adjuster = DynamicGraphAdjuster(max_structural_edits=2, apply_structural_edits=True)
        g = self._dense_graph()
        impacts = {"a": 1.0, "b": 0.9, "c": 0.8, "d": 0.7, "e": 0.6}
        state = {"transition_speed": 0.55, "temporal_regularity": 0.35, "schedule_density": 3.8}

        base = adjuster.adjust(
            g,
            impact_by_actant=impacts,
            state=state,
            supervisory_context={"confusion_score": 0.1, "forgetting_score": 0.1},
        )
        protected = adjuster.adjust(
            g,
            impact_by_actant=impacts,
            state=state,
            supervisory_context={"confusion_score": 0.1, "forgetting_score": 0.95},
        )

        self.assertLessEqual(
            protected.supervisory_policy_trace["drop_edge_bias"],
            base.supervisory_policy_trace["drop_edge_bias"],
        )
        self.assertGreaterEqual(
            protected.supervisory_policy_trace["eta_down"],
            base.supervisory_policy_trace["eta_down"],
        )
        self.assertLessEqual(protected.applied_drop_edges, base.applied_drop_edges)


if __name__ == "__main__":
    unittest.main()
