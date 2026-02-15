from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag.flow.analysis import FlowAnalyzerConfig, FlowDynamicsAnalyzer


class FlowAnalysisTests(unittest.TestCase):
    def test_transition_matrix_probability_bounds(self) -> None:
        analyzer = FlowDynamicsAnalyzer()
        trajectory = [
            {
                "temporal_regularity": 0.9,
                "transition_speed": 0.2,
                "schedule_density": 1.5,
            },
            {
                "temporal_regularity": 0.5,
                "transition_speed": 0.6,
                "schedule_density": 3.0,
            },
            {
                "temporal_regularity": 0.4,
                "transition_speed": 0.8,
                "schedule_density": 6.2,
            },
        ]
        shock = {"transition_speed": 0.3, "schedule_density": 0.2}
        out = analyzer.transition_analysis(trajectory, shock=shock)
        self.assertGreaterEqual(len(out.states), 1)
        self.assertIsNotNone(out.attractor_basin_state)
        self.assertGreaterEqual(out.attractor_basin_radius, 0.0)
        self.assertGreaterEqual(out.basin_occupancy, 0.0)
        self.assertLessEqual(out.basin_occupancy, 1.0)
        self.assertGreaterEqual(out.avg_trigger_confidence, 0.0)
        self.assertLessEqual(out.avg_trigger_confidence, 1.0)
        self.assertGreaterEqual(out.causal_alignment_score, 0.0)
        self.assertLessEqual(out.causal_alignment_score, 1.0)
        for row in out.transition_matrix.values():
            for prob in row.values():
                self.assertGreaterEqual(prob, 0.0)
                self.assertLessEqual(prob, 1.0)

    def test_resilience_has_non_negative_hysteresis(self) -> None:
        analyzer = FlowDynamicsAnalyzer()
        forward = [1.0, 0.8, 0.6, 0.55]
        backward = [1.0, 0.85, 0.7, 0.6]
        out = analyzer.resilience_analysis(forward, baseline_distances=backward)
        self.assertGreaterEqual(out.recovery_rate, 0.0)
        self.assertGreaterEqual(out.hysteresis_index, 0.0)
        self.assertGreaterEqual(out.overshoot_index, 0.0)
        self.assertGreaterEqual(out.settling_time, 1)
        self.assertGreaterEqual(out.path_efficiency, 0.0)
        self.assertLessEqual(out.path_efficiency, 1.0)

    def test_empty_inputs_are_safe(self) -> None:
        analyzer = FlowDynamicsAnalyzer()
        trans = analyzer.transition_analysis([])
        resil = analyzer.resilience_analysis([])
        self.assertEqual(trans.states, [])
        self.assertEqual(trans.transition_triggers, [])
        self.assertEqual(trans.attractor_basin_state, None)
        self.assertEqual(trans.attractor_basin_radius, 0.0)
        self.assertEqual(trans.basin_occupancy, 0.0)
        self.assertEqual(trans.avg_trigger_confidence, 0.0)
        self.assertEqual(trans.causal_alignment_score, 0.0)
        self.assertEqual(resil.recovery_steps, 0)
        self.assertEqual(resil.settling_time, 0)

    def test_quantile_basin_strategy(self) -> None:
        analyzer = FlowDynamicsAnalyzer(
            config=FlowAnalyzerConfig(basin_strategy="quantile", basin_quantile=0.5)
        )
        trajectory = [
            {"transition_speed": 0.2, "temporal_regularity": 0.8, "schedule_density": 1.0},
            {"transition_speed": 0.3, "temporal_regularity": 0.7, "schedule_density": 1.4},
            {"transition_speed": 0.5, "temporal_regularity": 0.6, "schedule_density": 2.1},
        ]
        out = analyzer.transition_analysis(trajectory, shock={"transition_speed": 0.2})
        self.assertGreaterEqual(out.attractor_basin_radius, 0.0)

    def test_causal_lag_config_is_applied(self) -> None:
        analyzer = FlowDynamicsAnalyzer(
            config=FlowAnalyzerConfig(causal_lag_steps=1, trigger_speed_jump_threshold=0.05)
        )
        trajectory = [
            {"transition_speed": 0.1, "temporal_regularity": 0.9, "schedule_density": 1.0},
            {"transition_speed": 0.2, "temporal_regularity": 0.85, "schedule_density": 1.1},
            {"transition_speed": 0.6, "temporal_regularity": 0.6, "schedule_density": 2.0},
        ]
        out = analyzer.transition_analysis(trajectory, shock={"transition_speed": 0.5})
        self.assertGreaterEqual(out.causal_alignment_score, 0.0)
        self.assertLessEqual(out.causal_alignment_score, 1.0)


if __name__ == "__main__":
    unittest.main()
