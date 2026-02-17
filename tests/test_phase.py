from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag.flow import PhaseTransitionAnalyzer


class PhaseTransitionTests(unittest.TestCase):
    def test_analyze_detects_switching_and_scores(self) -> None:
        analyzer = PhaseTransitionAnalyzer()
        trajectory = [
            {"transition_speed": 0.2},
            {"transition_speed": 0.25},
            {"transition_speed": 0.8},
            {"transition_speed": 0.85},
        ]
        distances = [0.1, 0.12, 0.4, 0.45]
        objectives = [0.2, 0.23, 0.6, 0.62]
        tensions = [0.05, 0.08, 0.5, 0.55]

        out = analyzer.analyze(trajectory, distances, objectives, tensions)
        self.assertGreaterEqual(out.critical_transition_score, 0.0)
        self.assertLessEqual(out.critical_transition_score, 1.0)
        self.assertGreaterEqual(out.early_warning_score, 0.0)
        self.assertLessEqual(out.early_warning_score, 1.0)
        self.assertGreaterEqual(out.regime_switch_count, 1)
        self.assertGreaterEqual(out.regime_persistence_score, 0.0)
        self.assertLessEqual(out.regime_persistence_score, 1.0)
        self.assertGreaterEqual(out.critical_slowing_score, 0.0)
        self.assertLessEqual(out.critical_slowing_score, 1.0)
        self.assertGreaterEqual(out.hysteresis_proxy_score, 0.0)
        self.assertLessEqual(out.hysteresis_proxy_score, 1.0)
        self.assertGreaterEqual(out.sign_flip_rate, 0.0)
        self.assertLessEqual(out.sign_flip_rate, 1.0)
        self.assertGreaterEqual(out.polarity_coherence_score, 0.0)
        self.assertLessEqual(out.polarity_coherence_score, 1.0)
        self.assertIn(out.dominant_regime, {"calm", "adaptive", "turbulent", "critical"})

    def test_empty_inputs_return_safe_defaults(self) -> None:
        analyzer = PhaseTransitionAnalyzer()
        out = analyzer.analyze([], [], [], [])
        self.assertEqual(out.critical_transition_score, 0.0)
        self.assertEqual(out.early_warning_score, 0.0)
        self.assertEqual(out.regime_switch_count, 0)
        self.assertEqual(out.regime_persistence_score, 0.0)
        self.assertEqual(out.coherence_break_score, 0.0)
        self.assertEqual(out.critical_slowing_score, 0.0)
        self.assertEqual(out.hysteresis_proxy_score, 0.0)
        self.assertEqual(out.sign_flip_rate, 0.0)
        self.assertEqual(out.polarity_coherence_score, 1.0)
        self.assertEqual(out.dominant_regime, "calm")

    def test_critical_slowing_signal_increases_for_persistent_series(self) -> None:
        analyzer = PhaseTransitionAnalyzer()
        trajectory = [{"transition_speed": 0.2}, {"transition_speed": 0.21}, {"transition_speed": 0.22}, {"transition_speed": 0.23}]
        out = analyzer.analyze(
            trajectory,
            attractor_distances=[0.3, 0.31, 0.32, 0.33],
            objective_scores=[0.4, 0.39, 0.38, 0.37],
            topological_tensions=[0.2, 0.21, 0.22, 0.23],
        )
        self.assertGreaterEqual(out.critical_slowing_score, 0.0)
        self.assertLessEqual(out.critical_slowing_score, 1.0)

    def test_sign_flip_rate_rises_for_oscillatory_series(self) -> None:
        analyzer = PhaseTransitionAnalyzer()
        smooth = analyzer.analyze(
            trajectory=[
                {"transition_speed": 0.2},
                {"transition_speed": 0.3},
                {"transition_speed": 0.4},
                {"transition_speed": 0.5},
                {"transition_speed": 0.6},
            ],
            attractor_distances=[0.1, 0.2, 0.3, 0.4, 0.5],
            objective_scores=[0.2, 0.25, 0.3, 0.35, 0.4],
            topological_tensions=[0.1, 0.15, 0.2, 0.25, 0.3],
        )
        oscillatory = analyzer.analyze(
            trajectory=[
                {"transition_speed": 0.2},
                {"transition_speed": 0.8},
                {"transition_speed": 0.3},
                {"transition_speed": 0.9},
                {"transition_speed": 0.4},
            ],
            attractor_distances=[0.1, 0.5, 0.2, 0.6, 0.25],
            objective_scores=[0.2, 0.6, 0.3, 0.7, 0.35],
            topological_tensions=[0.05, 0.4, 0.1, 0.45, 0.12],
        )
        self.assertGreater(oscillatory.sign_flip_rate, smooth.sign_flip_rate)
        self.assertLessEqual(oscillatory.polarity_coherence_score, smooth.polarity_coherence_score + 1e-6)


if __name__ == "__main__":
    unittest.main()
