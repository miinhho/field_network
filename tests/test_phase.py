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
        self.assertIn(out.dominant_regime, {"calm", "adaptive", "turbulent", "critical"})

    def test_empty_inputs_return_safe_defaults(self) -> None:
        analyzer = PhaseTransitionAnalyzer()
        out = analyzer.analyze([], [], [], [])
        self.assertEqual(out.critical_transition_score, 0.0)
        self.assertEqual(out.early_warning_score, 0.0)
        self.assertEqual(out.regime_switch_count, 0)
        self.assertEqual(out.regime_persistence_score, 0.0)
        self.assertEqual(out.coherence_break_score, 0.0)
        self.assertEqual(out.dominant_regime, "calm")


if __name__ == "__main__":
    unittest.main()
