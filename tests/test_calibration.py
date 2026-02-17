from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag.calibration import candidate_configs, candidate_profiles, run_calibration, run_calibration_with_summary


class CalibrationTests(unittest.TestCase):
    def test_candidate_configs_non_empty(self) -> None:
        cfgs = candidate_configs()
        self.assertGreaterEqual(len(cfgs), 1)

    def test_run_calibration_returns_sorted_rows(self) -> None:
        rows = run_calibration(num_scenarios=4, seed=7)
        self.assertGreaterEqual(len(rows), 1)
        for i in range(len(rows) - 1):
            self.assertLessEqual(rows[i].score, rows[i + 1].score)
        for row in rows:
            self.assertGreaterEqual(row.avg_coupling_penalty, 0.0)
            self.assertLessEqual(row.avg_coupling_penalty, 1.0)
            self.assertGreaterEqual(row.avg_supervisory_confusion, 0.0)
            self.assertLessEqual(row.avg_supervisory_confusion, 1.0)
            self.assertGreaterEqual(row.avg_supervisory_forgetting, 0.0)
            self.assertLessEqual(row.avg_supervisory_forgetting, 1.0)
            self.assertGreaterEqual(row.avg_longrun_churn, 0.0)
            self.assertGreaterEqual(row.avg_longrun_retention, 0.0)
            self.assertLessEqual(row.avg_longrun_retention, 1.0)
            self.assertGreaterEqual(row.avg_longrun_diversity, 0.0)
            self.assertLessEqual(row.avg_longrun_diversity, 1.0)
            self.assertGreaterEqual(row.avg_cluster_ann_cache_hit_rate, 0.0)
            self.assertLessEqual(row.avg_cluster_ann_cache_hit_rate, 1.0)
            self.assertGreaterEqual(row.avg_cluster_active_contexts, 0.0)
            self.assertGreaterEqual(row.avg_cluster_evicted_contexts, 0.0)

    def test_candidate_profiles_modes(self) -> None:
        self.assertGreaterEqual(len(candidate_profiles("default")), 1)
        self.assertGreaterEqual(len(candidate_profiles("plasticity")), 2)
        self.assertGreaterEqual(len(candidate_profiles("mixed")), len(candidate_profiles("default")))

    def test_run_calibration_with_summary_returns_valid_range(self) -> None:
        rows, summary = run_calibration_with_summary(num_scenarios=4, seed=5, batch="mixed", top_fraction=0.5)
        self.assertGreaterEqual(len(rows), 1)
        self.assertGreaterEqual(summary.candidate_count, 1)
        self.assertGreaterEqual(summary.top_count, 1)
        self.assertLessEqual(summary.eta_up_min, summary.eta_up_max)
        self.assertLessEqual(summary.theta_off_min, summary.theta_off_max)
        self.assertLessEqual(summary.theta_on_min, summary.theta_on_max)
        self.assertLessEqual(summary.hysteresis_dwell_min, summary.hysteresis_dwell_max)


if __name__ == "__main__":
    unittest.main()
