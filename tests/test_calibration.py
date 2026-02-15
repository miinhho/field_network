from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag.calibration import candidate_configs, run_calibration


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


if __name__ == "__main__":
    unittest.main()
