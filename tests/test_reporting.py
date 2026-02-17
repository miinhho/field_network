from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag.calibration import run_calibration_with_summary
from ffrag.reporting import calibration_markdown_report, calibration_rows_csv, calibration_summary_csv


class ReportingTests(unittest.TestCase):
    def test_reporting_templates_emit_expected_headers(self) -> None:
        rows, summary = run_calibration_with_summary(num_scenarios=2, seed=3, batch="default")
        rows_csv = calibration_rows_csv(rows)
        summary_csv = calibration_summary_csv(summary)
        md = calibration_markdown_report(rows, summary, top_k=3)

        self.assertIn("avg_longrun_churn", rows_csv)
        self.assertIn("avg_cluster_ann_cache_hit_rate", rows_csv)
        self.assertIn("candidate_count", summary_csv)
        self.assertIn("# Calibration Report", md)
        self.assertIn("Top Candidates", md)
        self.assertIn("Ctx Evicted", md)


if __name__ == "__main__":
    unittest.main()
