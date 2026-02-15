from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag.benchmark import run_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_returns_three_methods(self) -> None:
        rows = run_benchmark(num_scenarios=5, top_k=3, seed=7)
        methods = {row.method for row in rows}
        self.assertEqual(methods, {"plain_rag", "graph_rag", "flow_graph_rag"})

    def test_metric_ranges(self) -> None:
        rows = run_benchmark(num_scenarios=3, top_k=2, seed=3)
        for row in rows:
            self.assertGreaterEqual(row.avg_recall_at_k, 0.0)
            self.assertLessEqual(row.avg_recall_at_k, 1.0)
            self.assertGreaterEqual(row.avg_precision_at_k, 0.0)
            self.assertLessEqual(row.avg_precision_at_k, 1.0)


if __name__ == "__main__":
    unittest.main()
