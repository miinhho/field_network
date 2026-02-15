from datetime import datetime
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag import DynamicGraphSimulator, Perturbation


class DynamicSimulatorTests(unittest.TestCase):
    def test_simulator_produces_frames(self) -> None:
        sim = DynamicGraphSimulator()
        g = sim.demo_graph()
        p = Perturbation(
            perturbation_id="sim-test-p1",
            timestamp=datetime(2026, 2, 3, 9, 0, 0),
            targets=["hub"],
            intensity=1.0,
        )
        trace = sim.run(g, p, steps=3, top_k=4)
        self.assertEqual(len(trace.frames), 3)
        for frame in trace.frames:
            self.assertGreaterEqual(frame.critical_transition_score, 0.0)
            self.assertLessEqual(frame.critical_transition_score, 1.0)
            self.assertGreaterEqual(frame.planner_horizon, 2)
            self.assertLessEqual(frame.planner_horizon, 5)
            self.assertGreaterEqual(len(frame.top_final_nodes), 1)
            self.assertGreaterEqual(len(frame.node_positions), 1)

        moved = False
        for i in range(1, len(trace.frames)):
            prev_pos = trace.frames[i - 1].node_positions
            cur_pos = trace.frames[i].node_positions
            for nid in set(prev_pos.keys()) & set(cur_pos.keys()):
                dx = abs(prev_pos[nid][0] - cur_pos[nid][0])
                dy = abs(prev_pos[nid][1] - cur_pos[nid][1])
                if dx + dy > 1e-6:
                    moved = True
                    break
            if moved:
                break
        self.assertTrue(moved)

    def test_generate_large_graph(self) -> None:
        sim = DynamicGraphSimulator()
        g = sim.generate_graph(node_count=1200, avg_degree=3.0, seed=11)
        self.assertEqual(len(g.actants), 1200)
        self.assertGreaterEqual(len(g.interactions), 1199)


if __name__ == "__main__":
    unittest.main()
