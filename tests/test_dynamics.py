from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag.flow.dynamics import FlowFieldDynamics


class DynamicsTests(unittest.TestCase):
    def test_simulation_returns_snapshots(self) -> None:
        dyn = FlowFieldDynamics()
        state = {
            "social_entropy": 0.5,
            "temporal_regularity": 0.8,
            "spatial_range": 1.0,
            "schedule_density": 2.0,
            "network_centrality": 0.4,
            "transition_speed": 0.3,
        }
        history = [state, {**state, "schedule_density": 2.2, "transition_speed": 0.35}]
        shock = {"schedule_density": 0.2, "transition_speed": 0.1}
        result = dyn.simulate(initial_state=state, history=history, shock=shock, steps=4)

        self.assertGreaterEqual(len(result.snapshots), 1)
        self.assertIn("schedule_density", result.snapshots[-1].state)
        self.assertIn("transition_speed", result.snapshots[-1].velocity)

    def test_viscosity_limits_velocity(self) -> None:
        dyn = FlowFieldDynamics()
        low_visc = {
            "social_entropy": 0.2,
            "temporal_regularity": 0.1,
            "spatial_range": 0.2,
            "schedule_density": 1.0,
            "network_centrality": 0.2,
            "transition_speed": 0.1,
        }
        high_visc = {**low_visc, "temporal_regularity": 1.0, "schedule_density": 10.0}
        shock = {"schedule_density": 0.2, "transition_speed": 0.2}

        low = dyn.simulate(initial_state=low_visc, history=[low_visc], shock=shock, steps=1)
        high = dyn.simulate(initial_state=high_visc, history=[high_visc], shock=shock, steps=1)

        low_mag = abs(low.snapshots[0].velocity["schedule_density"])
        high_mag = abs(high.snapshots[0].velocity["schedule_density"])
        self.assertGreater(low_mag, high_mag)


if __name__ == "__main__":
    unittest.main()
