from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag.flow.dynamics import FlowFieldDynamics
import numpy as np


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

    def test_effective_mass_limits_velocity(self) -> None:
        dyn = FlowFieldDynamics()
        low_mass = {
            "social_entropy": 0.2,
            "temporal_regularity": 0.1,
            "spatial_range": 0.2,
            "schedule_density": 1.0,
            "network_centrality": 0.2,
            "transition_speed": 0.1,
        }
        high_mass = {**low_mass, "temporal_regularity": 1.0, "schedule_density": 10.0}
        shock = {"schedule_density": 0.2, "transition_speed": 0.2}

        low = dyn.simulate(initial_state=low_mass, history=[low_mass], shock=shock, steps=1)
        high = dyn.simulate(initial_state=high_mass, history=[high_mass], shock=shock, steps=1)

        low_mag = abs(low.snapshots[0].velocity["schedule_density"])
        high_mag = abs(high.snapshots[0].velocity["schedule_density"])
        self.assertGreater(low_mag, high_mag)

    def test_effective_mass_is_dimensionwise(self) -> None:
        dyn = FlowFieldDynamics(max_step_norm=0.8)
        force = np.array([0.05, 1.2, 0.02, 0.03, 0.04, 0.01], dtype=np.float64)
        mass = dyn._effective_mass(base_mass=1.5, force=force)
        self.assertEqual(mass.shape[0], force.shape[0])
        self.assertGreater(mass[1], mass[0])


if __name__ == "__main__":
    unittest.main()
