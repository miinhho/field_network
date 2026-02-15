from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FEATURE_ORDER = [
    "social_entropy",
    "temporal_regularity",
    "spatial_range",
    "schedule_density",
    "network_centrality",
    "transition_speed",
]


@dataclass(slots=True)
class DynamicsSnapshot:
    state: dict[str, float]
    velocity: dict[str, float]
    attractor_distance: float


@dataclass(slots=True)
class DynamicsResult:
    snapshots: list[DynamicsSnapshot]
    stabilized: bool


class FlowFieldDynamics:
    """Deterministic flow-field dynamics for PoC.

    x(t+1) = x(t) + dt * (A + R + S + T) * (1 - viscosity)

    - A: pull toward attractor
    - R: push away from repeller
    - S: perturbation shock term
    - T: turbulence term derived from historical drift
    """

    def __init__(
        self,
        attractor_strength: float = 0.45,
        repeller_strength: float = 0.25,
        turbulence_strength: float = 0.15,
    ) -> None:
        self.attractor_strength = attractor_strength
        self.repeller_strength = repeller_strength
        self.turbulence_strength = turbulence_strength

    def simulate(
        self,
        initial_state: dict[str, float],
        history: list[dict[str, float]],
        shock: dict[str, float],
        steps: int = 4,
        dt: float = 1.0,
    ) -> DynamicsResult:
        x = self._to_vector(initial_state)
        hist = [self._to_vector(state) for state in history] if history else [x.copy()]

        attractor = self._estimate_attractor(hist)
        repeller = self._estimate_repeller(hist)
        turbulence = self._estimate_turbulence(hist)
        viscosity = self._estimate_viscosity(initial_state)

        snapshots: list[DynamicsSnapshot] = []
        prev = x.copy()
        stabilized = False

        for _ in range(max(1, steps)):
            velocity = self._velocity(
                x=x,
                attractor=attractor,
                repeller=repeller,
                shock=self._to_vector(shock),
                turbulence=turbulence,
                viscosity=viscosity,
            )
            x = x + dt * velocity
            x = np.clip(x, 0.0, 10.0)

            dist = float(np.linalg.norm(x - attractor))
            snapshots.append(
                DynamicsSnapshot(
                    state=self._from_vector(x),
                    velocity=self._from_vector(velocity),
                    attractor_distance=round(dist, 6),
                )
            )

            delta = float(np.linalg.norm(x - prev))
            if delta < 0.02:
                stabilized = True
                break
            prev = x.copy()

        return DynamicsResult(snapshots=snapshots, stabilized=stabilized)

    def _velocity(
        self,
        x: np.ndarray,
        attractor: np.ndarray,
        repeller: np.ndarray,
        shock: np.ndarray,
        turbulence: np.ndarray,
        viscosity: float,
    ) -> np.ndarray:
        to_attr = self.attractor_strength * (attractor - x)
        from_repeller = self.repeller_strength * (x - repeller)
        turb = self.turbulence_strength * turbulence
        return (to_attr + from_repeller + shock + turb) * (1.0 - viscosity)

    def _estimate_attractor(self, hist: list[np.ndarray]) -> np.ndarray:
        return np.mean(np.vstack(hist), axis=0)

    def _estimate_repeller(self, hist: list[np.ndarray]) -> np.ndarray:
        return np.min(np.vstack(hist), axis=0)

    def _estimate_turbulence(self, hist: list[np.ndarray]) -> np.ndarray:
        if len(hist) < 2:
            return np.zeros(len(FEATURE_ORDER), dtype=np.float64)
        deltas = np.diff(np.vstack(hist), axis=0)
        return np.std(deltas, axis=0)

    def _estimate_viscosity(self, state: dict[str, float]) -> float:
        reg = float(state.get("temporal_regularity", 0.0))
        density = float(state.get("schedule_density", 0.0))
        density_norm = min(1.0, density / 10.0)
        return float(np.clip(0.6 * reg + 0.3 * density_norm, 0.0, 0.85))

    def _to_vector(self, values: dict[str, float]) -> np.ndarray:
        return np.array([float(values.get(key, 0.0)) for key in FEATURE_ORDER], dtype=np.float64)

    def _from_vector(self, vec: np.ndarray) -> dict[str, float]:
        return {FEATURE_ORDER[i]: float(round(vec[i], 6)) for i in range(len(FEATURE_ORDER))}
