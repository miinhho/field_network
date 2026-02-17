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
    kinetic_energy: float


@dataclass(slots=True)
class DynamicsResult:
    snapshots: list[DynamicsSnapshot]
    stabilized: bool


class FlowFieldDynamics:
    """Deterministic flow-field dynamics for PoC.

    x(t+1) = x(t) + dt * ((A + R + S + T) / M_eff)

    - A: pull toward attractor
    - R: push away from repeller
    - S: perturbation shock term
    - T: turbulence term derived from historical drift
    - M_eff: effective mass combining state inertia and step safety
    """

    def __init__(
        self,
        attractor_strength: float = 0.45,
        repeller_strength: float = 0.25,
        turbulence_strength: float = 0.15,
        max_step_norm: float = 0.8,
    ) -> None:
        self.attractor_strength = attractor_strength
        self.repeller_strength = repeller_strength
        self.turbulence_strength = turbulence_strength
        self.max_step_norm = max_step_norm

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
        base_mass = self._estimate_base_mass(initial_state)
        shock_vec = self._to_vector(shock)

        snapshots: list[DynamicsSnapshot] = []
        prev = x.copy()
        stabilized = False

        for _ in range(max(1, steps)):
            force_1 = self._force(
                x=x,
                attractor=attractor,
                repeller=repeller,
                shock=shock_vec,
                turbulence=turbulence,
            )
            mass_1 = self._effective_mass(base_mass, force_1)
            velocity_1 = force_1 / mass_1
            midpoint = x + 0.5 * dt * velocity_1
            force_2 = self._force(
                x=midpoint,
                attractor=attractor,
                repeller=repeller,
                shock=shock_vec,
                turbulence=turbulence,
            )
            mass_2 = self._effective_mass(base_mass, force_2)
            velocity_2 = force_2 / mass_2
            velocity = 0.5 * (velocity_1 + velocity_2)
            x = x + dt * velocity
            x = np.clip(x, 0.0, 10.0)

            dist = float(np.linalg.norm(x - attractor))
            kinetic = float(np.dot(velocity, velocity))
            snapshots.append(
                DynamicsSnapshot(
                    state=self._from_vector(x),
                    velocity=self._from_vector(velocity),
                    attractor_distance=round(dist, 6),
                    kinetic_energy=round(kinetic, 6),
                )
            )

            delta = float(np.linalg.norm(x - prev))
            if delta < 0.02:
                stabilized = True
                break
            prev = x.copy()

        return DynamicsResult(snapshots=snapshots, stabilized=stabilized)

    def _force(
        self,
        x: np.ndarray,
        attractor: np.ndarray,
        repeller: np.ndarray,
        shock: np.ndarray,
        turbulence: np.ndarray,
    ) -> np.ndarray:
        to_attr = self.attractor_strength * (attractor - x)
        from_repeller = self.repeller_strength * (x - repeller)
        turb = self.turbulence_strength * turbulence
        return to_attr + from_repeller + shock + turb

    def _estimate_attractor(self, hist: list[np.ndarray]) -> np.ndarray:
        return np.mean(np.vstack(hist), axis=0)

    def _estimate_repeller(self, hist: list[np.ndarray]) -> np.ndarray:
        return np.min(np.vstack(hist), axis=0)

    def _estimate_turbulence(self, hist: list[np.ndarray]) -> np.ndarray:
        if len(hist) < 2:
            return np.zeros(len(FEATURE_ORDER), dtype=np.float64)
        deltas = np.diff(np.vstack(hist), axis=0)
        return np.std(deltas, axis=0)

    def _estimate_base_mass(self, state: dict[str, float]) -> float:
        reg = float(state.get("temporal_regularity", 0.0))
        density = float(state.get("schedule_density", 0.0))
        density_norm = min(1.0, density / 10.0)
        inertia = float(np.clip(0.6 * reg + 0.3 * density_norm, 0.0, 0.85))
        return 1.0 + 4.0 * inertia

    def _effective_mass(self, base_mass: float, force: np.ndarray) -> np.ndarray:
        dim = max(1, force.shape[0])
        per_dim_limit = self.max_step_norm / (dim ** 0.5)
        abs_force = np.abs(force)
        step_mass = np.clip(abs_force / max(1e-9, per_dim_limit), 1.0, 5.0)
        return np.maximum(1.0, base_mass * step_mass)

    def _to_vector(self, values: dict[str, float]) -> np.ndarray:
        return np.array([float(values.get(key, 0.0)) for key in FEATURE_ORDER], dtype=np.float64)

    def _from_vector(self, vec: np.ndarray) -> dict[str, float]:
        return {FEATURE_ORDER[i]: float(round(vec[i], 6)) for i in range(len(FEATURE_ORDER))}
