from __future__ import annotations

from dataclasses import dataclass
import math

FEATURE_KEYS = (
    "social_entropy",
    "temporal_regularity",
    "spatial_range",
    "schedule_density",
    "network_centrality",
    "transition_speed",
)


@dataclass(slots=True)
class FlowAnalyzerConfig:
    basin_strategy: str = "centroid_std"
    basin_quantile: float = 0.8
    trigger_speed_jump_threshold: float = 0.15
    trigger_regularity_drop_threshold: float = 0.15
    trigger_density_jump_threshold: float = 0.5
    trigger_speed_weight: float = 1.8
    trigger_regularity_weight: float = 1.6
    trigger_density_weight: float = 0.6
    causal_lag_steps: int = 0


@dataclass(slots=True)
class TransitionAnalysis:
    states: list[str]
    transition_counts: dict[str, dict[str, int]]
    transition_matrix: dict[str, dict[str, float]]
    transition_triggers: list[str]
    attractor_basin_state: str | None
    attractor_basin_radius: float
    basin_occupancy: float
    avg_trigger_confidence: float
    causal_alignment_score: float


@dataclass(slots=True)
class ResilienceAnalysis:
    recovery_rate: float
    recovery_steps: int
    hysteresis_index: float
    overshoot_index: float
    settling_time: int
    path_efficiency: float


class FlowDynamicsAnalyzer:
    """Analyzes trajectory snapshots into transition and resilience metrics."""

    def __init__(self, config: FlowAnalyzerConfig | None = None) -> None:
        self.config = config or FlowAnalyzerConfig()

    def classify_state(self, values: dict[str, float]) -> str:
        regularity = float(values.get("temporal_regularity", 0.0))
        speed = float(values.get("transition_speed", 0.0))
        density = float(values.get("schedule_density", 0.0))

        if regularity >= 0.7 and speed <= 0.5:
            return "stable_routine"
        if speed >= 0.7 or density >= 6.0:
            return "high_flux"
        return "adaptive"

    def transition_analysis(
        self,
        trajectory: list[dict[str, float]],
        shock: dict[str, float] | None = None,
    ) -> TransitionAnalysis:
        labels = [self.classify_state(point) for point in trajectory]
        counts: dict[str, dict[str, int]] = {}

        for i in range(len(labels) - 1):
            src = labels[i]
            dst = labels[i + 1]
            if src not in counts:
                counts[src] = {}
            counts[src][dst] = counts[src].get(dst, 0) + 1

        matrix: dict[str, dict[str, float]] = {}
        for src, row in counts.items():
            total = sum(row.values())
            matrix[src] = {dst: round(cnt / total, 6) for dst, cnt in row.items()} if total else {}

        triggers = self._detect_transition_triggers(trajectory, labels)
        basin = self._dominant_state(labels)
        basin_radius, basin_occupancy = self._estimate_basin_boundary(trajectory, labels, basin)
        avg_confidence = self._average_trigger_confidence(triggers)
        causal_alignment = self._causal_alignment_score(trajectory, shock or {})
        return TransitionAnalysis(
            states=labels,
            transition_counts=counts,
            transition_matrix=matrix,
            transition_triggers=triggers,
            attractor_basin_state=basin,
            attractor_basin_radius=round(basin_radius, 6),
            basin_occupancy=round(basin_occupancy, 6),
            avg_trigger_confidence=round(avg_confidence, 6),
            causal_alignment_score=round(causal_alignment, 6),
        )

    def resilience_analysis(
        self,
        attractor_distances: list[float],
        baseline_distances: list[float] | None = None,
    ) -> ResilienceAnalysis:
        if not attractor_distances:
            return ResilienceAnalysis(
                recovery_rate=0.0,
                recovery_steps=0,
                hysteresis_index=0.0,
                overshoot_index=0.0,
                settling_time=0,
                path_efficiency=0.0,
            )

        first = attractor_distances[0]
        last = attractor_distances[-1]
        denom = max(1e-6, abs(first))
        recovery_rate = max(0.0, min(1.0, (first - last) / denom))

        recovery_steps = len(attractor_distances)
        for idx, value in enumerate(attractor_distances):
            if value <= first * 0.5:
                recovery_steps = idx + 1
                break

        hysteresis = self._hysteresis_index(attractor_distances, baseline_distances)
        overshoot = self._overshoot_index(attractor_distances)
        settling = self._settling_time(attractor_distances)
        path_eff = self._path_efficiency(attractor_distances)
        return ResilienceAnalysis(
            recovery_rate=round(recovery_rate, 6),
            recovery_steps=recovery_steps,
            hysteresis_index=round(hysteresis, 6),
            overshoot_index=round(overshoot, 6),
            settling_time=settling,
            path_efficiency=round(path_eff, 6),
        )

    def _hysteresis_index(
        self,
        forward: list[float],
        backward: list[float] | None,
    ) -> float:
        if not backward:
            return 0.0
        n = min(len(forward), len(backward))
        if n == 0:
            return 0.0
        total = 0.0
        for i in range(n):
            total += abs(forward[i] - backward[i])
        return total / n

    def _overshoot_index(self, values: list[float]) -> float:
        if not values:
            return 0.0
        start = values[0]
        peak = max(values)
        return max(0.0, peak - start)

    def _settling_time(self, values: list[float], tol_ratio: float = 0.05) -> int:
        if not values:
            return 0
        final = values[-1]
        tol = max(1e-6, abs(final) * tol_ratio)
        for i in range(len(values)):
            tail = values[i:]
            if all(abs(v - final) <= tol for v in tail):
                return i + 1
        return len(values)

    def _path_efficiency(self, values: list[float]) -> float:
        if len(values) < 2:
            return 1.0 if values else 0.0
        direct = abs(values[0] - values[-1])
        traversed = 0.0
        for i in range(len(values) - 1):
            traversed += abs(values[i + 1] - values[i])
        if traversed <= 1e-9:
            return 1.0
        return min(1.0, direct / traversed)

    def _detect_transition_triggers(
        self,
        trajectory: list[dict[str, float]],
        labels: list[str],
    ) -> list[str]:
        if len(trajectory) < 2 or len(labels) < 2:
            return []
        triggers: list[str] = []
        for i in range(len(trajectory) - 1):
            if labels[i] == labels[i + 1]:
                continue
            prev = trajectory[i]
            cur = trajectory[i + 1]
            speed_jump = cur.get("transition_speed", 0.0) - prev.get("transition_speed", 0.0)
            reg_drop = prev.get("temporal_regularity", 0.0) - cur.get("temporal_regularity", 0.0)
            density_jump = cur.get("schedule_density", 0.0) - prev.get("schedule_density", 0.0)
            reasons: list[str] = []
            if speed_jump > self.config.trigger_speed_jump_threshold:
                reasons.append("speed_jump")
            if reg_drop > self.config.trigger_regularity_drop_threshold:
                reasons.append("regularity_drop")
            if density_jump > self.config.trigger_density_jump_threshold:
                reasons.append("density_jump")
            reason_text = ",".join(reasons) if reasons else "state_boundary_crossed"
            confidence = self._trigger_confidence(speed_jump, reg_drop, density_jump)
            triggers.append(f"t{i}->{i+1}:{reason_text}|c={confidence:.3f}")
        return triggers

    def _dominant_state(self, labels: list[str]) -> str | None:
        if not labels:
            return None
        counts: dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        return max(counts.items(), key=lambda item: item[1])[0]

    def _trigger_confidence(self, speed_jump: float, reg_drop: float, density_jump: float) -> float:
        # Weighted by theorized transition drivers: speed-up, regularity drop, density jump.
        raw = (
            max(0.0, speed_jump) * self.config.trigger_speed_weight
            + max(0.0, reg_drop) * self.config.trigger_regularity_weight
            + max(0.0, density_jump) * self.config.trigger_density_weight
        )
        return max(0.0, min(1.0, raw))

    def _estimate_basin_boundary(
        self,
        trajectory: list[dict[str, float]],
        labels: list[str],
        dominant_state: str | None,
    ) -> tuple[float, float]:
        if not dominant_state or not trajectory or not labels:
            return 0.0, 0.0

        state_points = [trajectory[i] for i in range(len(labels)) if labels[i] == dominant_state]
        if not state_points:
            return 0.0, 0.0

        centroid = self._centroid(state_points)
        distances = [self._euclidean(self._vectorize(point), centroid) for point in state_points]
        if not distances:
            return 0.0, 0.0

        radius = self._basin_radius(distances)
        in_basin = 0
        for point in trajectory:
            if self._euclidean(self._vectorize(point), centroid) <= radius:
                in_basin += 1
        occupancy = in_basin / max(1, len(trajectory))
        return radius, occupancy

    def _basin_radius(self, distances: list[float]) -> float:
        if not distances:
            return 0.0
        if self.config.basin_strategy == "quantile":
            sorted_d = sorted(distances)
            q = max(0.0, min(1.0, self.config.basin_quantile))
            idx = int(round((len(sorted_d) - 1) * q))
            return sorted_d[idx]
        mean_dist = sum(distances) / len(distances)
        variance = sum((d - mean_dist) ** 2 for d in distances) / len(distances)
        return mean_dist + math.sqrt(max(0.0, variance))

    def _average_trigger_confidence(self, triggers: list[str]) -> float:
        if not triggers:
            return 0.0
        vals: list[float] = []
        for item in triggers:
            marker = "|c="
            pos = item.rfind(marker)
            if pos == -1:
                continue
            try:
                vals.append(float(item[pos + len(marker) :]))
            except ValueError:
                continue
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    def _causal_alignment_score(self, trajectory: list[dict[str, float]], shock: dict[str, float]) -> float:
        if len(trajectory) < 2 or not shock:
            return 0.0

        lag = max(0, self.config.causal_lag_steps)
        delta_sum = {key: 0.0 for key in FEATURE_KEYS}
        start = min(len(trajectory) - 1, lag)
        for i in range(start, len(trajectory) - 1):
            prev = trajectory[i]
            cur = trajectory[i + 1]
            for key in FEATURE_KEYS:
                delta_sum[key] += float(cur.get(key, 0.0) - prev.get(key, 0.0))

        total_weight = 0.0
        aligned = 0.0
        for key in FEATURE_KEYS:
            s = float(shock.get(key, 0.0))
            if abs(s) < 1e-9:
                continue
            d = delta_sum.get(key, 0.0)
            weight = abs(s)
            total_weight += weight
            if s * d > 0:
                aligned += weight

        if total_weight <= 1e-9:
            return 0.0
        return aligned / total_weight

    def _vectorize(self, point: dict[str, float]) -> list[float]:
        return [float(point.get(key, 0.0)) for key in FEATURE_KEYS]

    def _centroid(self, points: list[dict[str, float]]) -> list[float]:
        vecs = [self._vectorize(point) for point in points]
        dims = len(FEATURE_KEYS)
        if not vecs:
            return [0.0] * dims
        out = [0.0] * dims
        for vec in vecs:
            for i in range(dims):
                out[i] += vec[i]
        return [value / len(vecs) for value in out]

    def _euclidean(self, a: list[float], b: list[float]) -> float:
        total = 0.0
        for i in range(len(a)):
            diff = a[i] - b[i]
            total += diff * diff
        return math.sqrt(total)
