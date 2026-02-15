from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TransitionAnalysis:
    states: list[str]
    transition_counts: dict[str, dict[str, int]]
    transition_matrix: dict[str, dict[str, float]]
    transition_triggers: list[str]
    attractor_basin_state: str | None


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

    def classify_state(self, values: dict[str, float]) -> str:
        regularity = float(values.get("temporal_regularity", 0.0))
        speed = float(values.get("transition_speed", 0.0))
        density = float(values.get("schedule_density", 0.0))

        if regularity >= 0.7 and speed <= 0.5:
            return "stable_routine"
        if speed >= 0.7 or density >= 6.0:
            return "high_flux"
        return "adaptive"

    def transition_analysis(self, trajectory: list[dict[str, float]]) -> TransitionAnalysis:
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
        return TransitionAnalysis(
            states=labels,
            transition_counts=counts,
            transition_matrix=matrix,
            transition_triggers=triggers,
            attractor_basin_state=basin,
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
            if speed_jump > 0.15:
                reasons.append("speed_jump")
            if reg_drop > 0.15:
                reasons.append("regularity_drop")
            if density_jump > 0.5:
                reasons.append("density_jump")
            reason_text = ",".join(reasons) if reasons else "state_boundary_crossed"
            triggers.append(f"t{i}->{i+1}:{reason_text}")
        return triggers

    def _dominant_state(self, labels: list[str]) -> str | None:
        if not labels:
            return None
        counts: dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        return max(counts.items(), key=lambda item: item[1])[0]
