from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PhaseTransitionResult:
    critical_transition_score: float
    early_warning_score: float
    regime_switch_count: int
    regime_persistence_score: float
    coherence_break_score: float
    critical_slowing_score: float
    hysteresis_proxy_score: float
    sign_flip_rate: float
    polarity_coherence_score: float
    dominant_regime: str


class PhaseTransitionAnalyzer:
    """Detects phase-transition risk from core cycle time series."""

    def __init__(
        self,
        critical_threshold: float = 0.6,
        warning_derivative_threshold: float = 0.12,
        warning_variance_threshold: float = 0.03,
    ) -> None:
        self.critical_threshold = critical_threshold
        self.warning_derivative_threshold = warning_derivative_threshold
        self.warning_variance_threshold = warning_variance_threshold

    def analyze(
        self,
        trajectory: list[dict[str, float]],
        attractor_distances: list[float],
        objective_scores: list[float],
        topological_tensions: list[float],
    ) -> PhaseTransitionResult:
        n = max(len(trajectory), len(attractor_distances), len(objective_scores), len(topological_tensions))
        if n == 0:
            return PhaseTransitionResult(
                critical_transition_score=0.0,
                early_warning_score=0.0,
                regime_switch_count=0,
                regime_persistence_score=0.0,
                coherence_break_score=0.0,
                critical_slowing_score=0.0,
                hysteresis_proxy_score=0.0,
                sign_flip_rate=0.0,
                polarity_coherence_score=1.0,
                dominant_regime="calm",
            )

        speed_series = [float(point.get("transition_speed", 0.0)) for point in trajectory]
        speed_series = self._align(speed_series, n)
        dist_series = self._align([float(v) for v in attractor_distances], n)
        tension_series = self._align([float(v) for v in topological_tensions], n)
        objective_series = self._align([float(v) for v in objective_scores], n)

        speed_n = self._normalize(speed_series)
        dist_n = self._normalize(dist_series)
        tension_n = self._normalize(tension_series)
        objective_n = self._normalize(objective_series)

        # Order parameter for phase intensity.
        order = [
            0.45 * speed_n[i] + 0.35 * dist_n[i] + 0.2 * tension_n[i]
            for i in range(n)
        ]
        deriv = [abs(order[i] - order[i - 1]) for i in range(1, n)]
        obj_deriv = [abs(objective_n[i] - objective_n[i - 1]) for i in range(1, n)]

        order_var = self._variance(order)
        max_deriv = max(deriv) if deriv else 0.0
        mean_obj_deriv = (sum(obj_deriv) / len(obj_deriv)) if obj_deriv else 0.0

        critical_slowing = self._critical_slowing(order)
        hysteresis_proxy = self._hysteresis_proxy(order, objective_n)
        sign_flip_rate = self._sign_flip_rate(order)
        polarity_coherence = self._polarity_coherence(order, objective_n, tension_n)
        polarity_incoherence = 1.0 - polarity_coherence
        critical = self._clip(
            0.35 * max_deriv
            + 0.19 * order_var
            + 0.14 * mean_obj_deriv
            + 0.12 * critical_slowing
            + 0.08 * hysteresis_proxy
            + 0.07 * sign_flip_rate
            + 0.05 * polarity_incoherence
        )
        regime_labels = [self._regime(v) for v in order]
        switches = 0
        for i in range(1, len(regime_labels)):
            if regime_labels[i] != regime_labels[i - 1]:
                switches += 1

        dominant_regime = self._dominant(regime_labels)
        persistence = self._regime_persistence(regime_labels, dominant_regime)
        coherence_break = self._coherence_break(dist_n, objective_n, tension_n)

        local_var = self._variance(order[-3:]) if len(order) >= 3 else order_var
        local_deriv = max(deriv[-2:]) if len(deriv) >= 2 else (deriv[-1] if deriv else 0.0)
        warn = 0.0
        if local_deriv > self.warning_derivative_threshold:
            warn += min(0.5, local_deriv)
        if local_var > self.warning_variance_threshold:
            warn += min(0.3, local_var * 2.0)
        if tension_n and tension_n[-1] > 0.65:
            warn += 0.2
        warn += min(0.15, 0.25 * sign_flip_rate)
        warn += min(0.1, 0.2 * polarity_incoherence)
        warning = self._clip(warn)

        if critical >= self.critical_threshold:
            dominant_regime = "critical"

        return PhaseTransitionResult(
            critical_transition_score=round(critical, 6),
            early_warning_score=round(warning, 6),
            regime_switch_count=switches,
            regime_persistence_score=round(persistence, 6),
            coherence_break_score=round(coherence_break, 6),
            critical_slowing_score=round(critical_slowing, 6),
            hysteresis_proxy_score=round(hysteresis_proxy, 6),
            sign_flip_rate=round(sign_flip_rate, 6),
            polarity_coherence_score=round(polarity_coherence, 6),
            dominant_regime=dominant_regime,
        )

    def _normalize(self, values: list[float]) -> list[float]:
        if not values:
            return []
        lo = min(values)
        hi = max(values)
        if hi - lo <= 1e-9:
            return [0.0 for _ in values]
        return [(v - lo) / (hi - lo) for v in values]

    def _align(self, values: list[float], n: int) -> list[float]:
        if n <= 0:
            return []
        if not values:
            return [0.0] * n
        if len(values) >= n:
            return values[:n]
        return values + [values[-1]] * (n - len(values))

    def _variance(self, values: list[float]) -> float:
        if not values:
            return 0.0
        mean_v = sum(values) / len(values)
        return sum((v - mean_v) ** 2 for v in values) / len(values)

    def _regime(self, order_value: float) -> str:
        if order_value < 0.33:
            return "calm"
        if order_value < 0.66:
            return "adaptive"
        return "turbulent"

    def _dominant(self, labels: list[str]) -> str:
        if not labels:
            return "calm"
        counts: dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        return max(counts.items(), key=lambda item: item[1])[0]

    def _coherence_break(
        self,
        dist_norm: list[float],
        objective_norm: list[float],
        tension_norm: list[float],
    ) -> float:
        n = min(len(dist_norm), len(objective_norm), len(tension_norm))
        if n <= 1:
            return 0.0
        mismatch = 0.0
        for i in range(1, n):
            d_dist = dist_norm[i] - dist_norm[i - 1]
            d_obj = objective_norm[i] - objective_norm[i - 1]
            d_tension = tension_norm[i] - tension_norm[i - 1]
            if d_dist * d_obj < 0:
                mismatch += 1.0
            if d_obj * d_tension < 0:
                mismatch += 0.5
        return self._clip(mismatch / (1.5 * (n - 1)))

    def _regime_persistence(self, labels: list[str], dominant: str) -> float:
        if not labels:
            return 0.0
        longest = 1
        current = 1
        for i in range(1, len(labels)):
            if labels[i] == labels[i - 1]:
                current += 1
            else:
                longest = max(longest, current)
                current = 1
        longest = max(longest, current)
        dominant_occ = sum(1 for x in labels if x == dominant) / len(labels)
        # Blend temporal continuity (longest run) with occupancy dominance.
        return self._clip(0.6 * (longest / len(labels)) + 0.4 * dominant_occ)

    def _clip(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _critical_slowing(self, order: list[float]) -> float:
        if len(order) < 3:
            return 0.0
        ac1 = self._lag1_autocorr(order)
        rolling_var: list[float] = []
        for i in range(2, len(order)):
            window = order[max(0, i - 2) : i + 1]
            rolling_var.append(self._variance(window))
        var_trend = max(0.0, rolling_var[-1] - rolling_var[0]) if len(rolling_var) >= 2 else (rolling_var[-1] if rolling_var else 0.0)
        return self._clip(0.65 * ac1 + 0.35 * min(1.0, var_trend * 4.0))

    def _hysteresis_proxy(self, order: list[float], objective_norm: list[float]) -> float:
        n = min(len(order), len(objective_norm))
        if n < 3:
            return 0.0
        mismatch = 0.0
        for i in range(1, n):
            d_order = order[i] - order[i - 1]
            d_obj = objective_norm[i] - objective_norm[i - 1]
            if d_order * d_obj < 0:
                mismatch += 1.0
        return self._clip(mismatch / (n - 1))

    def _lag1_autocorr(self, values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        x0 = values[:-1]
        x1 = values[1:]
        m0 = sum(x0) / len(x0)
        m1 = sum(x1) / len(x1)
        num = sum((a - m0) * (b - m1) for a, b in zip(x0, x1, strict=False))
        den0 = sum((a - m0) ** 2 for a in x0)
        den1 = sum((b - m1) ** 2 for b in x1)
        den = (den0 * den1) ** 0.5
        if den <= 1e-9:
            return 0.0
        corr = num / den
        return max(0.0, min(1.0, (corr + 1.0) * 0.5))

    def _sign_flip_rate(self, values: list[float]) -> float:
        if len(values) < 3:
            return 0.0
        prev_sign = 0
        flips = 0
        opportunities = 0
        for i in range(1, len(values)):
            d = values[i] - values[i - 1]
            sign = 1 if d > 1e-9 else (-1 if d < -1e-9 else 0)
            if sign == 0:
                continue
            if prev_sign != 0:
                opportunities += 1
                if sign != prev_sign:
                    flips += 1
            prev_sign = sign
        if opportunities <= 0:
            return 0.0
        return self._clip(flips / opportunities)

    def _polarity_coherence(self, order: list[float], objective: list[float], tension: list[float]) -> float:
        n = min(len(order), len(objective), len(tension))
        if n < 2:
            return 1.0
        scores: list[float] = []
        for i in range(1, n):
            signs: list[int] = []
            for series in (order, objective, tension):
                d = series[i] - series[i - 1]
                if d > 1e-9:
                    signs.append(1)
                elif d < -1e-9:
                    signs.append(-1)
            if not signs:
                scores.append(1.0)
                continue
            pos = sum(1 for s in signs if s > 0)
            neg = sum(1 for s in signs if s < 0)
            scores.append(max(pos, neg) / len(signs))
        if not scores:
            return 1.0
        return self._clip(sum(scores) / len(scores))
