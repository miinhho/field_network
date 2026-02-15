from __future__ import annotations

from .calibration import CalibrationRow, CalibrationSummary


def calibration_rows_csv(rows: list[CalibrationRow]) -> str:
    header = (
        "rank,config_id,score,avg_adjustment_objective,avg_critical_transition,avg_coupling_penalty,"
        "avg_applied_edits,avg_converged,avg_supervisory_confusion,avg_supervisory_forgetting,"
        "avg_longrun_churn,avg_longrun_retention,avg_longrun_diversity"
    )
    lines = [header]
    for i, r in enumerate(rows, start=1):
        lines.append(
            f"{i},{r.config_id},{r.score:.6f},{r.avg_adjustment_objective:.6f},{r.avg_critical_transition:.6f},"
            f"{r.avg_coupling_penalty:.6f},{r.avg_applied_edits:.6f},{r.avg_converged:.6f},"
            f"{r.avg_supervisory_confusion:.6f},{r.avg_supervisory_forgetting:.6f},"
            f"{r.avg_longrun_churn:.6f},{r.avg_longrun_retention:.6f},{r.avg_longrun_diversity:.6f}"
        )
    return "\n".join(lines) + "\n"


def calibration_summary_csv(summary: CalibrationSummary) -> str:
    header = (
        "batch,candidate_count,top_count,eta_up_min,eta_up_max,eta_down_min,eta_down_max,"
        "theta_on_min,theta_on_max,theta_off_min,theta_off_max,hysteresis_dwell_min,hysteresis_dwell_max,"
        "risk_weight_base_min,risk_weight_base_max,volatility_weight_base_min,volatility_weight_base_max"
    )
    row = (
        f"{summary.batch},{summary.candidate_count},{summary.top_count},"
        f"{summary.eta_up_min:.6f},{summary.eta_up_max:.6f},"
        f"{summary.eta_down_min:.6f},{summary.eta_down_max:.6f},"
        f"{summary.theta_on_min:.6f},{summary.theta_on_max:.6f},"
        f"{summary.theta_off_min:.6f},{summary.theta_off_max:.6f},"
        f"{summary.hysteresis_dwell_min},{summary.hysteresis_dwell_max},"
        f"{summary.risk_weight_base_min:.6f},{summary.risk_weight_base_max:.6f},"
        f"{summary.volatility_weight_base_min:.6f},{summary.volatility_weight_base_max:.6f}"
    )
    return f"{header}\n{row}\n"


def calibration_markdown_report(rows: list[CalibrationRow], summary: CalibrationSummary, top_k: int = 5) -> str:
    top = rows[: max(1, top_k)]
    lines: list[str] = []
    lines.append("# Calibration Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(
        f"- Batch: `{summary.batch}` | Candidates: `{summary.candidate_count}` | Top Window: `{summary.top_count}`"
    )
    lines.append(
        "- Recommended Range: "
        f"`eta_up={summary.eta_up_min:.3f}..{summary.eta_up_max:.3f}`, "
        f"`eta_down={summary.eta_down_min:.3f}..{summary.eta_down_max:.3f}`, "
        f"`theta_on={summary.theta_on_min:.3f}..{summary.theta_on_max:.3f}`, "
        f"`theta_off={summary.theta_off_min:.3f}..{summary.theta_off_max:.3f}`, "
        f"`dwell={summary.hysteresis_dwell_min}..{summary.hysteresis_dwell_max}`"
    )
    lines.append("")
    lines.append("## Top Candidates")
    lines.append(
        "| Rank | Config | Score | Obj | Critical | Coupling | Churn(LR) | Retention(LR) | Diversity(LR) |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(top, start=1):
        lines.append(
            f"| {i} | `{r.config_id}` | {r.score:.4f} | {r.avg_adjustment_objective:.4f} | "
            f"{r.avg_critical_transition:.4f} | {r.avg_coupling_penalty:.4f} | "
            f"{r.avg_longrun_churn:.4f} | {r.avg_longrun_retention:.4f} | {r.avg_longrun_diversity:.4f} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- Lower `Score` is better.")
    lines.append("- Long-run (LR) metrics summarize repeated-cycle guardrail behavior.")
    return "\n".join(lines) + "\n"
