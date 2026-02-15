from __future__ import annotations

import argparse

from .calibration import run_calibration_with_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline planner calibration scenarios")
    parser.add_argument("--scenarios", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", choices=("default", "plasticity", "mixed"), default="mixed")
    parser.add_argument("--summary-out", type=str, default="")
    args = parser.parse_args()

    rows, summary = run_calibration_with_summary(
        num_scenarios=args.scenarios,
        seed=args.seed,
        batch=args.batch,
    )
    print(
        "rank,config_id,score,avg_adjustment_objective,avg_critical_transition,avg_coupling_penalty,avg_applied_edits,avg_converged,avg_supervisory_confusion,avg_supervisory_forgetting"
    )
    for i, r in enumerate(rows, start=1):
        print(
            f"{i},{r.config_id},{r.score:.6f},{r.avg_adjustment_objective:.6f},{r.avg_critical_transition:.6f},"
            f"{r.avg_coupling_penalty:.6f},{r.avg_applied_edits:.6f},{r.avg_converged:.6f},"
            f"{r.avg_supervisory_confusion:.6f},{r.avg_supervisory_forgetting:.6f}"
        )
    print(
        "summary,batch,candidate_count,top_count,eta_up_min,eta_up_max,eta_down_min,eta_down_max,theta_on_min,theta_on_max,theta_off_min,theta_off_max,dwell_min,dwell_max,risk_w_min,risk_w_max,vol_w_min,vol_w_max"
    )
    print(
        "summary,"
        f"{summary.batch},{summary.candidate_count},{summary.top_count},"
        f"{summary.eta_up_min:.6f},{summary.eta_up_max:.6f},"
        f"{summary.eta_down_min:.6f},{summary.eta_down_max:.6f},"
        f"{summary.theta_on_min:.6f},{summary.theta_on_max:.6f},"
        f"{summary.theta_off_min:.6f},{summary.theta_off_max:.6f},"
        f"{summary.hysteresis_dwell_min},{summary.hysteresis_dwell_max},"
        f"{summary.risk_weight_base_min:.6f},{summary.risk_weight_base_max:.6f},"
        f"{summary.volatility_weight_base_min:.6f},{summary.volatility_weight_base_max:.6f}"
    )
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as f:
            f.write(
                "batch,candidate_count,top_count,eta_up_min,eta_up_max,eta_down_min,eta_down_max,"
                "theta_on_min,theta_on_max,theta_off_min,theta_off_max,hysteresis_dwell_min,hysteresis_dwell_max,"
                "risk_weight_base_min,risk_weight_base_max,volatility_weight_base_min,volatility_weight_base_max\n"
            )
            f.write(
                f"{summary.batch},{summary.candidate_count},{summary.top_count},"
                f"{summary.eta_up_min:.6f},{summary.eta_up_max:.6f},"
                f"{summary.eta_down_min:.6f},{summary.eta_down_max:.6f},"
                f"{summary.theta_on_min:.6f},{summary.theta_on_max:.6f},"
                f"{summary.theta_off_min:.6f},{summary.theta_off_max:.6f},"
                f"{summary.hysteresis_dwell_min},{summary.hysteresis_dwell_max},"
                f"{summary.risk_weight_base_min:.6f},{summary.risk_weight_base_max:.6f},"
                f"{summary.volatility_weight_base_min:.6f},{summary.volatility_weight_base_max:.6f}\n"
            )


if __name__ == "__main__":
    main()
