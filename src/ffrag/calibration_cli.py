from __future__ import annotations

import argparse

from .calibration import run_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline planner calibration scenarios")
    parser.add_argument("--scenarios", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = run_calibration(num_scenarios=args.scenarios, seed=args.seed)
    print(
        "rank,config_id,score,avg_adjustment_objective,avg_critical_transition,avg_coupling_penalty,avg_applied_edits,avg_converged"
    )
    for i, r in enumerate(rows, start=1):
        print(
            f"{i},{r.config_id},{r.score:.6f},{r.avg_adjustment_objective:.6f},{r.avg_critical_transition:.6f},"
            f"{r.avg_coupling_penalty:.6f},{r.avg_applied_edits:.6f},{r.avg_converged:.6f}"
        )


if __name__ == "__main__":
    main()
