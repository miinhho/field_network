from __future__ import annotations

import argparse
import json
from datetime import datetime

from .dynamic_simulator import DynamicGraphSimulator
from .models import Perturbation


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-by-step dynamic graph simulator")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--target", type=str, default="hub")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--format", choices=("table", "json"), default="table")
    args = parser.parse_args()

    sim = DynamicGraphSimulator()
    graph = sim.demo_graph()
    perturbation = Perturbation(
        perturbation_id="sim-run",
        timestamp=datetime.utcnow(),
        targets=[args.target] if args.target else [],
        intensity=max(0.0, args.intensity),
        kind="simulation",
    )
    trace = sim.run(graph, perturbation, steps=max(1, args.steps), top_k=max(1, args.top_k))

    if args.format == "json":
        print(json.dumps(_to_json(trace), ensure_ascii=True, indent=2))
        return
    _print_table(trace)


def _to_json(trace) -> dict:
    return {
        "frame_count": len(trace.frames),
        "frames": [
            {
                "step": f.step,
                "objective_score": f.objective_score,
                "critical_transition_score": f.critical_transition_score,
                "early_warning_score": f.early_warning_score,
                "adjustment_scale": f.adjustment_scale,
                "planner_horizon": f.planner_horizon,
                "edit_budget": f.edit_budget,
                "top_impacts": f.top_impacts,
                "top_controls": f.top_controls,
                "top_final_nodes": f.top_final_nodes,
                "edge_deltas": [
                    {
                        "source_id": d.source_id,
                        "target_id": d.target_id,
                        "kind": d.kind,
                        "old_weight": d.old_weight,
                        "new_weight": d.new_weight,
                    }
                    for d in f.edge_deltas
                ],
            }
            for f in trace.frames
        ],
    }


def _print_table(trace) -> None:
    print("step  objective  critical  warning  scale  horiz  budget  top_final")
    for f in trace.frames:
        top = ",".join(f"{nid}:{v:.2f}" for nid, v in f.top_final_nodes[:3])
        print(
            f"{f.step:>4}  {f.objective_score:>9.4f}  {f.critical_transition_score:>8.4f}  "
            f"{f.early_warning_score:>7.4f}  {f.adjustment_scale:>5.2f}  {f.planner_horizon:>5}  "
            f"{f.edit_budget:>6}  {top}"
        )
        if f.edge_deltas:
            changes = "; ".join(
                f"{d.source_id}-{d.target_id}:{d.kind}:{d.old_weight:.2f}->{d.new_weight:.2f}" for d in f.edge_deltas[:3]
            )
            print(f"      edge_deltas: {changes}")


if __name__ == "__main__":
    main()
