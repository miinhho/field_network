from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from .dynamic_simulator import DynamicGraphSimulator
from .models import Perturbation


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-by-step dynamic graph simulator")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--target", type=str, default="hub")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--format", choices=("table", "json", "html"), default="table")
    parser.add_argument("--out", type=str, default="simulator_trace.html")
    args = parser.parse_args()

    sim = DynamicGraphSimulator()
    graph = sim.demo_graph()
    perturbation = Perturbation(
        perturbation_id="sim-run",
        timestamp=datetime.now(timezone.utc),
        targets=[args.target] if args.target else [],
        intensity=max(0.0, args.intensity),
        kind="simulation",
    )
    trace = sim.run(graph, perturbation, steps=max(1, args.steps), top_k=max(1, args.top_k))

    if args.format == "json":
        print(json.dumps(_to_json(trace), ensure_ascii=True, indent=2))
        return
    if args.format == "html":
        html = _to_html(trace)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"wrote_html,{args.out}")
        return
    _print_table(trace)


def _to_json(trace) -> dict:
    return {
        "frame_count": len(trace.frames),
        "layout": {k: [v[0], v[1]] for k, v in trace.layout.items()},
        "frames": [
            {
                "step": f.step,
                "objective_score": f.objective_score,
                "critical_transition_score": f.critical_transition_score,
                "early_warning_score": f.early_warning_score,
                "adjustment_scale": f.adjustment_scale,
                "planner_horizon": f.planner_horizon,
                "edit_budget": f.edit_budget,
                "node_impacts": f.node_impacts,
                "node_controls": f.node_controls,
                "node_final_values": f.node_final_values,
                "edges": [
                    {
                        "source_id": e.source_id,
                        "target_id": e.target_id,
                        "weight": e.weight,
                    }
                    for e in f.edges
                ],
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


def _to_html(trace) -> str:
    payload = json.dumps(_to_json(trace), ensure_ascii=True)
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Flow Graph Simulator Replay</title>
  <style>
    :root {{
      --bg: #0f1220;
      --panel: #171b2e;
      --ink: #e8ecff;
      --muted: #a4afd8;
      --accent: #67d3ff;
      --warn: #ffb357;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Noto Sans", sans-serif;
      background: radial-gradient(1200px 700px at 20% 0%, #1b2140, var(--bg));
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
      display: grid;
      grid-template-columns: 1.6fr 1fr;
      gap: 16px;
    }}
    .panel {{
      background: linear-gradient(180deg, #1a1f35, var(--panel));
      border: 1px solid #2a3259;
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 10px 28px rgba(0, 0, 0, 0.25);
    }}
    h1 {{
      grid-column: 1 / -1;
      margin: 0;
      font-size: 20px;
    }}
    #canvas {{
      width: 100%;
      height: 620px;
      background: #0b1022;
      border-radius: 10px;
      border: 1px solid #273058;
    }}
    .row {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 10px 0;
    }}
    input[type="range"] {{ width: 100%; }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-top: 8px;
    }}
    .kpi {{
      background: #101631;
      border: 1px solid #29325f;
      border-radius: 10px;
      padding: 8px;
    }}
    .kpi .label {{
      color: var(--muted);
      font-size: 12px;
    }}
    .kpi .val {{
      font-size: 16px;
      font-weight: 700;
    }}
    .list {{
      font-family: "IBM Plex Mono", "Consolas", monospace;
      font-size: 12px;
      line-height: 1.45;
      color: #c9d2ff;
      max-height: 200px;
      overflow: auto;
      white-space: pre-wrap;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Flow Graph Dynamic Replay</h1>
    <div class="panel">
      <canvas id="canvas" width="780" height="620"></canvas>
      <div class="row">
        <span>Step</span>
        <input id="step" type="range" min="0" max="0" value="0" />
        <strong id="stepLabel">1</strong>
      </div>
    </div>
    <div class="panel">
      <div class="kpis">
        <div class="kpi"><div class="label">Objective</div><div id="obj" class="val">-</div></div>
        <div class="kpi"><div class="label">Critical</div><div id="crit" class="val">-</div></div>
        <div class="kpi"><div class="label">Warning</div><div id="warn" class="val">-</div></div>
        <div class="kpi"><div class="label">Scale</div><div id="scale" class="val">-</div></div>
        <div class="kpi"><div class="label">Horizon</div><div id="horizon" class="val">-</div></div>
        <div class="kpi"><div class="label">Budget</div><div id="budget" class="val">-</div></div>
      </div>
      <h3>Top Final Node Values</h3>
      <div id="topNodes" class="list"></div>
      <h3>Edge Deltas</h3>
      <div id="edgeDeltas" class="list"></div>
    </div>
  </div>
  <script>
    const DATA = __PAYLOAD__;
    const frames = DATA.frames || [];
    const layout = DATA.layout || {{}};
    const nodeIds = Object.keys(layout);
    const slider = document.getElementById("step");
    slider.max = Math.max(0, frames.length - 1);
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    function mapPoint(x, y) {{
      const px = 80 + (x + 1) * 0.5 * (canvas.width - 160);
      const py = 80 + (y + 1) * 0.5 * (canvas.height - 160);
      return [px, py];
    }}

    function drawFrame(i) {{
      const f = frames[i];
      if (!f) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#0b1022";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const finalVals = f.node_final_values || {{}};
      const controls = f.node_controls || {{}};
      const edges = f.edges || [];

      // edges
      for (const e of edges) {{
        const a = layout[e.source_id];
        const b = layout[e.target_id];
        if (!a || !b) continue;
        const [x1, y1] = mapPoint(a[0], a[1]);
        const [x2, y2] = mapPoint(b[0], b[1]);
        ctx.strokeStyle = "rgba(148,165,255,0.35)";
        ctx.lineWidth = 1 + Math.min(4, e.weight * 0.9);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }}

      // nodes
      for (const n of nodeIds) {{
        const p = layout[n];
        const [x, y] = mapPoint(p[0], p[1]);
        const v = Number(finalVals[n] || 0);
        const c = Number(controls[n] || 0);
        const r = 8 + Math.min(18, v * 6);
        const hue = c >= 0 ? 195 : 28;
        const sat = 70;
        const lit = 42 + Math.min(35, Math.abs(c) * 40);
        ctx.fillStyle = `hsl(${hue} ${sat}% ${lit}%)`;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "rgba(240,245,255,0.75)";
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.fillStyle = "#ecf1ff";
        ctx.font = "12px IBM Plex Mono, monospace";
        ctx.fillText(`${n} (${v.toFixed(2)})`, x + r + 4, y + 4);
      }}

      document.getElementById("stepLabel").textContent = String(f.step);
      document.getElementById("obj").textContent = f.objective_score.toFixed(4);
      document.getElementById("crit").textContent = f.critical_transition_score.toFixed(4);
      document.getElementById("warn").textContent = f.early_warning_score.toFixed(4);
      document.getElementById("scale").textContent = f.adjustment_scale.toFixed(2);
      document.getElementById("horizon").textContent = String(f.planner_horizon);
      document.getElementById("budget").textContent = String(f.edit_budget);

      document.getElementById("topNodes").textContent = (f.top_final_nodes || [])
        .map(([n, v]) => `${{n}}: ${{Number(v).toFixed(4)}}`)
        .join("\\n");
      document.getElementById("edgeDeltas").textContent = (f.edge_deltas || [])
        .map(d => `${{d.source_id}}-${{d.target_id}}  ${{d.kind}}  ${{Number(d.old_weight).toFixed(3)}} -> ${{Number(d.new_weight).toFixed(3)}}`)
        .join("\\n");
    }}

    slider.addEventListener("input", () => drawFrame(Number(slider.value)));
    drawFrame(0);
  </script>
</body>
</html>"""
    return html.replace("__PAYLOAD__", payload)


if __name__ == "__main__":
    main()
