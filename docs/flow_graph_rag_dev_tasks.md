# Flow Graph RAG Development Tasks

## Purpose

This file is persistent memory for Codex and collaborators.
It defines the implementation task breakdown, ordering, and deliverables for the current design.

## Execution Model

- Priority order: P0 -> P1 -> P2
- Each task must produce a concrete artifact (code/spec/test/report)
- Keep all architecture and scope updates synchronized with:
  - `docs/flow_graph_rag_design_memory.md`
- Do not run comparative benchmark claims until core flow dynamics criteria are met.

## P0: Foundation (Must-Have for MVP)

1. Repository skeleton and packaging
- Deliverables:
  - `pyproject.toml`
  - `uv` workflow (`uv venv`, `uv sync`) and `.venv` usage
  - `src/ffrag/` package layout
  - basic CI/test scaffold
- Acceptance:
  - package imports cleanly
  - unit test runner executes

2. Core schema/types
- Deliverables:
  - typed models for `Actant`, `Interaction`, `LayeredGraph`, `StateVector`, `Perturbation`, `PropagationResult`
  - schema versioning field and validation rules
- Acceptance:
  - valid/invalid payload tests pass

3. Query taxonomy and router
- Deliverables:
  - query classifier for `describe/predict/intervene`
  - routing policy config
- Acceptance:
  - sample queries route to expected pipeline branch

4. Baseline graph retrieval path
- Deliverables:
  - minimal GraphRAG-style retrieval interface
  - local/global retrieval strategy abstraction
- Acceptance:
  - returns evidence nodes/edges for describe queries

5. v1 flow simulation engine
- Deliverables:
  - state vector builder
  - propagation operator (hop-based attenuation)
  - simple rewiring rule and stabilization check
- Acceptance:
  - deterministic simulation test on synthetic graph

6. Response schema + guardrails
- Deliverables:
  - output JSON schema with `claims`, `evidence_ids`, `metrics_used`, `uncertainty`
  - rule to reject motivation/personality inference language in system outputs
- Acceptance:
  - integration tests verify structured outputs for all 3 query classes

## P1: Robustness and Evaluation

7. Synthetic benchmark harness
- Deliverables:
  - graph/event generator with perturbation scenarios
  - baseline comparison runner (Plain RAG vs GraphRAG vs Flow Graph RAG)
- Acceptance:
  - reproducible benchmark report generated from CLI
  - enabled only after core dynamics readiness checklist is satisfied

8. Metrics and observability
- Deliverables:
  - latency/cost/quality metrics pipeline
  - run metadata logging (config hash, dataset version, model version)
- Acceptance:
  - each evaluation run produces traceable metrics artifacts

9. Incremental update pipeline
- Deliverables:
  - append-only event ingestion
  - soft delete/tombstone support
  - periodic compaction/reindex hooks
- Acceptance:
  - updates do not require full rebuild for common cases

10. Multi-domain adapters (at least two)
- Deliverables:
  - adapter A (calendar/events)
  - adapter B (non-calendar domain: logs/supply-chain/community)
- Acceptance:
  - same query API runs on both domains
  - adapters pass canonical contract validation and emit mapping report

## P2: Advanced Capabilities (Post-MVP)

11. Topology module
- Deliverables:
  - optional persistent/zigzag summaries integrated into analysis pipeline
- Acceptance:
  - added metrics appear in `predict/intervene` evidence

12. Edge-flow/Hodge analysis module
- Deliverables:
  - optional edge-flow decomposition output for interpretability
- Acceptance:
  - decomposition artifacts linked in structured responses

13. Performance scaling path
- Deliverables:
  - optional `igraph` backend
  - large-graph benchmark profile
- Acceptance:
  - documented scaling breakpoints and migration guide

14. Adaptive plasticity core
- Deliverables:
  - latent affinity state manager (`A_ij`, `E_ij`) with sparse candidate updates
  - node plasticity budget dynamics (`R_i`: fatigue/recovery)
  - hysteresis edge lifecycle (`theta_on`, `theta_off`) for emergent links/pruning
- Acceptance:
  - integration tests show stable emergent-link formation without rewiring oscillation
  - long-run tests keep bounded edge churn under repeated perturbations

15. Supervisory adaptive controller
- Deliverables:
  - confusion monitor (cluster margin, mixing entropy proxy)
  - forgetting monitor (important-node retention and connectivity loss)
  - policy hooks to modulate learning rates/thresholds/pruning budget at runtime
- Acceptance:
  - collapse and over-fragmentation scenarios trigger expected guardrail behavior
  - tests confirm diversity floor and retention floor are maintained

## Suggested Milestones

1. Milestone M1 (Week 1-2): Complete P0 tasks 1-3
2. Milestone M2 (Week 3-4): Complete P0 tasks 4-6
3. Milestone M3 (Week 5-6): Complete P1 tasks 7-10
4. Milestone M4 (Week 7+): Begin P2 tasks 11-13

## Risks and Mitigations

- Risk: dynamic simulation complexity grows too quickly
  - Mitigation: keep v1 operators linear and pluggable
- Risk: retrieval and simulation outputs conflict
  - Mitigation: enforce evidence-first answer schema and conflict flags
- Risk: frequent updates destabilize indexes
  - Mitigation: incremental ingest + scheduled compaction
- Risk: premature benchmark comparison creates misleading conclusions
  - Mitigation: gate comparison behind dynamics-readiness checklist

## Dynamics Readiness Checklist (Gate for Comparison)

1. Explicit flow dynamics equation (`x(t+1) = x(t) + F(x,t,u)`) implemented
2. Attractor/repeller/turbulence/viscosity terms exposed and tested
3. Predict/intervene pipeline consumes dynamics outputs, not only edge retrieval
4. Unit tests cover stabilization behavior and intervention counterfactual outputs

## Current Build Status

- Done:
  - Dynamics core equation implementation (`FlowFieldDynamics`)
  - Dynamic graph adjustment engine (`DynamicGraphAdjuster`) with edge weight adaptation and rewiring/drop suggestions
  - Core co-evolution cycle: iterative state evolution <-> graph adjustment loop integrated into predict/intervene
  - Node-level topological flow controller (`TopologicalFlowController`) integrated into co-evolution cycles
  - Physics-inspired dynamics upgrade (adaptive dt + RK2 integration + kinetic-energy tracking)
  - Topology-inspired controller upgrade (cycle-pressure term from triangle loop density)
  - Vector-search-inspired rewiring upgrade (cosine similarity over node embeddings + bridge score ranking)
  - Closed-loop stability monitoring (saturation ratio, oscillation index, convergence gate in core cycle)
  - Objective-aware control adaptation (objective score tracked and used for gain updates/convergence)
  - Topological rigor upgrade with Hodge-like decomposition metrics (`gradient/curl/harmonic` ratios)
  - Higher-order topology signal integrated (`higher_order_pressure`) and upgraded to simplicial approximation (triangle/tetra clique participation)
  - One-step lookahead gain selection for control parameters (`k_div`, `residual_damping`, `k_higher`)
  - Multiscale control path (`ClusterFlowController`): coarse cluster control + micro topological refinement
  - Multi-step lookahead objective rollout for gain search and stricter cross-scale convergence gate
  - Predict/intervene integration with dynamics snapshots
  - Transition/resilience analysis (`transition_matrix`, `recovery_rate`, `hysteresis_index`)
  - Basin/trigger and recovery-path metrics (`overshoot`, `settling_time`, `path_efficiency`)
  - Basin boundary and trigger-strength metrics (`basin_radius`, `basin_occupancy`, `avg_trigger_confidence`)
  - Basin strategy plugin support (`centroid_std`, `quantile`) and shock-to-transition causal alignment score
  - Time-axis phase-transition detector (`PhaseTransitionAnalyzer`) integrated into core cycle metrics
  - Phase-aware closed-loop policy: prior cycle phase risk feeds controller safety clamp and graph adjustment aggressiveness
  - Regime persistence metric integrated (`regime_persistence_score`)
  - Review-driven consistency fixes: phase signal path unified with analyzer output; drop/new edge conservativeness made phase-effective
  - Dynamic graph adjustment objective added (`adjustment_objective_score`) and integrated into pipeline metrics
  - Multi-step global planning PoC for adjustment scale selection (`selected_adjustment_scale`) with discounted objective rollout
  - Phase-aware rewiring candidate scoring upgraded (risk-adaptive stable-vs-exploratory scoring blend)
  - Adaptive weighting by graph profile added (density/noise-aware objective and planning candidates)
  - Control-adjustment coupling term added to adjustment objective (`coupling_penalty` from residual/divergence/energy)
  - Structural-edit execution loop added (apply subset of suggested new/drop edges per cycle under risk-aware budget)
  - Adaptive planner horizon added to adjustment rollout (risk/density/noise/coupling aware)
  - Phase-change rigor upgraded with critical-slowing and hysteresis-proxy metrics
  - Joint plan optimization added for adjustment scale + structural edit budget (`selected_edit_budget`)
  - Planner forecast model refined with damping/rebound dynamics; phase-rigor signals now constrain budget/horizon
  - Calibration interface added (`AdjustmentPlannerConfig`) for offline tuning of planner/objective coefficients
  - Objective term-level logging added for interpretability and debugging
  - Extreme integration scenario coverage added (sparse/dense/high-noise behavior bounds)
  - Structural safety constraints added for edit execution (bridge/degree protection on edge drops)
  - Completed planner signal path for phase-rigor metrics (`critical_slowing`, `hysteresis_proxy`)
  - Baseline adaptive plasticity state loop in adjuster (`A_ij`, `E_ij`, `R_i`) with hysteresis-based emergent suggestion path
  - Supervisory metrics/state module (`SupervisoryMetricsAnalyzer`, `SupervisoryControlState`) with confusion/forgetting proxies and deterministic unit tests
  - Supervisory policy hooks in adjuster (`eta_up/down`, `theta_on/off`, edit-budget/new-drop bias modulation) with policy trace observability
  - Pipeline integration of supervisory signals and metrics output (`supervisory_confusion_score`, `supervisory_forgetting_score`)
  - Long-run repeated-perturbation integration tests with guardrail thresholds (`tests/test_longrun_guardrails.py`)
  - Calibration runner extension with plasticity/hysteresis profile candidates and summary-range artifact output (`run_calibration_with_summary`, CLI `--batch/--summary-out`)
  - Adapter SDK foundation (`BaseAdapter`, adapter registry, canonical contract validator) and integration tests
  - Calendar adapter + generic rule-mapping adapter running on same query API (`predict/intervene`)
  - Core completion/readiness checklist and automated readiness tests added
  - Analyzer configuration object for domain tuning (`FlowAnalyzerConfig`: thresholds, weights, lag)
  - Unit tests for dynamics and flow analysis
- Missing:
  - Causal validation against exogenous signals (current trigger confidence is rule-based)
  - Basin boundary calibration against domain data (current boundary uses centroid + std radius)
  - Domain-specific trigger thresholds (currently global constants)
  - Dynamic graph adjustment policy learning/calibration from real outcomes
  - Stronger control optimization (current multistep lookahead is still local horizon, not global optimizer)
  - Persistent/topological-history aware phase-change calibration (current phase detector is heuristic, not data-calibrated)
  - Global optimization across multi-cycle graph rewiring (current phase-aware adjustment remains local-step policy)
  - Evaluation protocol redesign for dynamics validity (non-leaky ground truth)
  - Real GraphRAG generation path (current baseline remains retrieval-heavy)
  - Supervisory control loop for confusion/forgetting stabilization
  - Adaptive plasticity/hysteresis calibration (`theta_on/off`, dwell, eta, fatigue/recovery) against long-run behaviors

## Next Session Checklist (Action Order)

1. Implement supervisory metrics and state (Completed 2026-02-15)
- Deliverables:
  - confusion proxy: cluster margin + mixing entropy
  - forgetting proxy: important-node retention/connectivity loss
  - lightweight `SupervisoryControlState` object
- Acceptance:
  - deterministic unit tests for metric bounds and monotonic sanity checks

2. Wire supervisory policy into adjuster (Completed 2026-02-15)
- Deliverables:
  - modulation hooks for `eta_up/down`, `theta_on/off`, and edit budget bias
  - policy trace outputs for observability
- Acceptance:
  - high-confusion scenario reduces merge-prone rewiring
  - high-forgetting scenario reduces aggressive pruning

3. Long-run integration tests for phase outcomes (Completed 2026-02-15)
- Deliverables:
  - repeated-perturbation scenarios (collapse-prone, fragmentation-prone, noisy)
  - guardrail pass/fail thresholds
- Acceptance:
  - bounded edge churn and diversity/retention floors maintained

4. Calibration runner extension (Completed 2026-02-15)
- Deliverables:
  - scenario batch in calibration CLI for plasticity/hysteresis tuning
  - compact summary artifact with recommended coefficient ranges
- Acceptance:
  - reproducible run and stable recommended parameter window

5. Promote new metrics into reporting outputs (Completed 2026-02-15)
- Deliverables:
  - extend calibration report columns with long-run guardrail summary metrics
  - add concise markdown/CSV reporting template for session handoff
- Acceptance:
  - report includes candidate ranking + coefficient window + guardrail trend snapshots

6. Adapter hardening and second production domain adapter (Next)
- Deliverables:
  - stricter contract rules (provenance/confidence mandatory path)
  - one production-grade external domain adapter (supply-chain/community chosen by user)
  - adapter CLI/test template for external contributors
- Acceptance:
  - adapter builds must pass validation gate
  - same `FlowGraphRAG` API regression suite passes on both domains

## Stability Exit Criteria (Gate to Multi-Domain Adapter Work)

Declare stability phase complete when all conditions are satisfied:

1. Full automated test suite is green.
2. Core readiness CLI passes with zero failures/errors.
3. Long-run guardrail integration tests pass (collapse-prone, fragmentation-prone, noisy).
4. Calibration CLI runs reproducibly and emits summary artifact with recommended coefficient ranges.
5. Supervisory metrics are wired into pipeline outputs and adjustment policy trace is observable.

Status (2026-02-15):
- Criteria 1-5 satisfied.
- Multi-domain adapter implementation is unblocked and started.

## Edge-Case Review Checklist

1. Empty trajectory inputs should return safe defaults (no crash, zeroed metrics)
2. Near-constant trajectories should avoid divide-by-zero in efficiency/recovery metrics
3. High viscosity states should still produce bounded velocities and stable updates
4. Missing perturbation targets should fall back to deterministic default behavior
5. Transition triggers should be optional and robust to single-step trajectories
6. Emergent-link update must avoid O(N^2) blow-up on large graphs via candidate filtering
7. Plasticity exhaustion must not freeze graph permanently (recovery path required)
8. Hysteresis thresholds must prevent flip-flop rewiring under noisy impacts

## Change Log

- 2026-02-15: Initial task breakdown created from design discussion.
- 2026-02-15: Added uv virtualenv workflow requirement in P0 foundation tasks.
- 2026-02-15: Added initial synthetic benchmark harness implementation (Plain RAG vs GraphRAG vs Flow Graph RAG) with CLI runner.
- 2026-02-15: Added benchmark gating policy and dynamics-readiness checklist.
- 2026-02-15: Added transition/resilience analysis layer and integrated dynamics metrics into predict/intervene outputs.
- 2026-02-15: Added basin/trigger and recovery-path metrics with edge-case review checklist.
- 2026-02-15: Added basin radius/occupancy and trigger confidence scoring to predict/intervene outputs.
- 2026-02-15: Added basin strategy plugin and causal alignment scoring from shock to trajectory changes.
- 2026-02-15: Added `FlowAnalyzerConfig` for configurable trigger thresholds/weights and causal lag.
- 2026-02-15: Added `DynamicGraphAdjuster` and integrated structural adjustment metrics into predict/intervene.
- 2026-02-15: Switched predict/intervene to iterative core co-evolution cycle (state evolution + graph adjustment per cycle).
- 2026-02-15: Added node-level topological flow controller and control-energy/residual metrics.
- 2026-02-15: Upgraded core algorithms with adaptive RK2 dynamics, cycle-pressure control, and vector-similarity rewiring.
- 2026-02-15: Added convergence gate and oscillation/saturation stability metrics to co-evolution loop.
- 2026-02-15: Added objective-aware gain adaptation and objective-based convergence checks.
- 2026-02-15: Added Hodge-like decomposition metrics (gradient/curl/harmonic) to topological control outputs.
- 2026-02-15: Added higher-order pressure metric and control term from 4-cycle style topology signal.
- 2026-02-15: Added one-step lookahead objective search for control gain selection.
- 2026-02-15: Added multiscale coarse-to-fine control (`ClusterFlowController`) and cross-scale metrics.
- 2026-02-15: Upgraded to multi-step discounted lookahead and tightened convergence with cross-scale consistency condition.
- 2026-02-15: Added simplicial topology module and wired `simplex_density`/`topological_tension` into control objective and pipeline metrics.
- 2026-02-15: Added `PhaseTransitionAnalyzer` and exposed critical/early-warning/regime-switch metrics in predict/intervene outputs.
- 2026-02-15: Added phase-aware closed-loop safety clamp + conservative adjustment policy and regime persistence metric.
- 2026-02-15: Applied review fixes for phase-loop consistency and risk-policy efficacy (including post-adaptation safety clamp).
- 2026-02-15: Added explicit objective for graph adjustment (churn/volatility/rewiring/risk) and surfaced baseline/intervention metrics.
- 2026-02-15: Added global adjustment-scale planner and phase-aware rewiring scoring strategy.
- 2026-02-15: Added sparse/dense/noise-adaptive weighting and corresponding pipeline metrics.
- 2026-02-15: Added control-adjustment coupling penalty and pipeline-level observability metrics.
- 2026-02-15: Added applied structural edits in core cycle and observability metrics for applied new/drop counts.
- 2026-02-15: Added adaptive planner horizon and upgraded phase-change metrics for long-history-like signals.
- 2026-02-15: Added joint plan search over scale and edit-budget for cumulative objective minimization.
- 2026-02-15: Added phase-rigor-constrained planning and improved rollout forecast dynamics.
- 2026-02-15: Added calibration hooks, objective-term observability, and extreme scenario integration tests.
- 2026-02-15: Added structural drop-safety constraints and finalized phase-rigor constrained planner path.
- 2026-02-15: Added core readiness checklist and automated readiness verification tests.
- 2026-02-15: Added readiness-report CLI and offline calibration runner with automated tests.
- 2026-02-15: Added standalone dynamic simulator and CLI for per-step node/edge/control/phase observability.
- 2026-02-15: Added visual HTML replay output mode for dynamic simulator CLI.
- 2026-02-15: Fixed simulator replay to use per-frame node positions and added flow-advection displacement so node movement reflects dynamic influence each cycle.
- 2026-02-15: Added CLI-scale controls for large synthetic graphs (`--nodes`, `--avg-degree`, `--seed`) and rendering controls (`--render-max-edges`, `--render-max-labels`) plus no-`scipy` layout fallback for large runs.
- 2026-02-15: Added physics-based position update path (`--position-model physics`) and split high-scale renderer with `--format webgl` output for large graph replay.
- 2026-02-15: Added design-aligned P2 tasks for adaptive plasticity (`A/E/R`) and supervisory adaptive control (confusion/forgetting guardrails with hysteresis rewiring lifecycle).
- 2026-02-15: Implemented baseline `A/E/R` loop in adjuster with candidate filtering, hysteresis streak gating, and new observability metrics (`affinity_suggested_*`, tracked pairs, mean plasticity budget).
- 2026-02-15: Implemented supervisory metrics/state module (`flow.supervisory`) and added deterministic tests for bounds/monotonicity (`tests/test_supervisory.py`).
- 2026-02-15: Wired supervisory policy into adjuster/pipeline with runtime modulation + trace metrics; added acceptance tests for confusion-driven merge suppression and forgetting-driven pruning suppression.
- 2026-02-15: Added long-run integration tests (`tests/test_longrun_guardrails.py`) for collapse-prone, fragmentation-prone, and noisy repeated perturbations with guardrail thresholds.
- 2026-02-15: Extended calibration engine/CLI for plasticity-hysteresis tuning with mixed profile batches and summary coefficient-range artifact output.
- 2026-02-15: Added explicit stability exit gate and marked gate conditions as satisfied; proceeded to multi-domain adapter implementation.
- 2026-02-15: Replaced ad-hoc adapter approach with SDK-style adapter package (base/registry/validation) and added generic rule-mapping adapter for user-defined domain schemas.
- 2026-02-15: Removed concrete calendar adapter from core package and kept domain-specific adapter implementations in examples/external path to preserve SDK-only core boundaries.
- 2026-02-15: Completed reporting promotion with long-run guardrail calibration columns and CSV/Markdown handoff templates (`ffrag.calibration_cli --rows-out/--summary-out/--report-md-out`).
