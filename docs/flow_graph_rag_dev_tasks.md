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
  - Predict/intervene integration with dynamics snapshots
  - Transition/resilience analysis (`transition_matrix`, `recovery_rate`, `hysteresis_index`)
  - Basin/trigger and recovery-path metrics (`overshoot`, `settling_time`, `path_efficiency`)
  - Basin boundary and trigger-strength metrics (`basin_radius`, `basin_occupancy`, `avg_trigger_confidence`)
  - Basin strategy plugin support (`centroid_std`, `quantile`) and shock-to-transition causal alignment score
  - Analyzer configuration object for domain tuning (`FlowAnalyzerConfig`: thresholds, weights, lag)
  - Unit tests for dynamics and flow analysis
- Missing:
  - Causal validation against exogenous signals (current trigger confidence is rule-based)
  - Basin boundary calibration against domain data (current boundary uses centroid + std radius)
  - Domain-specific trigger thresholds (currently global constants)
  - Dynamic graph adjustment policy learning/calibration from real outcomes
  - Stronger control optimization (current objective adaptation is local heuristic, not global optimizer)
  - More rigorous high-order topology modeling (cell/simplicial complex level, currently graph-cycle approximation)
  - Evaluation protocol redesign for dynamics validity (non-leaky ground truth)
  - Real GraphRAG generation path (current baseline remains retrieval-heavy)

## Edge-Case Review Checklist

1. Empty trajectory inputs should return safe defaults (no crash, zeroed metrics)
2. Near-constant trajectories should avoid divide-by-zero in efficiency/recovery metrics
3. High viscosity states should still produce bounded velocities and stable updates
4. Missing perturbation targets should fall back to deterministic default behavior
5. Transition triggers should be optional and robust to single-step trajectories

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
