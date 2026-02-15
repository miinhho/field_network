# Flow Graph RAG Design Memory

## Purpose

This file is persistent memory for Codex and collaborators.
It captures the current agreed design and goals for the project so future sessions can continue without losing context.

## Project Vision

Build a Python-first, domain-agnostic Flow Graph RAG library.

- Not limited to calendar/scheduling data
- Supports multiple domains via adapters
- Focuses on structural reasoning and dynamic change, not personality/motivation inference

## Core Product Goal (v1)

Deliver a usable library that answers three query classes with graph-grounded evidence:

1. `describe`: explain current graph structure and key patterns
2. `predict`: estimate likely transitions under perturbations
3. `intervene`: simulate candidate interventions and expected structural effects

## Non-Goals (v1)

- End-to-end GNN training pipeline as required baseline
- Subjective psychological inference about individuals
- Perfect real-time guarantees for all components

## Architecture Summary

Flow Graph RAG = GraphRAG + temporal/dynamic simulation layer.

### Layer 1: Knowledge Graph Retrieval

- Entity/relation extraction and graph retrieval for factual and structural context
- Global/local query routing

### Layer 2: Flow Dynamics Engine

- State-space construction (`state_vector`)
- Perturbation propagation (`impact diffusion`, `attenuation`)
- Transition and resilience estimation (`transition_matrix`, `recovery metrics`)
- Explicit dynamics step model (`x(t+1) = x(t) + F(x,t,u)`) for predict/intervene
- Node-level topological control with Hodge-like decomposition terms (`gradient/curl/harmonic`) for co-evolution stability
- Higher-order topological pressure term (4-cycle style loop pressure) included in node control input
- One-step lookahead objective minimization for control gain selection (`k_div`, `residual_damping`, `k_higher`)
- Coarse-to-fine multiscale control added: cluster-level planning followed by node-level topological refinement
- Multi-step lookahead objective rollout added for control gain selection (short horizon discounted planning)
- Simplicial topology approximation integrated (`triangle`/`tetra` cliques) and fed into control as higher-order node pressure
- Topological tension (participation variance over simplices) integrated as objective penalty term
- Phase-transition analyzer integrated on top of co-evolution trajectories (critical score, early warning, regime switching)
- Phase-aware closed loop integrated: prior cycle phase risk now modulates both node-control safety clamp and graph-adjustment aggressiveness
- Regime persistence metric added for temporal stability tracking (`regime_persistence_score`)
- Signal-consistency refinement: phase signal for next-cycle control/adjustment now comes from the same `PhaseTransitionAnalyzer` path used for reported metrics
- Dynamic graph adjustment now has an explicit objective score (`adjustment_objective_score`) combining churn, volatility, rewiring cost, and phase-risk penalty
- Global-planning PoC added to graph adjustment: candidate adjustment scales are evaluated with short-horizon discounted objective rollout before applying edge updates
- Rewiring candidate policy is now phase-aware at scoring level (high-risk favors stable similarity; low-risk favors exploratory bridging)
- Adaptive weighting layer added: objective/planner/rewiring behavior now responds to graph density (sparse vs dense) and impact-noise profile
- Control-adjustment coupling integrated: graph adjustment objective now includes controller residual/divergence/energy penalty for tighter co-evolution feedback
- Structural edit execution integrated: selected new/drop edges are now partially applied to graph topology each cycle (not suggestion-only)
- Planner upgrade: adjustment rollout horizon is now adaptive to risk/density/noise/coupling, rather than fixed
- Phase rigor upgrade: critical-slowing and hysteresis-proxy scores added for stronger phase-change interpretation
- Planner upgrade 2: adjustment now optimizes both scale and structural edit budget via multi-step discounted objective rollout
- Planner-physics refinement: rollout forecast now uses damping/rebound terms (viscosity, coupling, noise, slowing, hysteresis) instead of fixed linear decay
- Phase-to-planner constraints tightened: `critical_slowing`/`hysteresis_proxy` now directly constrain horizon and structural edit budget
- Calibration hook added: `AdjustmentPlannerConfig` exposes objective/planner coefficients for offline tuning without changing core code
- Objective observability added: adjustment result now reports term-level contributions (`churn/volatility/rewiring/risk/coupling`)
- Extreme integration tests added for sparse/dense/high-noise-like scenarios
- Structural safety constraints added to edit execution: edge drops now preserve endpoint minimum degree and avoid increasing connected-component count
- Planner now jointly reasons over scale/budget with phase-rigor constraints (`critical_slowing`, `hysteresis_proxy`) propagated into adjustment policy

### Layer 3: Answer Composer + Guardrails

- Combines retrieval evidence + simulation outputs
- Enforces structure-first interpretation
- Emits machine-readable evidence fields

## Canonical Data Abstractions

- `Actant`: generalized actor (human/non-human)
- `Interaction`: time-bound relation/event
- `LayeredGraph`: temporal/social/spatial or domain-specific layers
- `StateVector`: compact dynamic state representation
- `Perturbation`: external shock or graph update
- `PropagationResult`: impact spread and stabilization outputs

## Why This Differs From Plain GraphRAG

- Plain GraphRAG: strong at retrieval and global sensemaking over static/semi-static graph structure
- Flow Graph RAG: adds explicit dynamic operators for "what changes next" reasoning

## Domain Generality Strategy

- Keep core ontology minimal and domain-neutral
- Push domain specifics to adapter and feature plugin interfaces
- Support both event streams and batch graph snapshots

## Tech Direction (Current)

- Language: Python
- Environment: `uv` + project-local `.venv`
- Graph: `networkx` (initial), optional `igraph` later for scale
- Numeric ops: `numpy` as default for core vector/math operations
- Topology/time extensions: optional modules (`gudhi`, `ripser.py`, temporal graph tooling)
- Storage: pluggable (in-memory first, external stores later)

## Success Criteria (v1)

- Works on at least 2 domains using same core API
- Produces evidence-linked responses for all 3 query classes
- Outperforms baseline GraphRAG on dynamic questions (`predict`, `intervene`) in offline evaluation
- Maintains acceptable cost/latency envelope defined in task doc

## Open Design Decisions

1. Initial persistence stack (in-memory only vs in-memory + graph DB)
2. Default simulation complexity (linear rules vs richer nonlinear operators)
3. Minimal required confidence calibration in response schema
4. Evaluation protocol for dynamics validity before any comparative benchmark claims

## Change Log

- 2026-02-15: Initial memory document created from design discussion.
- 2026-02-15: Updated PoC tech direction to uv-managed virtualenv and standard libraries (`numpy`, `networkx`).
- 2026-02-15: Added explicit flow dynamics equation requirement to the architecture memory.
- 2026-02-15: Added topological control direction using gradient/curl/harmonic decomposition in core flow layer.
- 2026-02-15: Added higher-order topological pressure signal to control objectives.
- 2026-02-15: Added one-step lookahead control gain selection to reduce local objective per cycle.
- 2026-02-15: Added multiscale (cluster -> node) control loop for global-to-local flow stabilization.
- 2026-02-15: Upgraded gain search from one-step to multi-step discounted lookahead.
- 2026-02-15: Replaced graph-cycle higher-order proxy with simplicial topology model and added topological tension penalty.
- 2026-02-15: Added time-axis phase-transition detection metrics to core cycle (`critical_transition_score`, `early_warning_score`, `regime_switch_count`).
- 2026-02-15: Added phase-aware closed-loop control/adjustment and regime persistence metric.
- 2026-02-15: Reviewed and corrected phase-loop consistency and risk-policy effectiveness (drop/new edge conservativeness + post-adaptation safety clamp).
- 2026-02-15: Added explicit adjustment objective formulation and exposed it in predict/intervene metrics.
- 2026-02-15: Added multi-step adjustment-scale planner and phase-aware rewiring scoring policy.
- 2026-02-15: Added sparse/dense/noise-adaptive weighting and exposed profile metrics (`graph_density`, `impact_noise`).
- 2026-02-15: Added control-to-adjustment coupling penalty and exposed coupling metrics in predict/intervene outputs.
- 2026-02-15: Enabled in-loop structural edits (`applied_new_edges`, `applied_drop_edges`) for true dynamic topology evolution.
- 2026-02-15: Added adaptive planner horizon and phase rigor metrics (`critical_slowing_score`, `hysteresis_proxy_score`).
- 2026-02-15: Added joint planning for adjustment scale + edit budget (`edit_budget`) in the core loop.
- 2026-02-15: Refined rollout dynamics model and wired phase-rigor signals into planner constraints.
- 2026-02-15: Added planner calibration hooks, term-level objective logging, and extreme scenario integration tests.
- 2026-02-15: Added structural safety constraints for drop edits and completed phase-rigor signal propagation into planner policy.
- 2026-02-15: Added explicit core completion checklist and readiness tests for 100%-implementation verification.
