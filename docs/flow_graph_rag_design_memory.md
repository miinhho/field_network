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
- Multiscale clustering policy upgraded from structure-only modularity to hybrid graph partitioning (structural edges + dynamic embedding kNN + temporal inertia links)
- ANN retrieval backend integrated for hybrid clustering (`FAISS` preferred, exact-cosine fallback), so dynamic-neighbor graph construction now follows retrieval-grade indexing path
- ANN policy tightened: default behavior is fail-fast on FAISS environment errors (fallback is opt-in only), to surface environment misconfiguration explicitly
- ANN index lifecycle improved: context-keyed ANN cache reuse added for repeated cluster queries within stable feature regimes
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
- Synapse-like adaptive rewiring (planned): latent affinity state over non-edges (`A_ij`), eligibility traces (`E_ij`), and node plasticity budgets (`R_i`) to model gradual link formation/pruning
- Supervisory adaptive controller (planned): higher-level feedback loop that monitors confusion/forgetting risks and modulates plasticity gains, rewiring thresholds, and pruning aggressiveness
- Hysteresis-based edge lifecycle (planned): `theta_on > theta_off` to stabilize emergent-link creation/removal and avoid oscillatory rewiring

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
- `LatentAffinityState`: non-edge coupling potential (`A_ij`) and eligibility traces (`E_ij`)
- `PlasticityState`: node-level plasticity resources (`R_i`) with fatigue/recovery
- `SupervisoryControlState`: global feedback signals (confusion risk, forgetting risk, diversity pressure)

## Why Synapse-Like Effects Emerge Here

Synapse-like effects appear naturally when a dynamic graph uses local reinforcement plus structural adaptation:

1. Repeated co-activation drives positive feedback on coupling potential (`A_ij`)
2. Reinforced couplings increase future co-activation probability (self-reinforcing loop)
3. Weakly used couplings decay and eventually prune (activity-dependent forgetting)
4. Without global constraints, systems move toward either rapid consensus/collapse or fragmentation

This is why we add supervisory/homeostatic constraints rather than pure Hebbian-like local updates.

## Why This Differs From Plain GraphRAG

- Plain GraphRAG: strong at retrieval and global sensemaking over static/semi-static graph structure
- Flow Graph RAG: adds explicit dynamic operators for "what changes next" reasoning

## Domain Generality Strategy

- Keep core ontology minimal and domain-neutral
- Push domain specifics to adapter and feature plugin interfaces
- Support both event streams and batch graph snapshots
- Enforce adapter output contract with built-in validation and mapping reports before core execution

## Tech Direction (Current)

- Language: Python
- Runtime baseline: Python 3.10 (for stable `faiss-gpu` wheel compatibility)
- Environment: `uv` + project-local `.venv`
- Graph: `networkx` (initial), optional `igraph` later for scale
- Numeric ops: `numpy` as default for core vector/math operations
- Numeric compatibility note: `numpy<2.0` pinned while using current `faiss-gpu` wheels
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
5. Latent-affinity candidate policy at scale (Top-K neighborhood vs ANN candidates vs mixed)
6. Supervisory controller policy form (rule-based first vs learned policy later)

## Session Handoff (Current Snapshot)

### Implemented Core

1. Dynamic co-evolution loop (state/control/phase/adjustment)
2. Physics-capable simulator with HTML and WebGL replay outputs
3. Baseline adaptive plasticity in adjuster (`A_ij`, `E_ij`, `R_i`) with hysteresis suggestion path
4. Structural safety constraints for drop edits (degree/connectivity preservation)
5. Supervisory metrics/state module added (confusion proxy: cluster margin + mixing entropy, forgetting proxy: retention + connectivity loss)
6. Supervisory policy integration in adjuster and pipeline (runtime modulation of `eta_up/down`, `theta_on/off`, edit budget bias, and policy trace observability)
7. Long-run guardrail integration tests added (collapse-prone / fragmentation-prone / noisy repeated-perturbation scenarios with bounded churn and diversity/retention floor checks)
8. Calibration runner extension added for plasticity/hysteresis tuning: mixed candidate batch (`planner x plasticity`), supervisory-aware scoring, and compact recommended coefficient-range artifact output
9. Initial adapter SDK structure implemented: `BaseAdapter` + registry + contract validator + mapping report path
10. Domain adapters on the SDK:
  - core package keeps generic rule-mapping adapter (`GenericMappingAdapter`, `MappingSpec`) for user-defined relations
  - concrete domain adapters are intended to live outside core (example moved under `examples/adapters/`)
11. Reporting promotion completed for calibration: row-level long-run guardrail metrics (`avg_longrun_churn/retention/diversity`) and export templates for CSV + Markdown handoff artifacts
12. Multiscale cluster planner upgraded to hybrid clustering graph: normalized structural weights + node-feature kNN similarity edges + prior-assignment inertia edges
13. ANN index abstraction added for clustering (`create_cosine_ann_index`): `FAISS` preferred, then exact fallback
14. Cluster inertia state key upgraded from graph-id-only to explicit context key (`context_id` / stream id), supporting persistent flow impact streams without cross-stream contamination
15. Context-state lifecycle controls added for clustering memory (forgetting-curve retention with importance/frequency mix + bounded capacity + explicit context clear API)
16. Cluster-control consistency fix: cluster control graph now aggregates hybrid partition graph edges (not only raw structural edges)
14. Implementation hygiene pass completed: UTC timestamp consistency fix in adjuster edge insertion path, and duplicate predict stabilization logic centralized

### Still Missing for Target Architecture

1. Supervisory adaptive controller (confusion + forgetting monitors)
2. Adaptive plasticity parameter calibration (`eta_up/down`, `theta_on/off`, dwell)
3. Candidate policy upgrade for large graphs (ANN or hybrid retrieval)
4. Long-horizon collapse/fragmentation validation protocol
5. Production-grade adapter coverage beyond initial adapters (schema hardening, richer domain mappings, ingestion contracts)
6. ANN quality calibration for dynamic neighbor graph (candidate recall/precision and cluster stability tuning), now that `FAISS` path exists

### Next Session Checklist

- [x] Add supervisory metrics module (cluster margin, mixing entropy proxy, retention loss)
- [x] Wire supervisory policy into adjuster parameter modulation
- [x] Add long-run integration test for anti-collapse/anti-fragmentation guardrails
- [x] Add calibration CLI scenario set for plasticity/hysteresis parameters
- [x] Promote new metrics into pipeline/reporting outputs

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
- 2026-02-15: Added readiness-report CLI and offline calibration runner with script entrypoints.
- 2026-02-15: Added step-by-step dynamic simulator (`ffrag-simulate`) to observe node/edge updates, control signals, and phase-adjustment interactions per cycle.
- 2026-02-15: Added HTML replay mode to simulator for visual step-by-step graph evolution playback.
- 2026-02-15: Updated simulator to expose and render per-frame node positions with flow-driven advection so influence-induced node movement is visible (not static final-layout replay).
- 2026-02-15: Added large-scale simulator support (parameterized synthetic graph generation for thousands of nodes, target fallback safety, and no-`scipy` layout fallback for high-node replay environments).
- 2026-02-15: Added physically inspired node-motion mode (`position_model=physics`) using mass/velocity/damping/spring/field-force integration, and added separate WebGL replay renderer for large-graph visualization.
- 2026-02-15: Revised architecture with adaptive-network evidence alignment: added planned latent-affinity/plasticity/hysteresis loop and a supervisory adaptive controller for confusion/forgetting stabilization.
- 2026-02-15: Implemented first adaptive-plasticity pass in `DynamicGraphAdjuster`: persistent latent affinity/eligibility/plasticity states with hysteresis-based emergent-link suggestions merged into structural edit candidates.
- 2026-02-15: Implemented `flow.supervisory` metrics/state module (`SupervisoryMetricsAnalyzer`, `SupervisoryControlState`) with deterministic tests for bounds and monotonic sanity checks.
- 2026-02-15: Wired supervisory policy into `DynamicGraphAdjuster` and `FlowGraphRAG` core loop; supervisory risk now modulates plasticity gains/thresholds/edit budget and is exposed via policy trace + pipeline metrics.
- 2026-02-15: Added long-run guardrail integration tests covering collapse-prone, fragmentation-prone, and noisy repeated-perturbation regimes with bounded edge churn and retention/diversity floor assertions.
- 2026-02-15: Extended calibration runner/CLI with plasticity-hysteresis profile batches, supervisory-aware calibration metrics, and summary artifact output for recommended coefficient windows.
- 2026-02-15: Reworked adapters into SDK-style package (`BaseAdapter`, registry, contract validator) to separate domain semantics from core engine.
- 2026-02-15: Kept core adapter layer SDK-focused; moved concrete calendar adapter out of core into `examples/` and retained generic mapping adapter in core.
- 2026-02-15: Added calibration reporting templates and long-run guardrail columns to calibration outputs (`rows.csv`, `summary.csv`, markdown report).
- 2026-02-17: Reframed multiscale clustering as control-oriented hybrid partitioning and implemented first pass (structure + dynamic kNN + temporal inertia) in `ClusterFlowController`.
- 2026-02-17: Replaced hard dynamic-kNN cutoff behavior with neural-inspired soft pattern-separation policy (anchor-prototype large-graph candidate routing + mutual-kNN + lateral-inhibition-style separation gate).
- 2026-02-17: Integrated real ANN backend for clustering (`FAISS` optional + exact fallback), added ANN index tests, and completed implementation audit refactors (UTC timestamp consistency + predict stabilization dedup).
- 2026-02-17: Consolidated ANN backend order to `FAISS -> exact` and removed `hnswlib` path to simplify dependency/ops surface.
- 2026-02-17: Switched project runtime baseline to Python 3.10 and pinned `numpy<2.0` to resolve `faiss-gpu` ABI compatibility.
- 2026-02-17: Updated ANN creation policy to strict FAISS fail-fast by default and wired cluster inertia memory to explicit `context_id` (stream-aware state continuity).
- 2026-02-17: Added context-keyed ANN cache reuse, bounded context-state LRU lifecycle, and hybrid-edge-aligned cluster control graph construction; dynamics mass scaling refined to per-feature effective mass.
- 2026-02-17: Replaced simple context LRU with forgetting-curve eviction policy (half-life decay + frequency + importance override) for stream-aware long-run memory management.
- 2026-02-17: Promoted cluster-memory observability into calibration reporting outputs (`avg_cluster_ann_cache_hit_rate`, `avg_cluster_active_contexts`, `avg_cluster_evicted_contexts`) for ANN/context retention tuning loops.
- 2026-02-17: Refined forgetting-curve lifecycle to include retention-floor expiration, so stale contexts can be evicted even without capacity pressure.
- 2026-02-17: Improved cluster-impact aggregation to reduce sign-cancellation loss (direction/magnitude decomposition with coherence-aware magnitude floor), strengthening coarse control signal fidelity.
- 2026-02-17: Updated coarse projection to preserve signed cluster-control impact, reducing information loss before micro-scale topological refinement.
- 2026-02-17: Updated node-level topological controller projection to preserve signed controlled impact, keeping coarse-to-fine signal semantics consistent.
- 2026-02-17: Replaced cross-scale consistency with a mixed criterion (sign-consistency + magnitude-stability) to better reflect signed dynamics coherence under clustered control.
- 2026-02-17: Added signed-flow long-run guardrail regression coverage (mean-centered signed impact stream) to validate stability/churn/retention under signed projection policy.
- 2026-02-17: Hardened adjuster rewiring scoring for signed impacts: candidate ranking now uses impact magnitude and pair-stability is signed-safe/bounded, preventing instability from sign-mixed inputs.
- 2026-02-17: Refined adjuster edge-update and drop-edge policy for signed dynamics: edge pressure now includes sign-alignment term, low-pressure damping uses absolute pressure, and drop scoring combines magnitude support with sign-mismatch penalty.
- 2026-02-17: Extended core `FlowSimulator` with signed perturbation seeding (`target_weights`) and edge polarity propagation (`metadata.polarity`), plus signed-safe rewiring candidate filtering.
- 2026-02-17: Propagated signed perturbation semantics into pipeline/simulator shock construction (`target_weights` -> shock polarity) and switched core history-state candidate ranking to absolute impact magnitude.
- 2026-02-17: Extended `PhaseTransitionAnalyzer` with signed-dynamics indicators (`sign_flip_rate`, `polarity_coherence_score`) and wired them into phase risk scoring plus pipeline/intervention metrics outputs.
- 2026-02-17: Integrated signed phase indicators into adjuster risk policy (`DynamicGraphAdjuster._phase_risk`) so high sign-flip / low polarity-coherence states directly reduce structural adjustment aggressiveness.
- 2026-02-17: Integrated signed phase indicators into supervisory policy modulation (`_supervisory_policy`) and pipeline handoff, so sign-instability directly suppresses adjustment budget/new-edge bias and increases conservative hysteresis behavior.
