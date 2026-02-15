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
