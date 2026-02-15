# AGENTS.md

## Repository Memory Contract

This repository uses file-based memory for Codex across sessions.

- Source of truth for current design and goals:
  - `docs/flow_graph_rag_design_memory.md`
- Source of truth for implementation tasks and execution order:
  - `docs/flow_graph_rag_dev_tasks.md`

These files are intended to be read and updated in future sessions as persistent project memory.

## Update Rules

- Keep both memory files synchronized when architecture or priorities change.
- Prefer appending a short "Change Log" entry with date when making major updates.
- Do not store secrets or credentials in memory files.

## Ongoing Reporting Rules

For every implementation step, always report:

1. What is completed
2. What is still missing
3. What should be worked on next

After each implementation step, always include an edge-case review that lists likely failure cases and current handling status.
