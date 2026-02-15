# ffrag

PoC implementation of a Flow Graph RAG library in Python.

## Environment (uv)

Use `uv` with a project-local virtual environment.

```bash
uv venv
uv sync
uv run python -m unittest discover -s tests -p 'test_*.py'
uv run python -m ffrag.benchmark_cli --scenarios 20 --top-k 3 --seed 42
uv run python -m ffrag.dynamic_simulator_cli --steps 5 --format table
uv run python -m ffrag.dynamic_simulator_cli --steps 5 --format html --out simulator_trace.html
```
