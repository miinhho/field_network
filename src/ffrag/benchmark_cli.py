from __future__ import annotations

import argparse

from .benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Flow Graph RAG synthetic benchmark")
    parser.add_argument("--scenarios", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = run_benchmark(num_scenarios=args.scenarios, top_k=args.top_k, seed=args.seed)
    print("method,avg_recall_at_k,avg_precision_at_k")
    for row in rows:
        print(f"{row.method},{row.avg_recall_at_k:.4f},{row.avg_precision_at_k:.4f}")


if __name__ == "__main__":
    main()
