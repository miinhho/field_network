from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from ..flow import FlowSimulator
from ..retrieval import GraphRetriever
from .scenarios import PredictScenario, generate_scenarios


@dataclass(slots=True)
class BenchmarkResult:
    method: str
    avg_recall_at_k: float
    avg_precision_at_k: float


class PlainPredictor:
    """Baseline that only uses lexical matching and one-hop adjacency."""

    def predict(self, scenario: PredictScenario, k: int) -> list[str]:
        target = scenario.perturbation.targets[0] if scenario.perturbation.targets else ""
        neighbors: list[str] = []
        for edge in scenario.graph.interactions:
            if edge.source_id == target:
                neighbors.append(edge.target_id)
            elif edge.target_id == target:
                neighbors.append(edge.source_id)
        ordered = [target] + neighbors
        return ordered[:k]


class GraphPredictor:
    """Graph baseline using retrieval scores over edges."""

    def __init__(self) -> None:
        self.retriever = GraphRetriever()

    def predict(self, scenario: PredictScenario, k: int) -> list[str]:
        query_text = f"predict spread from {scenario.perturbation.targets[0]}"
        edges = self.retriever.retrieve_local(scenario.graph, query_text, limit=max(2 * k, 4))
        nodes: list[str] = []
        for edge in edges:
            nodes.append(edge.source_id)
            nodes.append(edge.target_id)
        deduped: list[str] = []
        for node in nodes:
            if node not in deduped:
                deduped.append(node)
        return deduped[:k]


class FlowPredictor:
    """Flow Graph predictor using propagation outputs."""

    def __init__(self) -> None:
        self.simulator = FlowSimulator(attenuation=0.6, max_hops=3)

    def predict(self, scenario: PredictScenario, k: int) -> list[str]:
        result = self.simulator.propagate(scenario.graph, scenario.perturbation)
        ranked = sorted(result.impact_by_actant.items(), key=lambda item: item[1], reverse=True)
        return [node for node, _ in ranked[:k]]


def _score(predicted: list[str], expected: set[str]) -> tuple[float, float]:
    if not predicted:
        return 0.0, 0.0
    hits = sum(1 for node in predicted if node in expected)
    recall = hits / max(1, len(expected))
    precision = hits / len(predicted)
    return recall, precision


def run_benchmark(num_scenarios: int = 20, top_k: int = 3, seed: int = 42) -> list[BenchmarkResult]:
    scenarios = generate_scenarios(num_scenarios=num_scenarios, seed=seed)
    predictors = {
        "plain_rag": PlainPredictor(),
        "graph_rag": GraphPredictor(),
        "flow_graph_rag": FlowPredictor(),
    }

    results: list[BenchmarkResult] = []
    for name, predictor in predictors.items():
        recalls: list[float] = []
        precisions: list[float] = []
        for scenario in scenarios:
            pred = predictor.predict(scenario, top_k)
            recall, precision = _score(pred, scenario.expected_impacted)
            recalls.append(recall)
            precisions.append(precision)
        results.append(
            BenchmarkResult(
                method=name,
                avg_recall_at_k=round(mean(recalls), 4),
                avg_precision_at_k=round(mean(precisions), 4),
            )
        )

    return sorted(results, key=lambda row: row.avg_recall_at_k, reverse=True)
