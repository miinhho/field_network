from __future__ import annotations

from datetime import datetime, timezone

from .compose import compose_answer
from .flow import FlowSimulator, StateVectorBuilder
from .models import Answer, LayeredGraph, Perturbation, Query
from .retrieval import GraphRetriever
from .router import QueryRouter


class FlowGraphRAG:
    """PoC orchestration for describe/predict/intervene queries."""

    def __init__(self) -> None:
        self.router = QueryRouter()
        self.retriever = GraphRetriever()
        self.state_builder = StateVectorBuilder()
        self.simulator = FlowSimulator()

    def run(self, graph: LayeredGraph, query: Query, perturbation: Perturbation | None = None) -> Answer:
        query_type = self.router.classify(query)
        if query_type == QueryRouter.DESCRIBE:
            return self._describe(graph, query)
        if query_type == QueryRouter.PREDICT:
            return self._predict(graph, query, perturbation)
        if query_type == QueryRouter.INTERVENE:
            return self._intervene(graph, query, perturbation)
        raise ValueError(f"Unknown query type: {query_type}")

    def _describe(self, graph: LayeredGraph, query: Query) -> Answer:
        local = self.retriever.retrieve_local(graph, query.text)
        if not local:
            local = self.retriever.retrieve_global(graph)

        evidence_ids = [edge.interaction_id for edge in local]
        layer_count = len({edge.layer for edge in local})
        claims = [
            f"Retrieved {len(local)} evidence edges across {layer_count} layers.",
            "Structure summary is based on graph connectivity, not subjective intent.",
        ]
        metrics = {
            "evidence_edge_count": float(len(local)),
            "layer_coverage": float(layer_count),
        }
        return compose_answer("describe", claims, evidence_ids, metrics, uncertainty=0.2)

    def _predict(self, graph: LayeredGraph, query: Query, perturbation: Perturbation | None) -> Answer:
        p = perturbation or self._default_perturbation(graph)
        result = self.simulator.propagate(graph, p)
        top_impacts = sorted(result.impact_by_actant.items(), key=lambda item: item[1], reverse=True)[:3]

        claims = [
            f"Predicted propagation reached {len(result.impact_by_actant)} actants in {result.hops_executed} hops.",
            f"Top impacted actants: {', '.join(actant for actant, _ in top_impacts) if top_impacts else 'none'}.",
        ]
        metrics = {
            "hops_executed": float(result.hops_executed),
            "affected_actants": float(len(result.impact_by_actant)),
            "stabilized": 1.0 if result.stabilized else 0.0,
        }
        evidence_ids = [f"perturbation:{p.perturbation_id}"]
        return compose_answer("predict", claims, evidence_ids, metrics, uncertainty=0.35)

    def _intervene(self, graph: LayeredGraph, query: Query, perturbation: Perturbation | None) -> Answer:
        p = perturbation or self._default_perturbation(graph)
        result = self.simulator.propagate(graph, p)
        rewires = result.rewired_edges[:3]
        rewire_text = ", ".join(f"{src}->{dst}" for src, dst in rewires) if rewires else "no rewiring suggested"

        claims = [
            f"Suggested rewiring candidates: {rewire_text}.",
            "Intervention proposal is based on structural impact propagation.",
        ]
        metrics = {
            "rewire_candidates": float(len(rewires)),
            "affected_actants": float(len(result.impact_by_actant)),
        }
        evidence_ids = [f"perturbation:{p.perturbation_id}"]
        return compose_answer("intervene", claims, evidence_ids, metrics, uncertainty=0.4)

    def build_state_vector(self, graph: LayeredGraph, entity_id: str) -> dict[str, float]:
        return self.state_builder.build(graph, entity_id, timestamp=datetime.now(timezone.utc)).values

    def _default_perturbation(self, graph: LayeredGraph) -> Perturbation:
        target = next(iter(graph.actants), "")
        return Perturbation(
            perturbation_id="auto-default",
            timestamp=datetime.now(timezone.utc),
            targets=[target] if target else [],
            intensity=1.0,
            kind="default",
        )
