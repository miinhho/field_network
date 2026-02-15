from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag import (
    FlowGraphRAG,
    GenericMappingAdapter,
    MappingSpec,
    get_adapter,
    Query,
)


class DomainAdapterIntegrationTests(unittest.TestCase):
    def test_builtin_generic_adapter_runs_predict(self) -> None:
        base = datetime(2026, 2, 10, 9, 0, 0)
        records: list[dict[str, object]] = [
            {"ts": base, "actor": "u1", "asset": "a1", "peer": "a2"},
            {"ts": base + timedelta(hours=1), "actor": "u2", "asset": "a2", "peer": "a3"},
        ]
        spec = MappingSpec(
            node_fields={"actor": "person", "asset": "asset", "peer": "asset"},
            edge_rules=[("actor", "asset", "ownership", 0.7), ("asset", "peer", "dependency", 0.6)],
            timestamp_field="ts",
            perturb_target_field="asset",
            default_intensity=1.0,
        )
        adapter_factory = get_adapter("generic")
        adapter = adapter_factory(records=records, spec=spec, graph_id="g-sdk")
        graph = adapter.to_graph()
        perturbation = adapter.default_perturbation()

        self.assertGreaterEqual(len(graph.actants), 3)
        self.assertGreaterEqual(len(graph.interactions), 4)
        self.assertGreaterEqual(len(perturbation.targets), 1)

        rag = FlowGraphRAG()
        out = rag.run(graph, Query(text="predict next transition"), perturbation=perturbation)
        self.assertEqual(out.query_type, "predict")
        self.assertIn("affected_actants", out.metrics_used)

    def test_generic_mapping_adapter_builds_graph_and_runs_intervene(self) -> None:
        base = datetime(2026, 2, 11, 10, 0, 0)
        records: list[dict[str, object]] = [
            {
                "ts": base,
                "actor": "agent-1",
                "asset": "asset-a",
                "peer": "asset-b",
            },
            {
                "ts": base + timedelta(minutes=4),
                "actor": "agent-2",
                "asset": "asset-b",
                "peer": "asset-c",
            },
            {
                "ts": base + timedelta(minutes=8),
                "actor": "agent-1",
                "asset": "asset-c",
                "peer": "asset-a",
            },
        ]
        spec = MappingSpec(
            node_fields={"actor": "person", "asset": "asset", "peer": "asset"},
            edge_rules=[
                ("actor", "asset", "ownership", 0.8),
                ("asset", "peer", "dependency", 0.6),
            ],
            timestamp_field="ts",
            perturb_target_field="asset",
            default_intensity=1.1,
        )
        adapter = GenericMappingAdapter(records=records, spec=spec, graph_id="custom-domain")
        graph = adapter.to_graph()
        perturbation = adapter.default_perturbation()

        self.assertGreaterEqual(len(graph.actants), 4)
        self.assertGreaterEqual(len(graph.interactions), 4)
        self.assertGreaterEqual(len(perturbation.targets), 1)

        rag = FlowGraphRAG()
        out = rag.run(graph, Query(text="intervene to reduce risk"), perturbation=perturbation)
        self.assertEqual(out.query_type, "intervene")
        self.assertIn("intervention_improvement", out.metrics_used)


if __name__ == "__main__":
    unittest.main()
