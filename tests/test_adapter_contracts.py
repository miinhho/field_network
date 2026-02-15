from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag import (
    BaseAdapter,
    GenericMappingAdapter,
    MappingSpec,
    Perturbation,
    Actant,
    get_adapter,
    list_adapters,
    register_adapter,
)
from ffrag.models import Interaction, LayeredGraph
from ffrag.adapters.validation import GraphContractValidator


class _ToyAdapter(BaseAdapter):
    def to_graph(self) -> LayeredGraph:
        g = LayeredGraph(graph_id="toy", schema_version="0.1")
        g.actants["a"] = Actant("a", "entity", "A")
        g.actants["b"] = Actant("b", "entity", "B")
        g.interactions.append(
            Interaction(
                interaction_id="t1",
                timestamp=datetime(2026, 2, 15, 9, 0, 0),
                source_id="a",
                target_id="b",
                layer="toy",
                weight=0.8,
            )
        )
        return g

    def default_perturbation(self) -> Perturbation:
        return Perturbation(
            perturbation_id="tp0",
            timestamp=datetime(2026, 2, 15, 9, 5, 0),
            targets=["a"],
            intensity=1.0,
            kind="toy",
        )

    def mapping_report(self) -> dict[str, float]:
        return {"toy_records": 1.0}


class AdapterContractTests(unittest.TestCase):
    def test_validator_detects_contract_violations(self) -> None:
        g = LayeredGraph(graph_id="bad", schema_version="0.1")
        g.interactions.append(
            Interaction(
                interaction_id="x1",
                timestamp=datetime(2026, 2, 15, 10, 0, 0),
                source_id="missing-a",
                target_id="missing-b",
                layer="social",
                weight=8.0,
            )
        )
        result = GraphContractValidator(max_weight=5.0).validate(g)
        self.assertFalse(result.valid)
        self.assertGreaterEqual(result.error_count, 2)

    def test_adapter_build_returns_validation_and_report(self) -> None:
        adapter = _ToyAdapter()
        out = adapter.build()
        self.assertTrue(out.validation.valid)
        self.assertGreaterEqual(out.mapping_report.get("toy_records", 0.0), 1.0)
        self.assertGreaterEqual(len(out.default_perturbation.targets), 1)

    def test_registry_registers_and_returns_adapter(self) -> None:
        register_adapter("toy", lambda: _ToyAdapter())
        names = list_adapters()
        self.assertIn("toy", names)
        factory = get_adapter("toy")
        adapter = factory()
        self.assertIsInstance(adapter, _ToyAdapter)

    def test_generic_mapping_adapter_with_contract(self) -> None:
        spec = MappingSpec(
            node_fields={"a": "entity", "b": "entity"},
            edge_rules=[("a", "b", "custom", 0.7)],
            timestamp_field="ts",
            perturb_target_field="a",
        )
        records = [{"a": "n1", "b": "n2", "ts": datetime(2026, 2, 15, 11, 0, 0)}]
        adapter = GenericMappingAdapter(records=records, spec=spec, graph_id="g-custom")
        out = adapter.build()
        self.assertTrue(out.validation.valid)
        self.assertEqual(out.graph.graph_id, "g-custom")
        self.assertGreaterEqual(len(out.graph.interactions), 1)


if __name__ == "__main__":
    unittest.main()
