from datetime import datetime, timezone
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag import Actant, Interaction, LayeredGraph


class ModelValidationTests(unittest.TestCase):
    def test_actant_requires_id(self) -> None:
        with self.assertRaises(ValueError):
            Actant(actant_id="", kind="person", label="x")

    def test_interaction_weight_validation(self) -> None:
        with self.assertRaises(ValueError):
            Interaction(
                interaction_id="i1",
                timestamp=datetime.now(timezone.utc),
                source_id="a",
                target_id="b",
                layer="social",
                weight=-0.1,
            )

    def test_layered_graph_requires_version(self) -> None:
        with self.assertRaises(ValueError):
            LayeredGraph(graph_id="g1", schema_version="")


if __name__ == "__main__":
    unittest.main()
