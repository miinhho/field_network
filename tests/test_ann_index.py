from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffrag.ann_index import create_cosine_ann_index


class AnnIndexTests(unittest.TestCase):
    def test_exact_backend_query(self) -> None:
        idx = create_cosine_ann_index(backend="exact")
        ids = ["a", "b", "c"]
        vecs = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        idx.fit(ids, vecs)
        hits = idx.query(np.array([1.0, 0.0, 0.0], dtype=np.float64), k=2)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].item_id, "a")

    def test_auto_backend_falls_back_to_exact(self) -> None:
        idx = create_cosine_ann_index(backend="auto")
        self.assertIn(idx.backend_name, {"faiss", "exact"})

    def test_strict_faiss_raises_when_missing(self) -> None:
        with mock.patch("importlib.import_module", side_effect=ModuleNotFoundError("faiss missing")):
            with self.assertRaises(ModuleNotFoundError):
                create_cosine_ann_index(
                    backend="faiss",
                    allow_exact_fallback=False,
                )

    def test_opt_in_fallback_uses_exact_when_faiss_missing(self) -> None:
        with mock.patch("importlib.import_module", side_effect=ModuleNotFoundError("faiss missing")):
            idx = create_cosine_ann_index(
                backend="faiss",
                allow_exact_fallback=True,
            )
            self.assertEqual(idx.backend_name, "exact")


if __name__ == "__main__":
    unittest.main()
