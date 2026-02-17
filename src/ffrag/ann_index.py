from __future__ import annotations

from dataclasses import dataclass
import importlib

import numpy as np


@dataclass(slots=True)
class AnnQueryHit:
    item_id: str
    score: float


class BaseAnnIndex:
    backend_name: str = "base"

    def fit(self, item_ids: list[str], vectors: np.ndarray) -> None:
        raise NotImplementedError

    def query(self, vector: np.ndarray, k: int) -> list[AnnQueryHit]:
        raise NotImplementedError


class ExactCosineIndex(BaseAnnIndex):
    backend_name = "exact"

    def __init__(self) -> None:
        self._item_ids: list[str] = []
        self._vectors: np.ndarray | None = None

    def fit(self, item_ids: list[str], vectors: np.ndarray) -> None:
        self._item_ids = list(item_ids)
        self._vectors = self._normalize(vectors)

    def query(self, vector: np.ndarray, k: int) -> list[AnnQueryHit]:
        if self._vectors is None or not self._item_ids:
            return []
        q = self._normalize(vector.reshape(1, -1))[0]
        sims = self._vectors @ q
        order = np.argsort(-sims)
        limit = max(0, min(int(k), len(order)))
        out: list[AnnQueryHit] = []
        for idx in order[:limit]:
            out.append(AnnQueryHit(item_id=self._item_ids[int(idx)], score=float(sims[int(idx)])))
        return out

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        x = arr.astype(np.float64, copy=False)
        denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / denom


class FaissCosineIndex(BaseAnnIndex):
    backend_name = "faiss"

    def __init__(self, use_ivf: bool = False, nlist: int = 64, nprobe: int = 8) -> None:
        self._use_ivf = bool(use_ivf)
        self._nlist = max(8, int(nlist))
        self._nprobe = max(1, int(nprobe))
        self._item_ids: list[str] = []
        self._index = None
        self._faiss = importlib.import_module("faiss")

    def fit(self, item_ids: list[str], vectors: np.ndarray) -> None:
        self._item_ids = list(item_ids)
        x = self._normalize(vectors).astype(np.float32, copy=False)
        if x.ndim != 2:
            raise ValueError("vectors must be 2D")
        d = int(x.shape[1])
        if self._use_ivf and len(item_ids) >= self._nlist * 2:
            quantizer = self._faiss.IndexFlatIP(d)
            index = self._faiss.IndexIVFFlat(quantizer, d, self._nlist, self._faiss.METRIC_INNER_PRODUCT)
            index.train(x)
            index.add(x)
            index.nprobe = min(self._nprobe, self._nlist)
            self._index = index
            return
        index = self._faiss.IndexFlatIP(d)
        index.add(x)
        self._index = index

    def query(self, vector: np.ndarray, k: int) -> list[AnnQueryHit]:
        if self._index is None or not self._item_ids:
            return []
        q = self._normalize(vector.reshape(1, -1)).astype(np.float32, copy=False)
        kk = max(1, min(int(k), len(self._item_ids)))
        scores, labels = self._index.search(q, kk)
        out: list[AnnQueryHit] = []
        for i in range(len(labels[0])):
            label = int(labels[0][i])
            if label < 0 or label >= len(self._item_ids):
                continue
            score = float(scores[0][i])
            out.append(AnnQueryHit(item_id=self._item_ids[label], score=max(-1.0, min(1.0, score))))
        return out

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        x = arr.astype(np.float64, copy=False)
        denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / denom


def create_cosine_ann_index(
    backend: str = "auto",
    allow_exact_fallback: bool = False,
    faiss_use_ivf: bool = False,
    faiss_nlist: int = 64,
    faiss_nprobe: int = 8,
) -> BaseAnnIndex:
    mode = backend.strip().lower()
    if mode not in {"auto", "faiss", "exact"}:
        mode = "auto"
    if mode == "exact":
        return ExactCosineIndex()
    if mode in {"auto", "faiss"}:
        try:
            return FaissCosineIndex(
                use_ivf=faiss_use_ivf,
                nlist=faiss_nlist,
                nprobe=faiss_nprobe,
            )
        except ModuleNotFoundError:
            if not allow_exact_fallback:
                raise
    if allow_exact_fallback:
        return ExactCosineIndex()
    raise RuntimeError("FAISS backend is required but unavailable. Fix environment or enable exact fallback.")
