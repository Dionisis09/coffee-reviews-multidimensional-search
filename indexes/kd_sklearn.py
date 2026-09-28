# indexes/kd_sklearn.py (optional wrapper using sklearn KDTree)
from __future__ import annotations
from typing import List, Tuple
import numpy as np
try:
    from sklearn.neighbors import KDTree
except Exception:
    KDTree = None

from .base import MultiIndex

class KDIndex(MultiIndex):
    def __init__(self) -> None:
        self.tree = None
        self.X = None

    def build(self, X: np.ndarray) -> None:
        if KDTree is None:
            raise ImportError("scikit-learn not available")
        self.X = np.asarray(X); self.tree = KDTree(self.X)

    def knn(self, q: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        if self.tree is None: return [], []
        dist, ind = self.tree.query(np.asarray(q).reshape(1,-1), k=k)
        return ind[0].astype(int).tolist(), dist[0].astype(float).tolist()

    # simple stubs
    def insert(self, x: np.ndarray) -> None:
        self.X = np.vstack([self.X, x.reshape(1,-1)]); self.build(self.X)
    def delete(self, i: int) -> None:
        mask = np.ones(len(self.X), dtype=bool); mask[i]=False; self.build(self.X[mask])
    def update(self, i: int, x: np.ndarray) -> None:
        self.X[i]=x; self.build(self.X)
    def range_query(self, lo: np.ndarray, hi: np.ndarray) -> List[int]:
        X=self.X
        mask = np.logical_and(X>=lo, X<=hi).all(axis=1)
        return np.nonzero(mask)[0].astype(int).tolist()
