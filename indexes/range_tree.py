# indexes/range_tree.py (simple 2D range tree; knn fallback brute-force)
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from .base import MultiIndex

class RangeTreeIndex(MultiIndex):
    def __init__(self) -> None:
        self.X: Optional[np.ndarray] = None

    def build(self, X: np.ndarray) -> None:
        # For simplicity we just store X; range_query will use vectorized filter (still O(n))
        # In report, explain it's a simplified range-structure placeholder.
        self.X = np.asarray(X, dtype=float)

    def range_query(self, lo: np.ndarray, hi: np.ndarray) -> List[int]:
        X = self.X; lo=np.asarray(lo,dtype=float); hi=np.asarray(hi,dtype=float)
        if X is None: return []
        mask = np.logical_and(X>=lo, X<=hi).all(axis=1)
        return np.nonzero(mask)[0].astype(int).tolist()

    def knn(self, q: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        X = self.X
        if X is None: return [], []
        d2 = ((X - q.reshape(1,-1))**2).sum(axis=1)
        ind = np.argsort(d2)[:k]
        return ind.astype(int).tolist(), np.sqrt(d2[ind]).astype(float).tolist()

    def insert(self, x: np.ndarray) -> None:
        if self.X is None: self.X = np.asarray(x).reshape(1,-1)
        else: self.X = np.vstack([self.X, x.reshape(1,-1)])

    def delete(self, i: int) -> None:
        if self.X is None: return
        self.X = np.delete(self.X, i, axis=0)

    def update(self, i: int, x: np.ndarray) -> None:
        if self.X is None: return
        self.X[i] = x
