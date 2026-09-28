# indexes/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class MultiIndex(ABC):
    """Common interface for multi-dimensional indexes."""

    @abstractmethod
    def build(self, X: np.ndarray) -> None:
        raise NotImplementedError

    def insert(self, x: np.ndarray) -> None:
        raise NotImplementedError

    def delete(self, i: int) -> None:
        raise NotImplementedError

    def update(self, i: int, x: np.ndarray) -> None:
        raise NotImplementedError

    def range_query(self, lo: np.ndarray, hi: np.ndarray) -> List[int]:
        raise NotImplementedError

    def knn(self, q: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        raise NotImplementedError
