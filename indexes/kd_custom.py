# indexes/kd_custom.py
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import heapq
from .base import MultiIndex

class KDNode:
    __slots__ = ("point","index","left","right","axis","bbox_min","bbox_max")
    def __init__(self, point: np.ndarray, index: int, axis: int):
        self.point = point
        self.index = index
        self.axis = axis
        self.left: Optional[KDNode] = None
        self.right: Optional[KDNode] = None
        self.bbox_min = point.copy()
        self.bbox_max = point.copy()

def build_kd(points: np.ndarray, idxs: np.ndarray, depth: int=0) -> Optional[KDNode]:
    if len(points)==0: return None
    k = points.shape[1]
    axis = depth % k
    order = points[:,axis].argsort()
    mid = len(points)//2
    m_idx = order[mid]
    node = KDNode(points[m_idx], int(idxs[m_idx]), axis)
    left  = build_kd(points[order[:mid]],  idxs[order[:mid]],  depth+1)
    right = build_kd(points[order[mid+1:]], idxs[order[mid+1:]], depth+1)
    node.left, node.right = left, right
    # update bbox
    for child in (left, right):
        if child is not None:
            node.bbox_min = np.minimum(node.bbox_min, child.bbox_min)
            node.bbox_max = np.maximum(node.bbox_max, child.bbox_max)
    return node

def bbox_dist2(point: np.ndarray, node: KDNode) -> float:
    # squared distance from point to bbox
    clamped = np.maximum(node.bbox_min, np.minimum(point, node.bbox_max))
    d = clamped - point
    return float(np.dot(d, d))

def dist2(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b; return float(np.dot(d,d))

class KDTreeIndex(MultiIndex):
    def __init__(self) -> None:
        self.root: Optional[KDNode] = None
        self.X: Optional[np.ndarray] = None
        self.idxs: Optional[np.ndarray] = None

    def build(self, X: np.ndarray) -> None:
        self.X = np.asarray(X, dtype=float)
        self.idxs = np.arange(len(self.X), dtype=int)
        self.root = build_kd(self.X, self.idxs)

    def _knn(self, node: Optional[KDNode], q: np.ndarray, k: int, heap: List[Tuple[float,int]]):
        if node is None: return
        # Max-heap (negated distances)
        d2 = dist2(q, node.point)
        if len(heap) < k:
            heapq.heappush(heap, (-d2, node.index))
        else:
            if d2 < -heap[0][0]:
                heapq.heapreplace(heap, (-d2, node.index))
        # Choose branch
        axis = node.axis
        go_left = q[axis] <= node.point[axis]
        first, second = (node.left, node.right) if go_left else (node.right, node.left)
        self._knn(first, q, k, heap)
        # Check if we need to visit the other side
        if len(heap) < k or (q[axis]-node.point[axis])**2 < -heap[0][0]:
            # bbox pruning on second
            if second is not None:
                if len(heap) < k or bbox_dist2(q, second) < -heap[0][0]:
                    self._knn(second, q, k, heap)

    def knn(self, q: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        if self.root is None: return [], []
        q = np.asarray(q, dtype=float).ravel()
        heap: List[Tuple[float,int]] = []
        self._knn(self.root, q, k, heap)
        heap.sort(key=lambda x: -x[0])  # smallest first
        inds = [idx for (_, idx) in heap]
        dists = [(-h[0])**0.5 for h in heap]
        return inds, dists

    def range_query(self, lo: np.ndarray, hi: np.ndarray) -> List[int]:
        lo = np.asarray(lo, dtype=float); hi=np.asarray(hi, dtype=float)
        out: List[int] = []
        def dfs(node: Optional[KDNode]):
            if node is None: return
            # bbox test
            if np.any(node.bbox_max < lo) or np.any(node.bbox_min > hi):
                return
            if np.all(node.point >= lo) and np.all(node.point <= hi):
                out.append(node.index)
            dfs(node.left); dfs(node.right)
        dfs(self.root)
        return out

    # simple rebuild strategies
    def insert(self, x: np.ndarray) -> None:
        if self.X is None:
            self.build(np.asarray(x).reshape(1,-1)); return
        self.X = np.vstack([self.X, np.asarray(x).reshape(1,-1)])
        self.build(self.X)

    def delete(self, i: int) -> None:
        if self.X is None: return
        mask = np.ones(len(self.X), dtype=bool); mask[i]=False
        self.X = self.X[mask]; self.build(self.X)

    def update(self, i: int, x: np.ndarray) -> None:
        if self.X is None: return
        self.X[i] = np.asarray(x)
        self.build(self.X)
