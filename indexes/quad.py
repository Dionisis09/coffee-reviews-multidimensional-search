# indexes/quad.py (2D Quadtree; for d>2 falls back to brute-force for knn)
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import heapq
from .base import MultiIndex

class QuadNode:
    __slots__=("xmin","ymin","xmax","ymax","points","children","leaf")
    def __init__(self, xmin, ymin, xmax, ymax, capacity=16):
        self.xmin,self.ymin,self.xmax,self.ymax = xmin,ymin,xmax,ymax
        self.points: List[Tuple[float,float,int]] = []
        self.children: List[QuadNode] = []
        self.leaf = True
        self.capacity=capacity

    def subdivide(self):
        xm=(self.xmin+self.xmax)/2; ym=(self.ymin+self.ymax)/2
        self.children=[
            QuadNode(self.xmin,self.ymin,xm,ym,self.capacity),
            QuadNode(xm,self.ymin,self.xmax,ym,self.capacity),
            QuadNode(self.xmin,ym,xm,self.ymax,self.capacity),
            QuadNode(xm,ym,self.xmax,self.ymax,self.capacity),
        ]
        self.leaf=False

    def contains(self, x,y):
        return (self.xmin <= x <= self.xmax) and (self.ymin <= y <= self.ymax)

def rect_dist2(x,y,node: QuadNode)->float:
    cx = min(max(x, node.xmin), node.xmax)
    cy = min(max(y, node.ymin), node.ymax)
    dx = cx - x; dy = cy - y
    return dx*dx + dy*dy

class QuadIndex(MultiIndex):
    def __init__(self) -> None:
        self.root: Optional[QuadNode] = None
        self.X: Optional[np.ndarray] = None
        self.dim = 2

    def build(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        self.X = X
        if X.shape[1] != 2:
            # we will still keep X for brute-force knn/range
            self.root = None
            self.dim = X.shape[1]
            return
        self.dim = 2
        xmin, ymin = X.min(axis=0)
        xmax, ymax = X.max(axis=0)
        pad = 1e-6
        self.root = QuadNode(xmin-pad, ymin-pad, xmax+pad, ymax+pad, capacity=16)
        for i,(x,y) in enumerate(X):
            self._insert(self.root, x,y, int(i))

    def _insert(self, node: QuadNode, x:float, y:float, idx:int):
        if node.leaf and len(node.points) < node.capacity:
            node.points.append((x,y,idx))
            return
        if node.leaf:
            node.subdivide()
            # redistribute
            for px,py,pi in node.points:
                self._insert(next(n for n in node.children if n.contains(px,py)), px,py,pi)
            node.points.clear()
        # insert into appropriate child
        child = next(n for n in node.children if n.contains(x,y))
        self._insert(child, x,y,idx)

    def knn(self, q: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        q = np.asarray(q, dtype=float).ravel()
        if self.root is None or self.dim != 2 or self.X is None:
            # brute-force fallback
            d2 = ((self.X - q)**2).sum(axis=1)
            ind = np.argsort(d2)[:k]
            return ind.astype(int).tolist(), np.sqrt(d2[ind]).astype(float).tolist()
        x,y = q.tolist()
        heap: List[Tuple[float, QuadNode]] = [(0.0, self.root)]
        best: List[Tuple[float,int]] = []  # max-heap on -dist2
        while heap:
            _, node = heapq.heappop(heap)
            if node.leaf:
                for px,py,pi in node.points:
                    d2 = (px-x)*(px-x) + (py-y)*(py-y)
                    if len(best)<k: heapq.heappush(best, (-d2, pi))
                    elif d2 < -best[0][0]: heapq.heapreplace(best, (-d2, pi))
            else:
                for ch in node.children:
                    heapq.heappush(heap, (rect_dist2(x,y,ch), ch))
        best.sort(key=lambda t: -t[0])
        inds = [i for _,i in best]
        dists = [(-t[0])**0.5 for t in best]
        return inds, dists

    def range_query(self, lo: np.ndarray, hi: np.ndarray) -> List[int]:
        lo=np.asarray(lo); hi=np.asarray(hi)
        if self.root is None or self.dim != 2 or self.X is None:
            mask = np.logical_and(self.X>=lo, self.X<=hi).all(axis=1)
            return np.nonzero(mask)[0].astype(int).tolist()
        out: List[int] = []
        def overlap(node: QuadNode):
            return not (node.xmax < lo[0] or node.xmin > hi[0] or node.ymax < lo[1] or node.ymin > hi[1])
        def dfs(node: QuadNode):
            if not overlap(node): return
            if node.leaf:
                for x,y,i in node.points:
                    if lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]: out.append(i)
            else:
                for ch in node.children: dfs(ch)
        dfs(self.root)
        return out

    def insert(self, x: np.ndarray) -> None:
        self.build(np.vstack([self.X, x.reshape(1,-1)]))

    def delete(self, i: int) -> None:
        X = np.delete(self.X, i, axis=0); self.build(X)

    def update(self, i: int, x: np.ndarray) -> None:
        X = self.X.copy(); X[i]=x; self.build(X)
