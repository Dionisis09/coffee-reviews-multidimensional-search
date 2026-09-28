# indexes/rtree.py (very small R-tree with linear split; simplified)
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import math, heapq
from .base import MultiIndex

def mbr_of_points(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return P.min(axis=0), P.max(axis=0)

def rect_area(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.prod(np.maximum(hi - lo, 0.0)))

def rect_enlargement(lo: np.ndarray, hi: np.ndarray, p: np.ndarray) -> float:
    nlo = np.minimum(lo, p); nhi = np.maximum(hi, p)
    return rect_area(nlo, nhi) - rect_area(lo, hi)

def rect_dist2(q: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    clamped = np.maximum(lo, np.minimum(q, hi))
    d = clamped - q
    return float(np.dot(d, d))

class RNode:
    def __init__(self, leaf: bool, M: int, dim: int):
        self.leaf = leaf
        self.M = M
        self.dim = dim
        self.children: List[RNode] = []
        self.points: List[Tuple[np.ndarray, int]] = []
        self.lo = np.full(dim, np.inf)
        self.hi = np.full(dim, -np.inf)

    def update_mbr(self):
        if self.leaf:
            if not self.points:
                self.lo[:] = np.inf
                self.hi[:] = -np.inf
                return
            P = np.vstack([p for p, _ in self.points])
            self.lo, self.hi = mbr_of_points(P)
        else:
            if not self.children:
                self.lo[:] = np.inf
                self.hi[:] = -np.inf
                return
            lo = np.vstack([c.lo for c in self.children]).min(axis=0)
            hi = np.vstack([c.hi for c in self.children]).max(axis=0)
            self.lo, self.hi = lo, hi

class RTreeIndex(MultiIndex):
    def __init__(self, M: int = 8):
        self.root: Optional[RNode] = None
        self.M = M
        self.dim = 0

    def build(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        self.dim = X.shape[1]
        self.root = RNode(leaf=True, M=self.M, dim=self.dim)
        for p in X:   # index που αποθηκεύουμε στα leaves είναι μόνο τοπικός/προσωρινός
            self.insert(p)

    def _choose_leaf(self, node: RNode, p: np.ndarray) -> RNode:
        if node.leaf:
            return node
        # επέλεξε child με ελάχιστη αύξηση MBR
        best = None; best_inc = math.inf
        for c in node.children:
            inc = rect_enlargement(c.lo, c.hi, p)
            if inc < best_inc:
                best, best_inc = c, inc
        if best is None:
            raise RuntimeError("No child found for leaf selection.")
        return self._choose_leaf(best, p)

    def _split_node(self, node: RNode):
        """Linear split heuristic (χονδρικά: κόβουμε στη μεγαλύτερη διάσταση)."""
        if node.leaf:
            P = np.vstack([pt for pt, _ in node.points])
        else:
            # centers των child MBRs
            P = np.vstack([(c.lo + c.hi) / 2 for c in node.children])

        # NumPy >= 2.0: np.ptp(array, axis=0) αντί για array.ptp(axis=0)
        ranges = np.ptp(P, axis=0)
        axis = int(np.argmax(ranges))
        order = P[:, axis].argsort()
        mid = len(order) // 2
        left_idx = order[:mid]
        right_idx = order[mid:]

        left = RNode(node.leaf, node.M, node.dim)
        right = RNode(node.leaf, node.M, node.dim)

        if node.leaf:
            left.points  = [node.points[i] for i in left_idx]
            right.points = [node.points[i] for i in right_idx]
        else:
            left.children  = [node.children[i] for i in left_idx]
            right.children = [node.children[i] for i in right_idx]

        left.update_mbr(); right.update_mbr()
        return left, right

    def _adjust_tree(self, path: List[RNode]):
        """Ενημέρωσε MBRs προς τα πάνω· κάνε split αν υπάρχει overflow."""
        while path:
            n = path.pop()
            n.update_mbr()

            if n.leaf:
                overflow = len(n.points) > n.M
            else:
                overflow = len(n.children) > n.M
            if not overflow:
                continue

            left, right = self._split_node(n)
            if not path:
                # new root
                new_root = RNode(leaf=False, M=n.M, dim=n.dim)
                new_root.children = [left, right]
                new_root.update_mbr()
                self.root = new_root
                return

            parent = path[-1]
            if parent.leaf:
                raise RuntimeError("Parent of an internal node cannot be leaf.")
            # αντικατάσταση του n από τα left/right
            try:
                parent.children.remove(n)
            except ValueError:
                pass
            parent.children += [left, right]

    def insert(self, x: np.ndarray) -> None:
        p = np.asarray(x, dtype=float).ravel()
        if self.root is None:
            self.root = RNode(leaf=True, M=self.M, dim=len(p))
        # κατέβα μέχρι φύλλο
        path = [self.root]
        cur = self.root
        while cur is not None and not cur.leaf:
            best = None; best_inc = math.inf
            for c in cur.children:
                inc = rect_enlargement(c.lo, c.hi, p)
                if inc < best_inc:
                    best, best_inc = c, inc
            cur = best
            if cur is not None:
                path.append(cur)
        # εισαγωγή
        if cur is not None:
            cur.points.append((p, len(cur.points)))  # (το index εδώ είναι dummy)
            self._adjust_tree(path)
        else:
            raise RuntimeError("No leaf node found for insertion.")

    def _range_dfs(self, node: Optional[RNode], lo: np.ndarray, hi: np.ndarray,
                   out: List[int], counter: List[int]):
        if node is None:
            return
        # prune αν δεν τέμνεται
        if np.any(node.hi < lo) or np.any(node.lo > hi):
            return
        if node.leaf:
            for p, _ in node.points:
                if np.all(p >= lo) and np.all(p <= hi):
                    out.append(counter[0])
                counter[0] += 1
        else:
            for c in node.children:
                self._range_dfs(c, lo, hi, out, counter)

    def range_query(self, lo: np.ndarray, hi: np.ndarray) -> List[int]:
        if self.root is None:
            return []
        lo = np.asarray(lo); hi = np.asarray(hi)
        out: List[int] = []
        self._range_dfs(self.root, lo, hi, out, [0])
        return out

def knn(self, q: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
    if self.root is None:
        return [], []
    q = np.asarray(q, dtype=float).ravel()

    # --- priority queue με tie-breaker ---
    pq: List[Tuple[float, int, RNode]] = []
    node_counter = 0
    heapq.heappush(pq, (rect_dist2(q, self.root.lo, self.root.hi), node_counter, self.root))
    node_counter += 1

    best: List[Tuple[float, int]] = []  # (d2, id_counter) -> έχει ήδη tie-breaker int
    counter = 0

    while pq:
        d, _, node = heapq.heappop(pq)
        if best and d > best[0][0] and len(best) >= k:
            break
        if node.leaf:
            for p, _ in node.points:
                d2 = float(np.dot(p - q, p - q))
                if len(best) < k:
                    heapq.heappush(best, (d2, counter))
                elif d2 < best[0][0]:
                    heapq.heapreplace(best, (d2, counter))
                counter += 1
        else:
            for c in node.children:
                heapq.heappush(pq, (rect_dist2(q, c.lo, c.hi), node_counter, c))
                node_counter += 1

    best.sort(key=lambda t: t[0])
    inds = [i for _, i in best]
    dists = [math.sqrt(t[0]) for t in best]
    return inds, dists



    # Απλά stubs· για το project αρκεί rebuild μετά από batch αλλαγές
    def delete(self, i: int) -> None:
        pass

    def update(self, i: int, x: np.ndarray) -> None:
        pass
