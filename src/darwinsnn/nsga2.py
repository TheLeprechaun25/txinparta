"""NSGA-II selection (Deb et al., 2002), written to be read.

Every function works on an objective matrix F [n_individuals, n_objectives],
where all objectives are *minimised*.
"""

from __future__ import annotations

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """a dominates b: no worse in any objective and strictly better in at least one."""
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_fronts(F: np.ndarray) -> list[np.ndarray]:
    """Split individuals into Pareto fronts: front 0 = non-dominated, front 1 = next, ..."""
    n = len(F)
    dominated_by = [[] for _ in range(n)]   # individuals that i dominates
    n_dominators = np.zeros(n, int)          # how many individuals dominate i
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(F[i], F[j]):
                dominated_by[i].append(j); n_dominators[j] += 1
            elif dominates(F[j], F[i]):
                dominated_by[j].append(i); n_dominators[i] += 1
    fronts, current = [], [i for i in range(n) if n_dominators[i] == 0]
    while current:
        fronts.append(np.array(current))
        nxt = []
        for i in current:
            for j in dominated_by[i]:
                n_dominators[j] -= 1
                if n_dominators[j] == 0:
                    nxt.append(j)
        current = nxt
    return fronts


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """Per-individual crowding distance within one front (larger = more isolated)."""
    n, m = F.shape
    dist = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(m):
        order = np.argsort(F[:, k])
        span = F[order[-1], k] - F[order[0], k]
        dist[order[0]] = dist[order[-1]] = np.inf
        if span > 0:
            dist[order[1:-1]] += (F[order[2:], k] - F[order[:-2], k]) / span
    return dist


def rank_and_crowding(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rank, crowd = np.zeros(len(F), int), np.zeros(len(F))
    for r, front in enumerate(non_dominated_fronts(F)):
        rank[front] = r
        crowd[front] = crowding_distance(F[front])
    return rank, crowd


def select_survivors(F: np.ndarray, n: int) -> np.ndarray:
    """Indices of the n best individuals: by front, then by crowding distance."""
    chosen = []
    for front in non_dominated_fronts(F):
        if len(chosen) + len(front) <= n:
            chosen.extend(front)
        else:
            crowd = crowding_distance(F[front])
            chosen.extend(front[np.argsort(-crowd)][: n - len(chosen)])
            break
    return np.array(chosen)


def tournament(rank: np.ndarray, crowd: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Binary tournaments: lower rank wins, ties broken by larger crowding distance."""
    a, b = rng.integers(len(rank), size=(2, n))
    a_wins = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] >= crowd[b]))
    return np.where(a_wins, a, b)
