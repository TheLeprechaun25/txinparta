"""Crossover operators for motif networks.

Two networks of arbitrary motifs have no node-to-node correspondence, so we
cross over *parts* instead of aligned genes:

    child = part of parent A  +  part of parent B

  * motifs of both parts are copied with all their parameters
  * links inside a part are inherited unchanged
  * links that left a part (their target was not inherited) are *cut* and
    re-pointed to a random motif of the other parent's part, so every motif
    keeps its out-degree and the two parts become connected
  * input links of inherited motifs are inherited

Two ways to choose the parts:

  region_crossover    a connected region (BFS over the motif graph) of each
                      parent; works for any genome
  cluster_crossover   whole clusters of each parent (needs n_clusters > 1)

The fraction of cut links measures how disruptive a crossover is. The idea
behind clustered genomes is that few links run between clusters, so swapping
clusters cuts little and children keep their parents' working sub-circuits.
"""

from __future__ import annotations

import copy

import numpy as np

from .config import GenomeConfig
from .genome import Genome, connect


def combine(a: Genome, part_a: set[int], b: Genome, part_b: set[int], gc: GenomeConfig,
            rng: np.random.Generator) -> tuple[Genome, dict]:
    child = Genome(n_channels=a.n_channels)
    # keep the two parents' clusters apart; unstructured genomes stay a single cluster
    clustered = gc.n_clusters[1] > 1
    cluster_offset = {"a": 0,
                      "b": 1 + max((m.cluster for m in a.motifs.values()), default=0)
                      if clustered else 0}
    uid_map: dict[str, dict[int, int]] = {}
    for tag, parent, part in (("a", a, part_a), ("b", b, part_b)):
        uid_map[tag] = {}
        for uid in parent.motifs:                  # keep the parent's order
            if uid in part:
                gene = copy.deepcopy(parent.motifs[uid])
                gene.cluster += cluster_offset[tag]
                uid_map[tag][uid] = child.add_motif(gene)

    kept = cut = 0
    for tag, other, parent in (("a", "b", a), ("b", "a", b)):
        m, targets = uid_map[tag], list(uid_map[other].values())
        for (src, dst), link in parent.links.items():
            if src in m and dst in m:
                child.links[(m[src], m[dst])] = copy.deepcopy(link)
                kept += 1
            elif src in m:
                cut += 1
                if targets:
                    connect(child, m[src], int(rng.choice(targets)), gc, rng)
        for (channel, dst), link in parent.input_links.items():
            if dst in m:
                child.input_links[(channel, m[dst])] = copy.deepcopy(link)

    child.prune_unreachable()
    _renumber_clusters(child)
    return child, {"links_kept": kept, "links_cut": cut,
                   "cut_fraction": cut / (kept + cut) if kept + cut else 0.0}


def _renumber_clusters(g: Genome) -> None:
    """Make cluster ids 0..k-1 again after combining two parents."""
    new_id = {c: i for i, c in enumerate(sorted({m.cluster for m in g.motifs.values()}))}
    for m in g.motifs.values():
        m.cluster = new_id[m.cluster]


def _split_fraction(rng) -> float:
    """Share of the child that comes from parent A."""
    return float(rng.uniform(0.3, 0.7))


# ------------------------------------------------------------------- regions
def connected_region(g: Genome, size: int, rng: np.random.Generator) -> set[int]:
    """`size` motifs grown by breadth-first search over the (undirected) motif graph."""
    neighbours: dict[int, list[int]] = {u: [] for u in g.motifs}
    for src, dst in g.links:
        neighbours[src].append(dst)
        neighbours[dst].append(src)
    region: set[int] = set()
    unvisited = list(g.motifs)
    while len(region) < size and unvisited:
        start = unvisited[int(rng.integers(len(unvisited)))]
        frontier = [start]
        region.add(start)
        while frontier and len(region) < size:
            nxt = []
            for u in frontier:
                for v in rng.permutation(neighbours[u]):
                    if int(v) not in region and len(region) < size:
                        region.add(int(v))
                        nxt.append(int(v))
            frontier = nxt
        unvisited = [u for u in unvisited if u not in region]
    return region


def region_crossover(a: Genome, b: Genome, gc: GenomeConfig,
                     rng: np.random.Generator) -> tuple[Genome, dict]:
    f = _split_fraction(rng)
    part_a = connected_region(a, max(1, round(f * a.n_motifs)), rng)
    part_b = connected_region(b, max(1, round((1 - f) * b.n_motifs)), rng)
    return combine(a, part_a, b, part_b, gc, rng)


# ------------------------------------------------------------------ clusters
def cluster_crossover(a: Genome, b: Genome, gc: GenomeConfig,
                      rng: np.random.Generator) -> tuple[Genome, dict]:
    f = _split_fraction(rng)
    parts = []
    for parent, share in ((a, f), (b, 1 - f)):
        clusters = parent.clusters()
        ids = list(clusters)
        k = min(len(ids), max(1, round(share * len(ids))))
        chosen = rng.choice(ids, size=k, replace=False)
        parts.append({u for c in chosen for u in clusters[int(c)]})
    return combine(a, parts[0], b, parts[1], gc, rng)


CROSSOVERS = {"region": region_crossover, "cluster": cluster_crossover}
