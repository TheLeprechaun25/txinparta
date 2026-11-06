"""Genotype: a network described as motifs plus links between them.

    Genome
     ├─ motifs        uid -> MotifGene   parameters of each neuron of one motif
     ├─ links         (src uid, dst uid) -> Link   motif-to-motif synapse bundle
     └─ input_links   (channel, dst uid) -> Link   input-channel-to-motif bundle

Motifs are identified by a stable integer `uid`, so adding or removing
motifs never renumbers anything.

Motifs can be grouped into *clusters* (GenomeConfig.n_clusters > 1). Links are
then mostly inside a cluster and only a fraction p_inter_cluster go between
clusters, so the network is modular. Clusters make crossover cheaper: a child
can inherit whole clusters with only a few links cut (crossover.py). With one
cluster (the default) the genome is unstructured.

A Link holds one delay (and optionally one weight) per synapse, in an array
shaped [multiplicity, n_src, n_dst]:
  * motif link: n_src = outputs of the source motif, n_dst = inputs of the target motif
  * input link: n_src = 1 (the channel),             n_dst = inputs of the target motif
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import GenomeConfig
from .motifs import MOTIFS, motif_ids
from .network import Network


@dataclass
class MotifGene:
    kind: int                       # index into motifs.MOTIFS
    v_thresh: np.ndarray            # [n_neurons]
    v_rest: np.ndarray              # [n_neurons]
    refractory: np.ndarray          # [n_neurons] int
    delays: np.ndarray              # [n_internal_edges] int
    weights: np.ndarray | None      # [n_internal_edges] magnitudes, or None
    cluster: int = 0                # cluster the motif belongs to


@dataclass
class Link:
    delays: np.ndarray              # [multiplicity, n_src, n_dst] int
    weights: np.ndarray | None      # same shape, magnitudes, or None

    @property
    def n_synapses(self) -> int:
        return self.delays.size


@dataclass
class Genome:
    n_channels: int
    motifs: dict[int, MotifGene] = field(default_factory=dict)
    links: dict[tuple[int, int], Link] = field(default_factory=dict)
    input_links: dict[tuple[int, int], Link] = field(default_factory=dict)
    next_uid: int = 0

    # ---------------------------------------------------------------- sizes
    @property
    def n_motifs(self) -> int:
        return len(self.motifs)

    @property
    def n_neurons(self) -> int:
        return sum(MOTIFS[m.kind].n_neurons for m in self.motifs.values())

    @property
    def n_synapses(self) -> int:
        internal = sum(len(MOTIFS[m.kind].edges) for m in self.motifs.values())
        external = sum(l.n_synapses for l in self.links.values())
        inputs = sum(l.n_synapses for l in self.input_links.values())
        return internal + external + inputs

    def motif_counts(self) -> np.ndarray:
        """How many motifs of each type the genome contains."""
        return np.bincount([m.kind for m in self.motifs.values()], minlength=len(MOTIFS))

    def clusters(self) -> dict[int, list[int]]:
        """cluster id -> uids of its motifs."""
        out: dict[int, list[int]] = {}
        for uid, m in self.motifs.items():
            out.setdefault(m.cluster, []).append(uid)
        return out

    def n_inter_cluster_links(self) -> int:
        return sum(self.motifs[s].cluster != self.motifs[d].cluster for s, d in self.links)

    def copy(self) -> Genome:
        return copy.deepcopy(self)

    # -------------------------------------------------------------- editing
    def add_motif(self, gene: MotifGene) -> int:
        uid = self.next_uid
        self.motifs[uid] = gene
        self.next_uid += 1
        return uid

    def remove_motifs(self, uids) -> None:
        uids = set(uids)
        for uid in uids:
            self.motifs.pop(uid, None)
        self.links = {k: v for k, v in self.links.items() if k[0] not in uids and k[1] not in uids}
        self.input_links = {k: v for k, v in self.input_links.items() if k[1] not in uids}

    def prune_unreachable(self) -> int:
        """Remove motifs that no input spike can ever reach (they would stay silent).

        Returns the number of motifs removed.
        """
        children: dict[int, list[int]] = {}
        for src, dst in self.links:
            children.setdefault(src, []).append(dst)
        reached = {dst for _, dst in self.input_links}
        queue = deque(reached)
        while queue:
            for nxt in children.get(queue.popleft(), []):
                if nxt not in reached:
                    reached.add(nxt)
                    queue.append(nxt)
        dead = set(self.motifs) - reached
        self.remove_motifs(dead)
        return len(dead)


# ===================================================================== sampling
def low_biased_ints(low: int, high: int, size, rng: np.random.Generator) -> np.ndarray:
    """Integers in [low, high] where small values are more likely (geometric).

    Used for refractory periods and delays, where short values are the most
    common in biology.
    """
    return low + np.minimum(rng.geometric(0.4, size=size) - 1, high - low)


def new_motif_gene(kind: int, cfg: GenomeConfig, rng: np.random.Generator,
                   cluster: int = 0) -> MotifGene:
    motif = MOTIFS[kind]
    n, e = motif.n_neurons, len(motif.edges)
    return MotifGene(
        kind=kind,
        v_thresh=rng.uniform(*cfg.v_thresh, size=n),
        v_rest=rng.uniform(*cfg.v_rest, size=n),
        refractory=low_biased_ints(*cfg.refractory, size=n, rng=rng),
        delays=low_biased_ints(*cfg.delay, size=e, rng=rng),
        weights=rng.uniform(*cfg.weight, size=e) if cfg.evolve_weights else None,
        cluster=cluster,
    )


def new_link(n_src: int, n_dst: int, multiplicity: int, cfg: GenomeConfig,
             rng: np.random.Generator) -> Link:
    shape = (multiplicity, n_src, n_dst)
    return Link(
        delays=low_biased_ints(*cfg.delay, size=shape, rng=rng),
        weights=rng.uniform(*cfg.weight, size=shape) if cfg.evolve_weights else None,
    )


def connect(genome: Genome, src: int, dst: int, cfg: GenomeConfig,
            rng: np.random.Generator) -> None:
    """Create (or replace) the link src motif -> dst motif."""
    n_out = len(MOTIFS[genome.motifs[src].kind].outputs)
    n_in = len(MOTIFS[genome.motifs[dst].kind].inputs)
    mult = int(rng.integers(cfg.multiplicity[0], cfg.multiplicity[1] + 1))
    genome.links[(src, dst)] = new_link(n_out, n_in, mult, cfg, rng)


def connect_input(genome: Genome, channel: int, dst: int, cfg: GenomeConfig,
                  rng: np.random.Generator) -> None:
    """Create (or replace) the link input channel -> dst motif."""
    n_in = len(MOTIFS[genome.motifs[dst].kind].inputs)
    genome.input_links[(channel, dst)] = new_link(1, n_in, 1, cfg, rng)


def choose_partners(genome: Genome, cluster: int, k: int, cfg: GenomeConfig,
                    rng: np.random.Generator, exclude=()) -> list[int]:
    """Pick k distinct motifs to link with a motif of `cluster`.

    Each partner comes from another cluster with probability p_inter_cluster
    and from the same cluster otherwise. If the preferred group is empty, the
    other group is used.
    """
    exclude = set(exclude)
    same = [u for u, m in genome.motifs.items() if m.cluster == cluster and u not in exclude]
    other = [u for u, m in genome.motifs.items() if m.cluster != cluster and u not in exclude]
    chosen: list[int] = []
    for _ in range(min(k, len(same) + len(other))):
        inter = rng.random() < cfg.p_inter_cluster
        pool = (other if inter else same) or (same if inter else other)
        pick = pool.pop(int(rng.integers(len(pool))))
        chosen.append(int(pick))
    return chosen


def random_genome(cfg: GenomeConfig, n_channels: int, rng: np.random.Generator) -> Genome:
    """Random initial individual: motifs of random types, random sparse wiring.

    With n_clusters > 1, motifs are spread evenly over the clusters and links
    are mostly inside a cluster (see choose_partners).
    """
    allowed = motif_ids(cfg.motif_types)
    genome = Genome(n_channels=n_channels)
    n = int(rng.integers(cfg.n_motifs[0], cfg.n_motifs[1] + 1))
    n_clusters = int(rng.integers(cfg.n_clusters[0], cfg.n_clusters[1] + 1))
    uids = [genome.add_motif(new_motif_gene(int(rng.choice(allowed)), cfg, rng, i % n_clusters))
            for i in range(n)]

    for src in uids:
        k = int(rng.integers(cfg.fan_out[0], cfg.fan_out[1] + 1))
        for dst in choose_partners(genome, genome.motifs[src].cluster, k, cfg, rng, exclude={src}):
            connect(genome, src, dst, cfg, rng)

    for ch in range(n_channels):
        k = min(int(rng.integers(cfg.input_fan_out[0], cfg.input_fan_out[1] + 1)), n)
        for dst in rng.choice(uids, size=k, replace=False):
            connect_input(genome, ch, int(dst), cfg, rng)

    genome.prune_unreachable()
    return genome


# ===================================================================== decoding
def build_network(genome: Genome) -> Network:
    """Expand a genome into the flat arrays that the simulators run."""
    C = genome.n_channels
    offset: dict[int, int] = {}
    v_thresh, v_rest, refr, sign, motif_of, is_out = [], [], [], [], [], []
    n = 0
    for idx, (uid, gene) in enumerate(genome.motifs.items()):
        motif = MOTIFS[gene.kind]
        offset[uid] = n
        v_thresh.append(gene.v_thresh)
        v_rest.append(gene.v_rest)
        refr.append(gene.refractory)
        sign.append(np.array(motif.signs))
        motif_of.append(np.full(motif.n_neurons, idx))
        is_out.append(np.isin(np.arange(motif.n_neurons), motif.outputs))
        n += motif.n_neurons
    neuron_sign = np.concatenate(sign) if sign else np.zeros(0, int)

    pre, post, delay, weight = [], [], [], []
    evolve_w = any(g.weights is not None for g in genome.motifs.values())

    def add(p, q, d, w):
        pre.append(np.ravel(p)); post.append(np.ravel(q)); delay.append(np.ravel(d))
        if evolve_w:
            weight.append(np.ravel(w))

    # internal motif synapses
    for uid, gene in genome.motifs.items():
        edges = np.array(MOTIFS[gene.kind].edges).reshape(-1, 2)
        add(C + offset[uid] + edges[:, 0], offset[uid] + edges[:, 1], gene.delays, gene.weights)

    # motif -> motif: every output neuron of src to every input neuron of dst
    for (src, dst), link in genome.links.items():
        outs = offset[src] + np.array(MOTIFS[genome.motifs[src].kind].outputs)
        ins = offset[dst] + np.array(MOTIFS[genome.motifs[dst].kind].inputs)
        m = link.delays.shape[0]
        p = np.broadcast_to(C + outs[None, :, None], (m, len(outs), len(ins)))
        q = np.broadcast_to(ins[None, None, :], (m, len(outs), len(ins)))
        add(p, q, link.delays, link.weights)

    # input channel -> every input neuron of dst
    for (ch, dst), link in genome.input_links.items():
        ins = offset[dst] + np.array(MOTIFS[genome.motifs[dst].kind].inputs)
        add(np.full(link.delays.shape, ch), np.broadcast_to(ins, link.delays.shape),
            link.delays, link.weights)

    def cat(xs, dt):
        return np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)

    pre, post, delay = cat(pre, np.int32), cat(post, np.int32), cat(delay, np.int32)
    order = np.argsort(post, kind="stable")
    syn_sign = np.where(pre >= C, neuron_sign[np.maximum(pre - C, 0)] if n else 1, 1)

    return Network(
        n_inputs=C,
        v_thresh=cat(v_thresh, np.float32), v_rest=cat(v_rest, np.float32),
        refractory=cat(refr, np.int32), sign=neuron_sign.astype(np.int8),
        motif_of_neuron=cat(motif_of, np.int32), is_output=cat(is_out, bool),
        pre=pre[order], post=post[order], delay=delay[order],
        weight_magnitude=cat(weight, np.float32)[order] if evolve_w else None,
        syn_sign=syn_sign[order].astype(np.int8),
    )
