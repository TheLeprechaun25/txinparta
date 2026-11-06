"""Mutation operators. Each one edits a genome in place.

Structure:   add_motifs, remove_motifs, add_links, remove_links, rewire_inputs
Parameters:  neuron_params (threshold / rest / refractory), delays, weights

`mutate` copies the parent, applies `n_operators` operators picked at random
(weighted by the p_* values in MutationConfig), and removes any motif left
unreachable from the inputs. Crossover is optional and lives in crossover.py.

Structural operators respect clusters: a new motif joins an existing cluster,
and new links stay inside the cluster except for a p_inter_cluster fraction.
"""

from __future__ import annotations

import numpy as np

from .config import GenomeConfig, MutationConfig
from .genome import Genome, choose_partners, connect, connect_input, new_motif_gene
from .motifs import motif_ids


def _how_many(total: int, fraction: tuple[float, float], rng) -> int:
    """Number of elements to change: a random fraction of `total`, at least 1."""
    return max(1, int(round(total * rng.uniform(*fraction))))


# ---------------------------------------------------------------- structure
def add_motifs(g: Genome, gc: GenomeConfig, mc: MutationConfig, rng) -> None:
    """Add new motifs, each fed by existing motifs (or inputs) and feeding existing motifs."""
    allowed = motif_ids(gc.motif_types)
    existing = list(g.motifs)
    for _ in range(_how_many(len(existing), mc.motif_fraction, rng)):
        cluster = g.motifs[int(rng.choice(existing))].cluster if existing else 0
        uid = g.add_motif(new_motif_gene(int(rng.choice(allowed)), gc, rng, cluster))
        k_in = int(rng.integers(gc.fan_out[0], gc.fan_out[1] + 1))
        k_out = int(rng.integers(gc.fan_out[0], gc.fan_out[1] + 1))
        for src in choose_partners(g, cluster, k_in, gc, rng, exclude={uid}):
            connect(g, src, uid, gc, rng)
        for dst in choose_partners(g, cluster, k_out, gc, rng, exclude={uid}):
            connect(g, uid, dst, gc, rng)
        if not existing or rng.random() < 0.5:       # sometimes listen to the input directly
            connect_input(g, int(rng.integers(g.n_channels)), uid, gc, rng)


def remove_motifs(g: Genome, gc: GenomeConfig, mc: MutationConfig, rng) -> None:
    if g.n_motifs <= 1:
        return
    k = min(_how_many(g.n_motifs, mc.motif_fraction, rng), g.n_motifs - 1)
    g.remove_motifs(int(u) for u in rng.choice(list(g.motifs), size=k, replace=False))


def add_links(g: Genome, gc: GenomeConfig, mc: MutationConfig, rng) -> None:
    uids = list(g.motifs)
    if len(uids) < 2:
        return
    for _ in range(_how_many(len(g.links) or 1, mc.link_fraction, rng)):
        src = int(rng.choice(uids))
        for dst in choose_partners(g, g.motifs[src].cluster, 1, gc, rng, exclude={src}):
            connect(g, src, dst, gc, rng)


def remove_links(g: Genome, gc: GenomeConfig, mc: MutationConfig, rng) -> None:
    keys = list(g.links)
    if not keys:
        return
    k = min(_how_many(len(keys), mc.link_fraction, rng), len(keys))
    for i in rng.choice(len(keys), size=k, replace=False):
        del g.links[keys[i]]


def rewire_inputs(g: Genome, gc: GenomeConfig, mc: MutationConfig, rng) -> None:
    """Move some input projections to other motifs."""
    keys = list(g.input_links)
    if not keys:
        return
    uids = list(g.motifs)
    for i in rng.choice(len(keys), size=min(_how_many(len(keys), mc.link_fraction, rng), len(keys)),
                        replace=False):
        channel, _ = keys[i]
        del g.input_links[keys[i]]
        connect_input(g, channel, int(rng.choice(uids)), gc, rng)


# --------------------------------------------------------------- parameters
def neuron_params(g: Genome, gc: GenomeConfig, mc: MutationConfig, rng) -> None:
    """Perturb one neuron parameter (threshold, rest or refractory) of a fraction of neurons."""
    which = rng.integers(3)
    p = rng.uniform(*mc.param_fraction)
    for gene in g.motifs.values():
        hit = rng.random(len(gene.v_thresh)) < p
        if not hit.any():
            continue
        if which == 0:
            gene.v_thresh[hit] = np.clip(gene.v_thresh[hit] + rng.normal(0, mc.v_step, hit.sum()),
                                         *gc.v_thresh)
        elif which == 1:
            gene.v_rest[hit] = np.clip(gene.v_rest[hit] + rng.normal(0, mc.v_step, hit.sum()),
                                       *gc.v_rest)
        else:
            gene.refractory[hit] = np.clip(gene.refractory[hit] + rng.choice([-1, 1], hit.sum()),
                                           *gc.refractory)


def _synapse_arrays(g: Genome, attr: str):
    """All per-synapse arrays (internal, motif links, input links) of the genome."""
    yield from (getattr(m, attr) for m in g.motifs.values())
    yield from (getattr(l, attr) for l in g.links.values())
    yield from (getattr(l, attr) for l in g.input_links.values())


def delays(g: Genome, gc: GenomeConfig, mc: MutationConfig, rng) -> None:
    p = rng.uniform(*mc.param_fraction)
    for arr in _synapse_arrays(g, "delays"):
        hit = rng.random(arr.shape) < p
        arr[hit] = np.clip(arr[hit] + rng.choice([-1, 1], hit.sum()), *gc.delay)


def weights(g: Genome, gc: GenomeConfig, mc: MutationConfig, rng) -> None:
    if not gc.evolve_weights:
        return
    p = rng.uniform(*mc.param_fraction)
    for arr in _synapse_arrays(g, "weights"):
        hit = rng.random(arr.shape) < p
        arr[hit] = np.clip(arr[hit] + rng.normal(0, mc.weight_step, hit.sum()), *gc.weight)


# ------------------------------------------------------------------ driver
OPERATORS = {
    "add_motifs": add_motifs, "remove_motifs": remove_motifs,
    "add_links": add_links, "remove_links": remove_links, "rewire_inputs": rewire_inputs,
    "neuron_params": neuron_params, "delays": delays, "weights": weights,
}


def operator_probabilities(gc: GenomeConfig, mc: MutationConfig) -> dict[str, float]:
    p = {name: getattr(mc, f"p_{name}") for name in OPERATORS}
    if not gc.evolve_weights:
        p["weights"] = 0.0
    total = sum(p.values())
    return {k: v / total for k, v in p.items()}


def mutate(parent: Genome, gc: GenomeConfig, mc: MutationConfig,
           rng: np.random.Generator) -> tuple[Genome, list[str]]:
    """Return a mutated copy of `parent` and the names of the operators applied."""
    child = parent.copy()
    probs = operator_probabilities(gc, mc)
    names = list(rng.choice(list(probs), size=mc.n_operators, p=list(probs.values())))
    for name in names:
        OPERATORS[name](child, gc, mc, rng)
    child.prune_unreachable()
    return child, names
