"""Shared analysis helpers: control networks and test-set scoring.

Used by the experiment scripts in experiments/.
"""

from __future__ import annotations

import numpy as np

from .evaluation import Evaluator
from .fitness import features, input_features, separation_and_accuracy
from .genome import (
    Genome,
    build_network,
    connect,
    connect_input,
    new_link,
    new_motif_gene,
)
from .readouts import ridge_accuracy


# ------------------------------------------------------------ control networks
def rewired_like(g: Genome, cfg, rng) -> Genome:
    """Same motif types and link counts as `g`, random wiring and parameters."""
    out = Genome(n_channels=g.n_channels)
    uids = [out.add_motif(new_motif_gene(m.kind, cfg.genome, rng)) for m in g.motifs.values()]
    pairs = set()
    while len(pairs) < min(len(g.links), len(uids) * (len(uids) - 1)):
        s, d = rng.choice(uids, size=2, replace=False)
        pairs.add((int(s), int(d)))
    for s, d in pairs:
        connect(out, s, d, cfg.genome, rng)
    inputs = set()
    while len(inputs) < min(len(g.input_links), g.n_channels * len(uids)):
        inputs.add((int(rng.integers(g.n_channels)), int(rng.choice(uids))))
    for ch, d in inputs:
        connect_input(out, ch, d, cfg.genome, rng)
    out.prune_unreachable()
    return out


def reparametrised(g: Genome, cfg, rng) -> Genome:
    """Same topology as `g`, all neuron parameters and delays (and weights) redrawn."""
    out = g.copy()
    for uid, m in out.motifs.items():
        out.motifs[uid] = new_motif_gene(m.kind, cfg.genome, rng)
    for links in (out.links, out.input_links):
        for key, link in links.items():
            m, a, b = link.delays.shape
            links[key] = new_link(a, b, m, cfg.genome, rng)
    return out


# ------------------------------------------------------------------- scoring
def score_network(genome, cfg, sim, train, test, plan, ridge_idx, rng) -> dict:
    """Medoid readout (as in the search) and ridge readout, averaged over weight draws."""
    ev = Evaluator(cfg, sim)
    metrics = ev.evaluate(genome, train, plan, test)
    net = build_network(genome)
    bin_size = cfg.fitness.bin_size if cfg.fitness.features == "binned" else None
    T = cfg.simulator.time_steps
    ridge = []
    for rep in plan.repetitions:
        w = ev.weights_for(net, rep.weight_seed)
        X_tr = features(sim.run(net, w, train, ridge_idx, T, bin_size), net, cfg.fitness)
        X_te = features(sim.run(net, w, test, np.arange(len(test)), T, bin_size), net, cfg.fitness)
        ridge.append(ridge_accuracy(X_tr, train.labels[ridge_idx], X_te, test.labels,
                                    train.n_classes, rng=rng))
    return {**metrics, "ridge_accuracy": float(np.mean(ridge)),
            "n_motifs": genome.n_motifs, "n_neurons": genome.n_neurons}


def score_input(cfg, train, test, plan, ridge_idx, rng) -> dict:
    f = cfg.fitness
    sep_acc = [separation_and_accuracy(
        input_features(train, r.ref_idx, f), train.labels[r.ref_idx],
        input_features(test, r.query_idx, f), test.labels[r.query_idx], train.n_classes, f)
        for r in plan.repetitions]
    ridge = ridge_accuracy(input_features(train, ridge_idx, f), train.labels[ridge_idx],
                           input_features(test, np.arange(len(test)), f), test.labels,
                           train.n_classes, rng=rng)
    sep, acc = np.mean(sep_acc, axis=0)
    return {"separation": sep, "accuracy": acc, "ridge_accuracy": ridge,
            "spikes": 0.0, "synops": 0.0, "n_motifs": 0, "n_neurons": 0}


# ---------------------------------------------------------------------- main
def summarize(rows: list[dict], keys) -> dict:
    return {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])))
            for k in keys}




def hypervolume_2d(points, ref: tuple[float, float]) -> float:
    """Area dominated by `points` (both columns minimised), bounded by the reference point."""
    pts = sorted((float(x), float(y)) for x, y in points if x < ref[0] and y < ref[1])
    front, best_y = [], ref[1]
    for x, y in pts:                       # keep the non-dominated staircase
        if y < best_y:
            front.append((x, y))
            best_y = y
    area = 0.0
    for i, (x, y) in enumerate(front):
        x_next = front[i + 1][0] if i + 1 < len(front) else ref[0]
        area += (x_next - x) * (ref[1] - y)
    return area
