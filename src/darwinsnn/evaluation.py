"""Evaluate genomes: simulate them and compute their objectives.

Noise control. An evaluation is noisy, because it depends on which samples
are drawn and, when weights are not evolved, on the weight draw. So every
generation draws one EvalPlan (samples + weight seeds) and evaluates *all*
individuals, parents and offspring alike, with it (common random numbers).
Individuals are then compared on equal terms, and a parent cannot survive
only because of an old lucky evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig
from .data import SpikeDataset
from .fitness import energy, features, separation_and_accuracy
from .genome import Genome, build_network
from .simulators import Simulator

METRICS = ("separation", "accuracy", "spikes", "synops")


@dataclass
class Repetition:
    ref_idx: np.ndarray             # samples used to build the prototypes
    query_idx: np.ndarray           # held-out samples used to score
    weight_seed: int                # seed of the weight draw (weight-agnostic search)


@dataclass
class EvalPlan:
    repetitions: list[Repetition]


class Evaluator:
    def __init__(self, cfg: ExperimentConfig, simulator: Simulator):
        self.cfg = cfg
        self.sim = simulator

    # ------------------------------------------------------------------ plans
    def make_plan(self, data: SpikeDataset, rng: np.random.Generator,
                  query_data: SpikeDataset | None = None, n_query: int | None = None) -> EvalPlan:
        """Draw reference and query samples (and weight seeds) for every repetition.

        By default the queries are other samples of `data`. Pass `query_data`
        (e.g. the test set) to draw the queries from it instead.
        """
        f = self.cfg.fitness
        n_query = n_query or f.n_query
        reps = []
        for _ in range(f.n_repetitions):
            ref = data.split_by_class(rng, f.n_reference)
            if query_data is None:
                query = data.split_by_class(rng, n_query, exclude=ref)
            else:
                query = query_data.split_by_class(rng, n_query)
            reps.append(Repetition(ref, query, int(rng.integers(2**31))))
        return EvalPlan(reps)

    # ------------------------------------------------------------- evaluation
    def weights_for(self, net, seed: int) -> np.ndarray:
        """Signed weights: the evolved ones, or a fresh random draw."""
        g = self.cfg.genome
        if g.evolve_weights:
            return net.signed_weights()
        magnitudes = np.random.default_rng(seed).uniform(*g.weight, size=net.n_synapses)
        return net.signed_weights(magnitudes.astype(np.float32))

    def evaluate(self, genome: Genome, data: SpikeDataset, plan: EvalPlan,
                 query_data: SpikeDataset | None = None) -> dict[str, float]:
        """Mean of every metric over the plan's repetitions.

        If `query_data` is given (e.g. the test set), `query_idx` refers to it;
        otherwise both index sets refer to `data`.
        """
        return self.evaluate_population([genome], data, plan, query_data)[0]

    def evaluate_population(self, genomes: list[Genome], data: SpikeDataset, plan: EvalPlan,
                            query_data: SpikeDataset | None = None) -> list[dict[str, float]]:
        """`evaluate` for many genomes at once.

        Backends with a `run_many` method (TorchSimulator) simulate all the
        networks of one repetition in a single pass; the others loop.
        """
        query_data = query_data or data
        f, T = self.cfg.fitness, self.cfg.simulator.time_steps
        bin_size = f.bin_size if f.features == "binned" else None
        alive = [i for i, g in enumerate(genomes) if g.n_motifs > 0]
        nets = {i: build_network(genomes[i]) for i in alive}
        per_rep = {i: [] for i in alive}

        def simulate(ws, dataset, idx):
            if hasattr(self.sim, "run_many"):
                return self.sim.run_many([nets[i] for i in alive], ws, dataset, idx, T, bin_size)
            return [self.sim.run(nets[i], w, dataset, idx, T, bin_size) for i, w in zip(alive, ws)]

        for rep in plan.repetitions:
            ws = [self.weights_for(nets[i], rep.weight_seed) for i in alive]
            refs = simulate(ws, data, rep.ref_idx)
            queries = simulate(ws, query_data, rep.query_idx)
            for i, r_ref, r_q in zip(alive, refs, queries):
                net = nets[i]
                sep, acc = separation_and_accuracy(
                    features(r_ref, net, f), data.labels[rep.ref_idx],
                    features(r_q, net, f), query_data.labels[rep.query_idx],
                    data.n_classes, f)
                spikes, synops = energy(r_q, net)
                per_rep[i].append((sep, acc, spikes, synops))

        empty = {"separation": 0.0, "accuracy": 0.0, "spikes": 0.0, "synops": 0.0}
        return [dict(zip(METRICS, map(float, np.mean(per_rep[i], axis=0)))) if i in per_rep
                else dict(empty) for i in range(len(genomes))]
