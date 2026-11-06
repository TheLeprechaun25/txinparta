"""The neuroevolution loop (NSGA-II, mutation only).

    population = random genomes
    for each generation:
        plan      = fresh samples + weight seeds          (shared by everybody)
        parents   -> re-evaluated with `plan`
        offspring = [crossover of two tournament-selected parents, with prob. p_crossover]
                    + mutation                           -> evaluated with `plan`
        population = best pop_size of parents + offspring (rank, then crowding)
    final: Pareto front evaluated on the held-out test set

Output (in evolution.out_dir):
    config.json         the full configuration
    provenance.json     code versions (git commits), seed, machine
    log.jsonl           one line per evaluated individual and generation
    final_front.pkl     genomes of the final Pareto front + their test metrics
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .crossover import CROSSOVERS
from .data import SpikeDataset, load_dataset
from .evaluation import Evaluator
from .fitness import DIRECTION
from .genome import Genome, random_genome
from .mutation import mutate
from .nsga2 import rank_and_crowding, select_survivors, tournament
from .parallel import PopulationEvaluator
from .provenance import provenance


@dataclass
class Individual:
    genome: Genome
    metrics: dict[str, float] | None = None
    origin: str = "init"            # "init" or the operators that created it
    crossover: dict | None = None   # statistics of the crossover that created it, if any


def objective_matrix(pop: list[Individual], objectives: list[str]) -> np.ndarray:
    """Metrics as an NSGA-II matrix where every column is minimised."""
    return np.array([[DIRECTION[o] * ind.metrics[o] for o in objectives] for ind in pop])


def describe(ind: Individual) -> dict:
    g = ind.genome
    return {"n_motifs": g.n_motifs, "n_neurons": g.n_neurons, "n_synapses": g.n_synapses,
            "n_links": len(g.links), "motif_counts": g.motif_counts().tolist(),
            "n_clusters": len(g.clusters()), "n_inter_cluster_links": g.n_inter_cluster_links(),
            "origin": ind.origin, "crossover": ind.crossover, **ind.metrics}


class Evolution:
    def __init__(self, cfg: ExperimentConfig, data: SpikeDataset | None = None,
                 test: SpikeDataset | None = None, verbose: bool = True):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.evolution.seed)
        if data is None:
            data, test = load_dataset(cfg.data, self.rng)
        self.data, self.test = data, test
        self.evaluator = Evaluator(cfg, simulator=None)  # only used to draw EvalPlans
        self.out = Path(cfg.evolution.out_dir)
        self.verbose = verbose

    def _evaluate(self, pop: list[Individual], plan, on_test: bool = False) -> list[dict]:
        metrics = self.workers.evaluate([ind.genome for ind in pop], plan, on_test)
        if not on_test:
            for ind, m in zip(pop, metrics):
                ind.metrics = m
        return metrics

    def _log(self, gen: int, pop: list[Individual], role: str) -> None:
        with open(self.out / "log.jsonl", "a") as f:
            for i, ind in enumerate(pop):
                f.write(json.dumps({"generation": gen, "role": role, "index": i,
                                    **describe(ind)}) + "\n")

    def _report(self, gen: int, pop: list[Individual], seconds: float) -> None:
        if not self.verbose:
            return
        def best(o):  # max for maximised objectives, min for minimised ones
            values = [ind.metrics[o] for ind in pop]
            return max(values) if DIRECTION[o] < 0 else min(values)
        cols = " ".join(f"{o}={best(o):.3g}" for o in self.cfg.fitness.objectives)
        sizes = [ind.genome.n_motifs for ind in pop]
        print(f"gen {gen:3d}  best: {cols}  motifs: {min(sizes)}-{max(sizes)}  ({seconds:.1f}s)",
              flush=True)

    def run(self) -> list[Individual]:
        cfg = self.cfg
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
        (self.out / "provenance.json").write_text(
            json.dumps(provenance(cfg.evolution.seed, cfg.simulator.backend), indent=2))
        (self.out / "log.jsonl").write_text("")
        self.workers = PopulationEvaluator(cfg, self.data, self.test, cfg.evolution.n_workers)
        try:
            return self._loop()
        finally:
            self.workers.close()

    def _loop(self) -> list[Individual]:
        cfg, rng = self.cfg, self.rng
        objectives, n = cfg.fitness.objectives, cfg.evolution.pop_size

        pop = [Individual(random_genome(cfg.genome, self.data.n_channels, rng)) for _ in range(n)]
        t0 = time.time()
        self._evaluate(pop, self.evaluator.make_plan(self.data, rng))
        self._log(0, pop, "population")
        self._report(0, pop, time.time() - t0)

        for gen in range(1, cfg.evolution.generations + 1):
            t0 = time.time()
            plan = self.evaluator.make_plan(self.data, rng)
            rank, crowd = rank_and_crowding(objective_matrix(pop, objectives))
            offspring = []
            mates = tournament(rank, crowd, n, rng)
            for p, q in zip(tournament(rank, crowd, n, rng), mates):
                base, stats, prefix = pop[p].genome, None, ""
                if rng.random() < cfg.evolution.p_crossover:
                    base, stats = CROSSOVERS[cfg.evolution.crossover](
                        pop[p].genome, pop[q].genome, cfg.genome, rng)
                    prefix = f"x_{cfg.evolution.crossover}+"
                child, ops = mutate(base, cfg.genome, cfg.mutation, rng)
                offspring.append(Individual(child, origin=prefix + "+".join(ops), crossover=stats))
            # parents and offspring in one batch, so the workers stay busy
            self._evaluate((pop if cfg.evolution.reevaluate_parents else []) + offspring, plan)
            self._log(gen, offspring, "offspring")

            merged = pop + offspring
            pop = [merged[i] for i in select_survivors(objective_matrix(merged, objectives), n)]
            self._log(gen, pop, "population")
            self._report(gen, pop, time.time() - t0)

        return self.finish(pop)

    def finish(self, pop: list[Individual]) -> list[Individual]:
        """Evaluate the final Pareto front on the test set and save it."""
        objectives = self.cfg.fitness.objectives
        rank, _ = rank_and_crowding(objective_matrix(pop, objectives))
        front = [ind for ind, r in zip(pop, rank) if r == 0]
        results = []
        if self.test is not None:
            plan = self.evaluator.make_plan(self.data, self.rng, query_data=self.test)
            for ind, test_metrics in zip(front, self._evaluate(front, plan, on_test=True)):
                results.append({"search": ind.metrics, "test": test_metrics})
                if self.verbose:
                    print("front:", {k: round(v, 4) for k, v in test_metrics.items()},
                          f"motifs={ind.genome.n_motifs}")
        with open(self.out / "final_front.pkl", "wb") as f:
            pickle.dump({"genomes": [ind.genome for ind in front], "metrics": results}, f)
        return front
