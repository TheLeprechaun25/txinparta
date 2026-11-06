"""Evaluate a whole population in parallel, one process per worker.

Individuals are independent, so this is the cheapest big speed-up: about
n_workers times, against at most 2x from Pultsito's own threads. Every worker
builds its own simulator (a ctypes library cannot be pickled) and receives
the datasets once, at start-up. Per task only the genome and the EvalPlan are
sent.

Results do not depend on the number of workers: every random draw comes from
the EvalPlan, which is made in the main process.

When using many workers with the Pultsito backend, set simulator.n_threads = 1
so the machine is not oversubscribed. The torch (GPU) backend ignores n_workers:
it evaluates the whole population in one process and one pass per repetition.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

from .config import ExperimentConfig
from .data import SpikeDataset
from .evaluation import EvalPlan, Evaluator
from .genome import Genome
from .simulators import make_simulator

_worker: dict = {}


def _init_worker(cfg: ExperimentConfig, data: SpikeDataset, test: SpikeDataset | None) -> None:
    _worker["evaluator"] = Evaluator(cfg, make_simulator(cfg.simulator))
    _worker["data"], _worker["test"] = data, test


def _evaluate(task: tuple[Genome, EvalPlan, bool]) -> dict[str, float]:
    genome, plan, on_test = task
    query = _worker["test"] if on_test else None
    return _worker["evaluator"].evaluate(genome, _worker["data"], plan, query)


class PopulationEvaluator:
    """`evaluate(genomes, plan)` -> list of metric dicts, serial or parallel."""

    def __init__(self, cfg: ExperimentConfig, data: SpikeDataset, test: SpikeDataset | None,
                 n_workers: int = 1):
        self.n_workers = n_workers
        self.pool = None
        self.batched = cfg.simulator.backend == "torch"
        if n_workers > 1 and not self.batched:
            self.pool = ProcessPoolExecutor(n_workers, initializer=_init_worker,
                                            initargs=(cfg, data, test))
        else:
            _init_worker(cfg, data, test)

    def evaluate(self, genomes: list[Genome], plan: EvalPlan,
                 on_test: bool = False) -> list[dict[str, float]]:
        if self.batched:          # GPU backend: the whole population in one pass
            ev = _worker["evaluator"]
            query = _worker["test"] if on_test else None
            return ev.evaluate_population(genomes, _worker["data"], plan, query)
        tasks = [(g, plan, on_test) for g in genomes]
        if self.pool is None:
            return [_evaluate(t) for t in tasks]
        return list(self.pool.map(_evaluate, tasks))

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()
