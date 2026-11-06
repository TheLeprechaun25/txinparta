"""Baseline study: does evolution find networks that beat simple references?

Conditions (all scored on the *test* set with the same samples and weight draws):

  input        the raw input spike trains, no network
  random       random networks, as in generation 0
  evolved      best network (by search accuracy) of each evolution seed
  rewired      per evolved network: same motif types, numbers of links and input
               links, but random wiring and fresh parameters
  reparam      per evolved network: same topology, but neuron parameters and
               delays redrawn

Readouts: the search readout (nearest class medoid; prototypes from n_reference
training samples per class) and a ridge classifier trained on n_train
samples per class (the stronger reference).

    python experiments/baselines.py configs/temporal_patterns.toml --seeds 5 --workers 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from darwinsnn.analysis import (
    reparametrised,
    rewired_like,
    score_input,
    score_network,
    summarize,
)
from darwinsnn.config import load_config
from darwinsnn.data import load_dataset
from darwinsnn.evaluation import Evaluator
from darwinsnn.evolution import Evolution
from darwinsnn.genome import random_genome
from darwinsnn.simulators import make_simulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-random", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=40, help="ridge training samples per class")
    ap.add_argument("--n-query", type=int, default=20, help="test queries per class")
    ap.add_argument("--repetitions", type=int, default=5, help="weight draws for scoring")
    ap.add_argument("--out", default="results/baselines")
    ap.add_argument("--backend", default="", help="override simulator.backend (e.g. torch)")
    args = ap.parse_args()

    def load(path):
        c = load_config(path)
        if args.backend:
            c.simulator.backend = args.backend
        return c

    cfg = load(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)
    train, test = load_dataset(cfg.data, np.random.default_rng(cfg.evolution.seed))
    sim = make_simulator(cfg.simulator)

    # one scoring plan shared by every condition (common random numbers)
    score_cfg = load(args.config)
    score_cfg.fitness.n_repetitions = args.repetitions
    plan = Evaluator(score_cfg, sim).make_plan(train, rng, query_data=test, n_query=args.n_query)
    ridge_idx = train.split_by_class(rng, args.n_train)

    # evolution runs
    evolved = []
    for seed in range(args.seeds):
        run_cfg = load(args.config)
        run_cfg.evolution.seed = seed
        run_cfg.evolution.n_workers = args.workers
        run_cfg.evolution.out_dir = str(out / f"run_seed{seed}")
        print(f"--- evolution seed {seed}", flush=True)
        front = Evolution(run_cfg, train, test).run()
        evolved.append(max(front, key=lambda ind: ind.metrics["accuracy"]).genome)

    conditions = {
        "input": [None],
        "random": [random_genome(cfg.genome, train.n_channels, rng) for _ in range(args.n_random)],
        "evolved": evolved,
        "rewired": [rewired_like(g, cfg, rng) for g in evolved],
        "reparam": [reparametrised(g, cfg, rng) for g in evolved],
    }
    keys = ["accuracy", "ridge_accuracy", "separation", "spikes", "synops", "n_motifs"]
    results = {}
    for name, genomes in conditions.items():
        print(f"--- scoring {name} ({len(genomes)} networks)", flush=True)
        rows = [score_input(score_cfg, train, test, plan, ridge_idx, rng) if g is None else
                score_network(g, score_cfg, sim, train, test, plan, ridge_idx, rng)
                for g in genomes]
        results[name] = {"per_network": rows, "summary": summarize(rows, keys)}

    (out / "results.json").write_text(json.dumps(results, indent=2))
    chance = 1.0 / train.n_classes
    lines = [f"# Baselines: {cfg.data.name}, readout features = {cfg.fitness.features}",
             "", f"chance = {chance:.3f}; test queries/class = {args.n_query}; "
             f"weight draws = {args.repetitions}; evolution seeds = {args.seeds}", "",
             "| condition | n | medoid acc | ridge acc | separation | spikes/sample "
             "| synops/sample | motifs |",
             "|---|---|---|---|---|---|---|---|"]

    def cell(summary, key, fmt=".3f"):
        mean, std = summary[key]
        return f"{mean:{fmt}} ± {std:{fmt}}"

    for name, r in results.items():
        s = r["summary"]
        lines.append(f"| {name} | {len(r['per_network'])} | {cell(s, 'accuracy')} | "
                     f"{cell(s, 'ridge_accuracy')} | {cell(s, 'separation')} | "
                     f"{cell(s, 'spikes', '.0f')} | {cell(s, 'synops', '.0f')} | "
                     f"{cell(s, 'n_motifs', '.0f')} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
