"""Choose the weight range: where do random networks keep the input's information?

Too strong weights saturate the network (most neurons fire every step, so all
samples look alike). Too weak weights silence it. For each candidate weight
range this scores random genomes with one shared plan and reports accuracy,
separation, firing rate, and the spread between networks (if every network
scores the same, selection has nothing to work with).

    python experiments/calibrate.py configs/shd.toml --backend torch --out results/calibrate_shd
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from darwinsnn.config import load_config
from darwinsnn.data import load_dataset
from darwinsnn.evaluation import Evaluator
from darwinsnn.genome import random_genome
from darwinsnn.simulators import make_simulator

WEIGHT_RANGES = [(0.5, 1.5), (1.0, 3.0), (2.0, 5.0), (3.0, 8.0), (4.0, 12.0), (8.0, 16.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--backend", default="")
    ap.add_argument("--n-genomes", type=int, default=24)
    ap.add_argument("--plan", type=int, nargs=3, default=(16, 16, 2),
                    metavar=("REF", "QUERY", "REPS"))
    ap.add_argument("--out", default="results/calibrate")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.backend:
        cfg.simulator.backend = args.backend
    cfg.fitness.n_reference, cfg.fitness.n_query, cfg.fitness.n_repetitions = args.plan
    train, test = load_dataset(cfg.data, np.random.default_rng(cfg.evolution.seed))
    sim = make_simulator(cfg.simulator)
    rng = np.random.default_rng(5)
    plan = Evaluator(cfg, sim).make_plan(train, rng, query_data=test)
    # the same topologies for every weight range: only the weight draw changes
    genomes = [random_genome(cfg.genome, train.n_channels, np.random.default_rng(100 + i))
               for i in range(args.n_genomes)]
    T = cfg.simulator.time_steps

    rows = []
    for lo, hi in WEIGHT_RANGES:
        cfg.genome.weight = (lo, hi)
        res = Evaluator(cfg, sim).evaluate_population(genomes, train, plan, test)
        acc = np.array([r["accuracy"] for r in res])
        sep = np.array([r["separation"] for r in res])
        rate = np.array([r["spikes"] / (g.n_neurons * T) for r, g in zip(res, genomes)])
        row = {"weight": [lo, hi], "acc_mean": acc.mean(), "acc_std": acc.std(),
               "acc_max": acc.max(), "sep_mean": sep.mean(), "sep_std": sep.std(),
               "rate_mean": rate.mean(), "rate_min": rate.min(), "rate_max": rate.max(),
               "synops_mean": float(np.mean([r["synops"] for r in res]))}
        rows.append({k: (float(v) if not isinstance(v, list) else v) for k, v in row.items()})
        print(f"w=[{lo}, {hi}]: acc {acc.mean():.3f} ± {acc.std():.3f} (max {acc.max():.3f})  "
              f"sep {sep.mean():.3f} ± {sep.std():.3f}  "
              f"rate {rate.mean():.4f} spikes/neuron/step", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibrate.json").write_text(json.dumps(rows, indent=2))
    lines = [f"# Weight calibration: {cfg.data.name}, {args.n_genomes} random genomes, "
             f"plan {tuple(args.plan)} (queries from the test set)", "",
             "| weight range | accuracy (mean ± std, max) | separation | firing rate "
             "(spikes/neuron/step) | synops/sample |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['weight']} | {r['acc_mean']:.3f} ± {r['acc_std']:.3f} "
                     f"({r['acc_max']:.3f}) | {r['sep_mean']:.3f} ± {r['sep_std']:.3f} | "
                     f"{r['rate_mean']:.4f} [{r['rate_min']:.4f}, {r['rate_max']:.4f}] | "
                     f"{r['synops_mean']:.0f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
