"""How noisy is one evaluation, and how big must the EvalPlan be?

For each plan size (reference samples/class, query samples/class, weight draws):
  * score every genome with two *independent* plans (other samples, other weights)
    -> rank agreement between the two (Spearman), i.e. could selection tell them apart?
  * compare with a large "truth" evaluation on the test set
    -> how well does a cheap search evaluation predict test performance?

Genomes: the final fronts of earlier runs (--fronts, searched recursively for
final_front.pkl) plus random genomes, so both good and bad networks are present.

    python experiments/noise.py configs/temporal_patterns.toml --fronts results/variants \
        --backend torch --out results/noise
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from darwinsnn.config import load_config
from darwinsnn.data import load_dataset
from darwinsnn.evaluation import Evaluator
from darwinsnn.genome import random_genome
from darwinsnn.simulators import make_simulator

PLAN_SIZES = [  # (reference/class, query/class, weight draws); the first is the old default
    (8, 8, 2), (8, 16, 2), (16, 16, 2), (16, 16, 4), (16, 32, 4), (24, 32, 4),
]
METRICS = ("accuracy", "separation", "synops")


def load_genomes(fronts: str | None, cfg, n_channels: int, n_random: int, rng) -> list:
    genomes = []
    if fronts:
        for path in sorted(Path(fronts).rglob("final_front.pkl")):
            with open(path, "rb") as f:
                genomes += pickle.load(f)["genomes"]
    genomes += [random_genome(cfg.genome, n_channels, rng) for _ in range(n_random)]
    return [g for g in genomes if g.n_motifs > 0]


def evaluate(cfg, sim, genomes, train, query, n_ref, n_query, n_reps, rng):
    cfg.fitness.n_reference, cfg.fitness.n_query, cfg.fitness.n_repetitions = n_ref, n_query, n_reps
    ev = Evaluator(cfg, sim)
    plan = ev.make_plan(train, rng, query_data=query)
    rows = ev.evaluate_population(genomes, train, plan, query)
    return {m: np.array([r[m] for r in rows]) for m in METRICS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--fronts", default=None)
    ap.add_argument("--n-random", type=int, default=64)
    ap.add_argument("--backend", default="")
    ap.add_argument("--truth", type=int, nargs=3, default=(32, 48, 8),
                    metavar=("REF", "QUERY", "REPS"), help="size of the test-set 'truth' plan")
    ap.add_argument("--out", default="results/noise")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.backend:
        cfg.simulator.backend = args.backend
    rng = np.random.default_rng(99)
    train, test = load_dataset(cfg.data, np.random.default_rng(cfg.evolution.seed))
    sim = make_simulator(cfg.simulator)
    genomes = load_genomes(args.fronts, cfg, train.n_channels, args.n_random, rng)
    print(f"{len(genomes)} genomes", flush=True)

    t0 = time.time()
    truth = evaluate(cfg, sim, genomes, train, test, *args.truth, rng)
    print(f"truth evaluated ({time.time() - t0:.0f}s)", flush=True)

    # the networks selection must tell apart: top quarter by test ("truth") accuracy
    top = truth["accuracy"] >= np.percentile(truth["accuracy"], 75)
    rows, per_genome = [], {"truth": {m: truth[m].tolist() for m in METRICS}}
    for n_ref, n_query, n_reps in PLAN_SIZES:
        t0 = time.time()
        a = evaluate(cfg, sim, genomes, train, None, n_ref, n_query, n_reps, rng)
        b = evaluate(cfg, sim, genomes, train, None, n_ref, n_query, n_reps, rng)
        row = {"n_reference": n_ref, "n_query": n_query, "n_repetitions": n_reps,
               "samples_per_individual": (n_ref + n_query) * train.n_classes * n_reps,
               "seconds": (time.time() - t0) / 2}
        for m in METRICS:
            row[f"{m}_repeat_spearman"] = float(spearmanr(a[m], b[m]).statistic)
            row[f"{m}_truth_spearman"] = float(spearmanr((a[m] + b[m]) / 2, truth[m]).statistic)
            row[f"{m}_abs_diff"] = float(np.mean(np.abs(a[m] - b[m])))
            row[f"{m}_repeat_spearman_top"] = float(spearmanr(a[m][top], b[m][top]).statistic)
            row[f"{m}_truth_spearman_top"] = float(
                spearmanr((a[m][top] + b[m][top]) / 2, truth[m][top]).statistic)
        rows.append(row)
        per_genome[f"{n_ref}_{n_query}_{n_reps}"] = {"a": {m: a[m].tolist() for m in METRICS},
                                                      "b": {m: b[m].tolist() for m in METRICS}}
        print(f"{n_ref:3d} ref {n_query:3d} query {n_reps} reps: "
              f"acc repeat rho={row['accuracy_repeat_spearman']:.3f} "
              f"truth rho={row['accuracy_truth_spearman']:.3f} "
              f"|diff|={row['accuracy_abs_diff']:.3f} "
              f"top-quarter repeat rho={row['accuracy_repeat_spearman_top']:.3f}  "
              f"sep repeat rho={row['separation_repeat_spearman']:.3f}  ({row['seconds']:.1f}s)",
              flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "noise.json").write_text(json.dumps({"n_genomes": len(genomes), "rows": rows,
                                                "per_genome": per_genome}))
    lines = [f"# Evaluation noise: {cfg.data.name}, {len(genomes)} genomes", "",
             "rho = Spearman rank correlation. *repeat*: two independent plans of the same size. "
             f"*truth*: their mean vs a test-set plan of {tuple(args.truth)}.", "",
             "*top*: only the top quarter of genomes by test accuracy, the ones selection "
             "must tell apart.", "",
             "| ref/class | query/class | weight draws | samples/individual | acc repeat rho "
             "| acc truth rho | acc repeat rho (top) | acc truth rho (top) | mean abs acc diff "
             "| sep repeat rho | sep truth rho | sep truth rho (top) | synops repeat rho |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['n_reference']} | {r['n_query']} | {r['n_repetitions']} | "
            f"{r['samples_per_individual']} | {r['accuracy_repeat_spearman']:.3f} | "
            f"{r['accuracy_truth_spearman']:.3f} | {r['accuracy_repeat_spearman_top']:.3f} | "
            f"{r['accuracy_truth_spearman_top']:.3f} | {r['accuracy_abs_diff']:.3f} | "
            f"{r['separation_repeat_spearman']:.3f} | {r['separation_truth_spearman']:.3f} | "
            f"{r['separation_truth_spearman_top']:.3f} | {r['synops_repeat_spearman']:.3f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
