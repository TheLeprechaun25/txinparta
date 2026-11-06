"""Compare search variants with the same budget and the same test scoring.

  mutation_only        unstructured genomes, mutation only (reference)
  crossover            unstructured genomes, region crossover + mutation
  clustered            clustered genomes, mutation only   (control: structure alone)
  clustered_crossover  clustered genomes, cluster crossover + mutation
  clustered_crossover_low   same with p_crossover = 0.2 instead of 0.5

For every variant and seed this script saves:
  * the best network (by *search* accuracy) of the final front, scored on the
    test set with the medoid and the ridge readout (shared plan, as in baselines.py)
  * the hypervolume of the whole final front on the test set, in
    (1 - accuracy, log10(1 + synops)), with reference point (1 - chance, 5)
  * search curves per generation and crossover statistics

    python experiments/variants.py configs/temporal_patterns.toml --seeds 5 --workers 8
    python experiments/variants.py CONFIG --backend torch --variants crossover   # one variant
    python experiments/variants.py CONFIG --summarize                           # merge results

Each (variant, seed) writes OUT/<variant>/seed<k>/result.json, so variants and
seeds can run as separate (SLURM) jobs (--seed-list); --summarize merges whatever
is there into OUT/summary.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from darwinsnn.analysis import hypervolume_2d, score_network
from darwinsnn.config import load_config, validate
from darwinsnn.data import load_dataset
from darwinsnn.evaluation import Evaluator
from darwinsnn.evolution import Evolution
from darwinsnn.simulators import make_simulator

# Every variant sets both the structure and the crossover explicitly, so the
# result does not depend on the defaults of the config file.
FLAT, CLUSTERED = {"genome.n_clusters": (1, 1)}, {"genome.n_clusters": (4, 8)}
NO_X = {"evolution.p_crossover": 0.0}
VARIANTS = {
    "mutation_only": {**FLAT, **NO_X},
    "crossover": {**FLAT, "evolution.p_crossover": 0.5, "evolution.crossover": "region"},
    "clustered": {**CLUSTERED, **NO_X},
    "clustered_crossover": {**CLUSTERED, "evolution.p_crossover": 0.5,
                            "evolution.crossover": "cluster"},
    "clustered_crossover_low": {**CLUSTERED, "evolution.p_crossover": 0.2,
                                "evolution.crossover": "cluster"},
}


def make_config(path, overrides: dict, seed: int, args, out_dir: Path):
    cfg = load_config(path)
    for key, value in overrides.items():
        section, name = key.split(".")
        setattr(getattr(cfg, section), name, value)
    cfg.evolution.seed = seed
    cfg.evolution.pop_size = args.pop_size
    cfg.evolution.generations = args.generations
    cfg.evolution.n_workers = args.workers
    cfg.evolution.out_dir = str(out_dir)
    if args.backend:
        cfg.simulator.backend = args.backend
    if args.search_plan:
        f = cfg.fitness
        f.n_reference, f.n_query, f.n_repetitions = args.search_plan
    validate(cfg)
    return cfg


def read_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def curves(log: list[dict]) -> dict[str, list[float]]:
    """Per generation: best search accuracy and lowest synops in the population."""
    gens = sorted({r["generation"] for r in log})
    pop = [[r for r in log if r["generation"] == g and r["role"] == "population"] for g in gens]
    return {"best_accuracy": [max(r["accuracy"] for r in p) for p in pop],
            "median_accuracy": [float(np.median([r["accuracy"] for r in p])) for p in pop],
            "min_synops": [min(r["synops"] for r in p) for p in pop],
            "n_clusters": [float(np.mean([r["n_clusters"] for r in p])) for p in pop],
            "inter_cluster_fraction": [
                float(np.mean([r["n_inter_cluster_links"] / max(r["n_links"], 1) for r in p]))
                for p in pop]}


def crossover_stats(log: list[dict]) -> dict:
    """Disruption of crossovers, and how often each kind of offspring survives selection."""
    def key(r):
        return (r["generation"], r["n_motifs"], r["n_synapses"], r["accuracy"], r["synops"])
    survivors = {key(r) for r in log if r["role"] == "population"}
    off = [r for r in log if r["role"] == "offspring"]
    x = [r for r in off if r["crossover"] is not None]
    m = [r for r in off if r["crossover"] is None]

    def rate(rows):
        return float(np.mean([key(r) in survivors for r in rows])) if rows else None

    return {"n_crossover_offspring": len(x), "n_mutation_offspring": len(m),
            "cut_fraction": (float(np.mean([r["crossover"]["cut_fraction"] for r in x]))
                             if x else None),
            "survival_crossover": rate(x), "survival_mutation": rate(m)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-list", type=int, nargs="*", default=None,
                    help="run exactly these seeds (overrides --seeds)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--pop-size", type=int, default=32)
    ap.add_argument("--generations", type=int, default=50)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-query", type=int, default=20)
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--out", default="results/variants")
    ap.add_argument("--backend", default="", help="override simulator.backend (e.g. torch)")
    ap.add_argument("--search-plan", type=int, nargs=3, default=None,
                    metavar=("REF", "QUERY", "REPS"), help="override the search EvalPlan size")
    ap.add_argument("--summarize", action="store_true", help="only merge existing results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        write_summary(load_results(out), out, args)
        return
    base = load_config(args.config)
    if args.backend:
        base.simulator.backend = args.backend
    train, test = load_dataset(base.data, np.random.default_rng(base.evolution.seed))
    sim = make_simulator(base.simulator)
    rng = np.random.default_rng(1234)
    base.fitness.n_repetitions = args.repetitions
    plan = Evaluator(base, sim).make_plan(train, rng, query_data=test, n_query=args.n_query)
    ridge_idx = train.split_by_class(rng, args.n_train)
    ref = (1.0 - 1.0 / train.n_classes, 5.0)

    seeds = args.seed_list if args.seed_list is not None else range(args.seeds)
    for name in args.variants:
        for seed in seeds:
            run_dir = out / name / f"seed{seed}"
            cfg = make_config(args.config, VARIANTS[name], seed, args, run_dir)
            t0 = time.time()
            front = Evolution(cfg, train, test, verbose=False).run()
            best = max(front, key=lambda ind: ind.metrics["accuracy"])
            scored = score_network(best.genome, cfg, sim, train, test, plan, ridge_idx, rng)
            front_test = Evaluator(cfg, sim).evaluate_population(
                [ind.genome for ind in front], train, plan, test)
            hv = hypervolume_2d(
                [(1 - m["accuracy"], np.log10(1 + m["synops"])) for m in front_test], ref)
            log = read_log(run_dir / "log.jsonl")
            row = {"seed": seed, **scored, "hypervolume": hv, "front_size": len(front),
                   "minutes": (time.time() - t0) / 60, "curves": curves(log),
                   "crossover_stats": crossover_stats(log)}
            (run_dir / "result.json").write_text(json.dumps(row, indent=2))
            print(f"{name:20s} seed {seed}: acc={scored['accuracy']:.3f} "
                  f"ridge={scored['ridge_accuracy']:.3f} hv={hv:.3f} "
                  f"synops={scored['synops']:.0f} ({row['minutes']:.1f} min)", flush=True)

    write_summary(load_results(out), out, args)


def load_results(out: Path) -> dict[str, list[dict]]:
    """Results of every variant found in OUT, in the order of VARIANTS."""
    found = {}
    for name in VARIANTS:
        per_seed = sorted((out / name).glob("seed*/result.json"),
                          key=lambda p: int(p.parent.name[4:]))
        if per_seed:
            found[name] = [json.loads(p.read_text()) for p in per_seed]
        elif (out / name / "results.json").exists():      # older layout: one file per variant
            found[name] = json.loads((out / name / "results.json").read_text())
    return found


def write_summary(results: dict, out: Path, args) -> None:
    keys = [("accuracy", ".3f"), ("ridge_accuracy", ".3f"), ("hypervolume", ".3f"),
            ("synops", ".0f"), ("n_motifs", ".0f")]
    if not results:
        print(f"no results found in {out}")
        return
    n_seeds = max(len(rows) for rows in results.values())
    lines = [f"# Search variants ({n_seeds} seeds, pop {args.pop_size}, "
             f"{args.generations} generations)", "",
             "Best-of-front network scored on the test set (medoid and ridge readouts); "
             "hypervolume of the whole front on the test set.", "",
             "| variant | medoid acc | ridge acc | hypervolume | synops | motifs | cut fraction "
             "| survival x / mut |", "|---|---|---|---|---|---|---|---|"]
    for name, rows in results.items():
        cells = [f"{np.mean([r[k] for r in rows]):{f}} ± {np.std([r[k] for r in rows]):{f}}"
                 for k, f in keys]
        xs = [r["crossover_stats"] for r in rows]
        cut = [x["cut_fraction"] for x in xs if x["cut_fraction"] is not None]
        sx = [x["survival_crossover"] for x in xs if x["survival_crossover"] is not None]
        sm = [x["survival_mutation"] for x in xs if x["survival_mutation"] is not None]
        cut_cell = f"{np.mean(cut):.3f}" if cut else "–"
        survival = (f"{np.mean(sx):.2f}" if sx else "–") + f" / {np.mean(sm):.2f}"
        lines.append("| " + " | ".join([name, *cells, cut_cell, survival]) + " |")

    ref_name = "mutation_only"
    if ref_name in results:
        lines += ["", f"One-sided Mann-Whitney p-values, variant > {ref_name}:", "",
                  "| variant | medoid acc | ridge acc | hypervolume |", "|---|---|---|---|"]
        for name, rows in results.items():
            if name == ref_name:
                continue
            ps = [mannwhitneyu([r[k] for r in rows], [r[k] for r in results[ref_name]],
                               alternative="greater").pvalue
                  for k in ("accuracy", "ridge_accuracy", "hypervolume")]
            lines.append(f"| {name} | " + " | ".join(f"{p:.3f}" for p in ps) + " |")

    lines += ["", "Mean best search accuracy in the population, by generation:", "",
              "| variant | " + " | ".join(f"gen {g}" for g in (0, 10, 25, 50)) + " |",
              "|---|---|---|---|---|"]
    for name, rows in results.items():
        curve = np.mean([r["curves"]["best_accuracy"] for r in rows], axis=0)
        gens = [g for g in (0, 10, 25, 50) if g < len(curve)]
        lines.append(f"| {name} | " + " | ".join(f"{curve[g]:.3f}" for g in gens) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
