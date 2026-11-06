"""Command line:

    darwinsnn run configs/temporal_patterns.toml [--set section.key=value ...]
    darwinsnn baseline configs/mnist.toml       # score the raw input (no network)
"""

from __future__ import annotations

import argparse
import ast

import numpy as np

from .config import load_config, validate
from .data import load_dataset
from .evolution import Evolution


def _apply_overrides(cfg, overrides: list[str]) -> None:
    for item in overrides:
        key, value = item.split("=", 1)
        section, name = key.split(".")
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass  # keep as a string
        target = getattr(cfg, section)
        if not hasattr(target, name):
            raise KeyError(f"unknown setting {key}")
        setattr(target, name, tuple(value) if isinstance(value, list) else value)
    validate(cfg)


def baseline(cfg) -> None:
    """Score the input spike trains themselves with the same readout.

    Any network worth keeping should beat this.
    """
    from .fitness import input_features, separation_and_accuracy
    rng = np.random.default_rng(cfg.evolution.seed)
    data, _ = load_dataset(cfg.data, rng)
    f = cfg.fitness
    scores = []
    for _ in range(max(f.n_repetitions, 10)):
        ref = data.split_by_class(rng, f.n_reference)
        query = data.split_by_class(rng, f.n_query, exclude=ref)
        scores.append(separation_and_accuracy(
            input_features(data, ref, f), data.labels[ref],
            input_features(data, query, f), data.labels[query], data.n_classes, f))
    sep, acc = np.mean(scores, axis=0)
    print(f"input baseline ({f.features}): separation={sep:.3f} accuracy={acc:.3f}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="darwinsnn")
    parser.add_argument("command", choices=["run", "baseline"])
    parser.add_argument("config")
    parser.add_argument("--set", nargs="*", default=[], metavar="SECTION.KEY=VALUE")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    _apply_overrides(cfg, args.set)
    if args.command == "run":
        Evolution(cfg).run()
    else:
        baseline(cfg)


if __name__ == "__main__":
    main()
