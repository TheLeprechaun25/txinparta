import json

import numpy as np

from darwinsnn.config import ExperimentConfig, FitnessConfig
from darwinsnn.data import make_temporal_patterns
from darwinsnn.evolution import Evolution
from darwinsnn.fitness import separation_and_accuracy
from darwinsnn.nsga2 import crowding_distance, non_dominated_fronts, select_survivors


# ------------------------------------------------------------------ fitness
def clusters(n_classes, n_per_class, spread, rng):
    centers = rng.normal(0, 10, (n_classes, 5))
    y = np.repeat(np.arange(n_classes), n_per_class)
    return centers[y] + rng.normal(0, spread, (len(y), 5)), y


def test_well_separated_classes_score_high():
    rng = np.random.default_rng(0)
    Xr, yr = clusters(4, 10, 0.1, rng)
    Xq, yq = Xr + rng.normal(0, 0.1, Xr.shape), yr
    sep, acc = separation_and_accuracy(Xr, yr, Xq, yq, 4, FitnessConfig())
    assert acc == 1.0 and sep > 0.9


def test_identical_patterns_score_zero():
    X, y = np.ones((40, 5)), np.repeat(np.arange(4), 10)
    sep, acc = separation_and_accuracy(X, y, X, y, 4, FitnessConfig())
    assert sep == 0.0 and acc == 0.25          # every query goes to prototype 0


# ------------------------------------------------------------------ NSGA-II
def test_fronts_and_survivors():
    F = np.array([[1, 5], [2, 2], [5, 1], [3, 3], [4, 4], [6, 6]], float)
    fronts = non_dominated_fronts(F)
    assert [sorted(f.tolist()) for f in fronts] == [[0, 1, 2], [3], [4], [5]]
    assert sorted(select_survivors(F, 4).tolist()) == [0, 1, 2, 3]
    crowd = crowding_distance(F[[0, 1, 2]])
    assert np.isinf(crowd[0]) and np.isinf(crowd[2]) and np.isfinite(crowd[1])


# ---------------------------------------------------------------- evolution
def test_evolution_smoke(tmp_path):
    cfg = ExperimentConfig()
    cfg.genome.n_motifs = (5, 15)
    cfg.simulator.time_steps = 70
    cfg.fitness.features = "counts"
    cfg.fitness.n_reference = cfg.fitness.n_query = 3
    cfg.fitness.n_repetitions = 1
    cfg.evolution.pop_size, cfg.evolution.generations = 4, 2
    cfg.evolution.out_dir = str(tmp_path)
    rng = np.random.default_rng(0)
    data = make_temporal_patterns(3, 10, 60, 8, 2.0, 0.0, rng)
    test = make_temporal_patterns(3, 10, 60, 4, 2.0, 0.0, rng)
    front = Evolution(cfg, data, test, verbose=False).run()
    assert front
    lines = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert {l["generation"] for l in lines} == {0, 1, 2}
    assert (tmp_path / "final_front.pkl").exists()


def test_parallel_evaluation_gives_identical_results(tmp_path):
    def run(n_workers):
        cfg = ExperimentConfig()
        cfg.genome.n_motifs = (5, 15)
        cfg.simulator.time_steps = 70
        cfg.fitness.n_reference = cfg.fitness.n_query = 3
        cfg.fitness.n_repetitions = 1
        cfg.evolution.pop_size, cfg.evolution.generations = 4, 2
        cfg.evolution.n_workers = n_workers
        cfg.evolution.out_dir = str(tmp_path / f"w{n_workers}")
        rng = np.random.default_rng(0)
        data = make_temporal_patterns(3, 10, 60, 8, 2.0, 0.0, rng)
        Evolution(cfg, data, None, verbose=False).run()
        log = (tmp_path / f"w{n_workers}" / "log.jsonl").read_text().splitlines()
        return [json.loads(l) for l in log]

    assert run(1) == run(2)


def test_ridge_readout_separates_clusters():
    from darwinsnn.readouts import ridge_accuracy
    rng = np.random.default_rng(3)
    Xtr, ytr = clusters(4, 30, 0.5, rng)
    Xte = Xtr + rng.normal(0, 0.5, Xtr.shape)
    assert ridge_accuracy(Xtr, ytr, Xte, ytr, 4) > 0.95


def test_hypervolume_2d():
    from darwinsnn.analysis import hypervolume_2d
    # unit square reference; points (0.5, 0.5) and (0.25, 0.75) -> 0.25 + 0.25*0.25
    assert hypervolume_2d([(0.5, 0.5), (0.25, 0.75), (0.9, 0.9)], (1.0, 1.0)) == 0.3125
    assert hypervolume_2d([(2.0, 0.1)], (1.0, 1.0)) == 0.0


def test_provenance_reads_commit_without_git_binary():
    import subprocess

    from darwinsnn.provenance import REPO, read_head
    if not (REPO / ".git").exists():
        return
    expected = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    assert read_head(REPO) == expected
    sub = REPO / "third_party" / "pultsito"
    expected_sub = subprocess.run(["git", "-C", str(sub), "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()
    assert read_head(sub) == expected_sub
