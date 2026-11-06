import numpy as np
import pytest
from helpers import check_network

from darwinsnn.config import GenomeConfig
from darwinsnn.crossover import cluster_crossover, connected_region, region_crossover
from darwinsnn.genome import random_genome


def clustered_config(**kw):
    return GenomeConfig(n_motifs=(40, 80), n_clusters=(4, 8), p_inter_cluster=0.1, **kw)


def test_clustered_genomes_are_modular():
    gc, rng = clustered_config(), np.random.default_rng(0)
    fractions = []
    for _ in range(20):
        g = random_genome(gc, 10, rng)
        assert 4 <= len(g.clusters()) <= 8
        fractions.append(g.n_inter_cluster_links() / len(g.links))
    assert 0.05 < np.mean(fractions) < 0.2           # close to p_inter_cluster = 0.1


def test_unstructured_genomes_have_one_cluster():
    g = random_genome(GenomeConfig(n_motifs=(30, 50)), 10, np.random.default_rng(0))
    assert list(g.clusters()) == [0] and g.n_inter_cluster_links() == 0


def test_connected_region_has_requested_size():
    g = random_genome(GenomeConfig(n_motifs=(30, 50)), 10, np.random.default_rng(1))
    region = connected_region(g, 12, np.random.default_rng(2))
    assert len(region) == 12 and region <= set(g.motifs)


@pytest.mark.parametrize("op,clustered", [(region_crossover, False), (region_crossover, True),
                                          (cluster_crossover, True)])
def test_children_are_valid(op, clustered):
    gc = clustered_config() if clustered else GenomeConfig(n_motifs=(40, 80))
    rng = np.random.default_rng(3)
    for _ in range(30):
        a, b = random_genome(gc, 10, rng), random_genome(gc, 10, rng)
        child, stats = op(a, b, gc, rng)
        check_network(child, gc)
        assert 0 < child.n_motifs <= a.n_motifs + b.n_motifs
        assert stats["links_kept"] + stats["links_cut"] > 0
        clusters = sorted(child.clusters())
        assert clusters == list(range(len(clusters)))     # ids renumbered 0..k-1
        if not clustered:
            assert clusters == [0]


def test_cluster_crossover_cuts_fewer_links():
    """The motivation for clustered genomes, checked on random parents."""
    gc, rng = clustered_config(), np.random.default_rng(4)
    flat = GenomeConfig(n_motifs=(40, 80))
    cut_cluster, cut_region = [], []
    for _ in range(40):
        a, b = random_genome(gc, 10, rng), random_genome(gc, 10, rng)
        cut_cluster.append(cluster_crossover(a, b, gc, rng)[1]["cut_fraction"])
        a, b = random_genome(flat, 10, rng), random_genome(flat, 10, rng)
        cut_region.append(region_crossover(a, b, flat, rng)[1]["cut_fraction"])
    assert np.mean(cut_cluster) < 0.5 * np.mean(cut_region)
