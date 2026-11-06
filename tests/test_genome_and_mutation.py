import numpy as np
import pytest
from helpers import check_network

from darwinsnn.config import GenomeConfig, MutationConfig
from darwinsnn.genome import (
    Genome,
    build_network,
    connect,
    connect_input,
    new_motif_gene,
    random_genome,
)
from darwinsnn.motifs import MOTIF_INDEX, MOTIFS
from darwinsnn.mutation import OPERATORS, mutate


def test_motif_library():
    assert [m.name for m in MOTIFS] == ["FFE", "FBE", "FBI", "RCE", "RCI", "LTI", "CPG"]
    assert MOTIFS[MOTIF_INDEX["RCI"]].signs == (-1, -1, -1, -1)


def test_link_expands_to_all_outputs_times_inputs():
    gc = GenomeConfig(multiplicity=(2, 2))
    rng = np.random.default_rng(0)
    g = Genome(n_channels=3)
    a = g.add_motif(new_motif_gene(MOTIF_INDEX["CPG"], gc, rng))   # 3 outputs
    b = g.add_motif(new_motif_gene(MOTIF_INDEX["LTI"], gc, rng))   # 2 inputs
    connect(g, a, b, gc, rng)
    connect_input(g, 0, a, gc, rng)
    net = check_network(g, gc)
    internal = len(MOTIFS[MOTIF_INDEX["CPG"]].edges) + len(MOTIFS[MOTIF_INDEX["LTI"]].edges)
    # link: multiplicity 2 x 3 CPG outputs x 2 LTI inputs; input link: the 3 CPG inputs
    assert net.n_synapses == internal + 2 * 3 * 2 + 3


def test_prune_removes_unreachable_motifs():
    gc, rng = GenomeConfig(), np.random.default_rng(0)
    g = Genome(n_channels=1)
    a, b, c = (g.add_motif(new_motif_gene(0, gc, rng)) for _ in range(3))
    connect_input(g, 0, a, gc, rng)
    connect(g, a, b, gc, rng)            # c is not reachable
    connect(g, c, a, gc, rng)
    assert g.prune_unreachable() == 1
    assert set(g.motifs) == {a, b} and (c, a) not in g.links


@pytest.mark.parametrize("evolve_weights", [False, True])
def test_random_mutations_keep_genomes_valid(evolve_weights):
    gc = GenomeConfig(n_motifs=(10, 30), evolve_weights=evolve_weights)
    mc = MutationConfig(n_operators=2)
    rng = np.random.default_rng(1)
    g = random_genome(gc, n_channels=8, rng=rng)
    for _ in range(200):
        g, _ = mutate(g, gc, mc, rng)
        if g.n_motifs == 0:
            g = random_genome(gc, 8, rng)
        net = check_network(g, gc)
        assert (net.weight_magnitude is not None) == evolve_weights
        for gene in g.motifs.values():
            assert np.all((gene.v_thresh >= gc.v_thresh[0]) & (gene.v_thresh <= gc.v_thresh[1]))
            r = gene.refractory
            assert np.all((r >= gc.refractory[0]) & (r <= gc.refractory[1]))


@pytest.mark.parametrize("name", list(OPERATORS))
def test_each_operator_runs_on_an_independent_copy(name):
    gc = GenomeConfig(n_motifs=(10, 20), evolve_weights=True)
    rng = np.random.default_rng(2)
    parent = random_genome(gc, 5, rng)
    snapshot = parent.copy()
    child = parent.copy()
    OPERATORS[name](child, gc, MutationConfig(), rng)
    assert build_network(parent).n_synapses == build_network(snapshot).n_synapses
    assert all(np.array_equal(parent.motifs[u].v_thresh, snapshot.motifs[u].v_thresh)
               for u in parent.motifs)
