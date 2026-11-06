"""Library of neural motifs: the building blocks of every network.

A motif is a small circuit with a fixed internal wiring. When two motifs are
connected, every *output* neuron of the source motif is wired to every
*input* neuron of the target motif.

The seven motifs follow DarwinSNNs (github.com/AlexAmenabar/DarwinSNNs).
In FBI the output is the excitatory neuron 1; set outputs=(2,) to output
from the inhibitory neuron instead.

Neuron signs follow Dale's law: a neuron is excitatory (+1) or inhibitory
(-1), and that sign is applied to all of its outgoing synapses.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Motif:
    name: str
    signs: tuple[int, ...]               # +1 excitatory / -1 inhibitory, one per neuron
    edges: tuple[tuple[int, int], ...]   # internal synapses (pre, post), local indices
    inputs: tuple[int, ...]              # neurons that receive connections from other motifs
    outputs: tuple[int, ...]             # neurons that project to other motifs

    @property
    def n_neurons(self) -> int:
        return len(self.signs)

    def __post_init__(self):
        n = len(self.signs)
        for pre, post in self.edges:
            assert 0 <= pre < n and 0 <= post < n, f"{self.name}: edge out of range"
        assert all(0 <= i < n for i in self.inputs + self.outputs)
        assert self.inputs and self.outputs, f"{self.name}: needs inputs and outputs"


E, I = +1, -1

MOTIFS: tuple[Motif, ...] = (
    # Feed-forward excitation: 0 -> 1
    Motif("FFE", (E, E), ((0, 1),), inputs=(0,), outputs=(1,)),
    # Feedback excitation: 0 <-> 1
    Motif("FBE", (E, E), ((0, 1), (1, 0)), inputs=(0,), outputs=(1,)),
    # Feedback inhibition: 0 -> 1 -> 2 -| 0
    Motif("FBI", (E, E, I), ((0, 1), (1, 2), (2, 0)), inputs=(0,), outputs=(1,)),
    # Recurrent excitation: ring 0 -> 1 -> 2 -> 3 -> 0
    Motif("RCE", (E, E, E, E), ((0, 1), (1, 2), (2, 3), (3, 0)),
          inputs=(0, 1, 2, 3), outputs=(0, 1, 2, 3)),
    # Recurrent inhibition: same ring, all inhibitory
    Motif("RCI", (I, I, I, I), ((0, 1), (1, 2), (2, 3), (3, 0)),
          inputs=(0, 1, 2, 3), outputs=(0, 1, 2, 3)),
    # Lateral inhibition: 0 -> 1 -| 2
    Motif("LTI", (E, I, E), ((0, 1), (1, 2)), inputs=(0, 2), outputs=(0, 2)),
    # Central pattern generator: all-to-all excitatory triangle
    Motif("CPG", (E, E, E), ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)),
          inputs=(0, 1, 2), outputs=(0, 1, 2)),
)

MOTIF_INDEX: dict[str, int] = {m.name: i for i, m in enumerate(MOTIFS)}


def motif_ids(names: list[str] | None) -> list[int]:
    """Translate motif names from a config into indices (None = all motifs)."""
    if names is None:
        return list(range(len(MOTIFS)))
    return [MOTIF_INDEX[n] for n in names]
