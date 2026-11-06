from .base import SimulationResult, Simulator
from .numpy_sim import NumpySimulator


def make_simulator(cfg) -> Simulator:
    """Create the backend described by a SimulatorConfig."""
    if cfg.backend == "numpy":
        return NumpySimulator(tau=cfg.tau, batch_size=cfg.batch_size)
    if cfg.backend == "pultsito":
        from .pultsito import PultsitoSimulator
        return PultsitoSimulator(cfg.library, cfg.batch_size, cfg.n_threads)
    if cfg.backend == "torch":
        from .torch_sim import TorchSimulator
        return TorchSimulator(tau=cfg.tau, batch_size=cfg.batch_size, device=cfg.device)
    raise ValueError(f"unknown simulator backend {cfg.backend}")


__all__ = ["SimulationResult", "Simulator", "NumpySimulator", "make_simulator"]
