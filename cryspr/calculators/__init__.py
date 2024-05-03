"""Include calculators"""
from packaging import version
import matgl
if version.parse(matgl.__version__) > version.parse("1.0.0"):
    from matgl.ext.ase import PESCalculator
else:
    from matgl.ext.ase import M3GNetCalculator as PESCalculator
from matgl.apps.pes import Potential

from mace.calculators import mace_mp as MACEMPCalculator

from chgnet.model.model import CHGNet
from chgnet.model import CHGNetCalculator as _CHGNetCalculator

class M3GNetCalculator(PESCalculator):
    """
    Wrapper of matgl.ext.ase.PESCalculator
    using eV/A^3 unit for stress and
    pretrained M3GNet-MP-2021.2.8-PES.
    """
    def __init__(
        self,
        potential: Potential = None,
        stress_weight: float = 1 / 160.21766208,
        device = "cpu",
        **kwargs,
    ):
        if potential is None:
            import torch
            torch.set_default_device(device)
            potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")

        super().__init__(potential=potential, stress_weight=stress_weight, **kwargs)

class CHGNetCalculator(_CHGNetCalculator):
    """
    Wrapper of chgnet.model.CHGNetCalculator with cpu as default device.
    """
    def __init__(
            self,
            model: CHGNet = None,
            device: str = "cpu",
    ):
        super().__init__(model=model, use_device=device)

__all__ = [
    "M3GNetCalculator",
    "MACEMPCalculator",
    "CHGNetCalculator",
]