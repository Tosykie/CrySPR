"""Include calculators"""
import matgl
from matgl.ext.ase import PESCalculator
from matgl.apps.pes import Potential

from mace.calculators import mace_mp, MACECalculator

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
        potential: Potential = matgl.load_model("M3GNet-MP-2021.2.8-PES"),
        stress_weight: float = 1 / 160.21766208,
        **kwargs,
    ):
        super().__init__(potential=potential, stress_weight=stress_weight, **kwargs)


# Override the original MACECalculator
# _MACECalculator: MACECalculator = mace_mp()
# class MaceMPCalculator(_MACECalculator):
#     """
#     Wrapper of mace.calculators.mace_mp function using
#     pretrained model based on the Materials Project, with
#     cpu as compute device and float32 precision.
#     """
#     def __init__(
#         self,
#         model = None,
#         device: str = "cpu",
#         default_dtype: str = "float32",
#         **kwargs,
#     ):
#         super().__init__(model=model, device=device, default_dtype=default_dtype, **kwargs)

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

