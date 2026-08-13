"""
Calculator construction.

Two things worth knowing here.

One: a FAIRChemCalculator caches a results buffer sized to the first
structure it sees, so reusing one across systems with different atom
counts raises a numpy broadcast error. Build a fresh one per system.

Two: OC20 is trained on RPBE, which contains no dispersion term. For
anything dispersion-dominated the bare model returns a physically wrong
answer that converges cleanly. D3 is added as a second additive
calculator so it contributes forces during relaxation, not just a
single-point correction afterwards - the geometry shift is most of the
effect.
"""

import sys
from pathlib import Path

from ase.calculators.mixing import SumCalculator

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def new_calculator(with_d3: bool = False):
    """A fresh calculator. Never reuse one across different systems."""
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    unit = load_predict_unit(path=str(config.MODEL_PATH), device=config.DEVICE)
    uma = FAIRChemCalculator(unit, task_name=config.TASK_NAME)

    if not with_d3:
        return uma

    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

    d3 = TorchDFTD3Calculator(
        damping=config.D3_DAMPING,
        xc=config.D3_XC,
        device=config.DEVICE,
    )
    return SumCalculator([uma, d3])
