"""
Calculator construction.

One entry point, several backends. Everything model-specific lives in
config.MODELS; this file only knows how to turn a spec into an ASE
calculator.

Three things worth knowing.

One: a FAIRChemCalculator caches a results buffer sized to the first
structure it sees, so reusing one across systems with different atom
counts raises a numpy broadcast error. Build a fresh one per system.
That rule applies to every backend here - never cache these.

Two: OC20 is trained on RPBE, which contains no dispersion term. For
anything dispersion-dominated the bare model returns a physically wrong
answer that converges cleanly. D3 is added as a second additive
calculator so it contributes forces during relaxation, not just a
single-point correction afterwards - the geometry shift is most of the
effect.

Three: D3 parameters are fitted per functional. A model trained on RPBE
data needs RPBE damping; one trained on PBE data needs PBE. Using one
set across models would make the dispersion comparison meaningless,
which is why d3_xc is a per-model field rather than a global constant.
"""

import sys
from pathlib import Path

from ase.calculators.mixing import SumCalculator

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def model_spec(model_key: str = None) -> dict:
    """The registry entry for one model. Raises clearly on a typo."""
    key = model_key or config.DEFAULT_MODEL
    if key not in config.MODELS:
        raise KeyError(
            f"Unknown model '{key}'. Registered: {', '.join(config.MODELS)}"
        )
    return config.MODELS[key]


def _d3_calculator(spec: dict):
    """A standalone D3 term, for backends without one built in."""
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

    return TorchDFTD3Calculator(
        damping=spec.get("d3_damping", "zero"),
        xc=spec["d3_xc"],
        device=config.DEVICE,
    )


def _build_fairchem(spec: dict):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    unit = load_predict_unit(path=str(spec["path"]), device=config.DEVICE)
    return FAIRChemCalculator(unit, task_name=spec["task"])


def _build_mace(spec: dict, with_d3: bool):
    """MACE applies D3 itself, so dispersion is passed in rather than summed."""
    from mace.calculators import mace_mp

    kwargs = {
        "model": spec["path"],
        "device": config.DEVICE,
        "default_dtype": "float64",
        "dispersion": with_d3,
    }
    if with_d3:
        kwargs["dispersion_xc"] = spec["d3_xc"]
    if spec.get("head"):
        kwargs["head"] = spec["head"]
    return mace_mp(**kwargs)

def _build_orb(spec: dict):
    from orb_models.forcefield import pretrained

    try:
        from orb_models.forcefield.calculator import ORBCalculator
    except ModuleNotFoundError:
        # newer orb-models moved this under .inference
        from orb_models.forcefield.inference.calculator import ORBCalculator

    if not hasattr(pretrained, spec["path"]):
        available = sorted(n for n in dir(pretrained) if n.startswith("orb"))
        raise AttributeError(
            f"orb-models has no loader named '{spec['path']}'.\n"
            f"Available: {', '.join(available)}"
        )

    loader = getattr(pretrained, spec["path"])
    try:
        result = loader(device=config.DEVICE, precision="float32-high")
    except TypeError:
        result = loader(device=config.DEVICE)      # older loaders lack precision

    # newer versions return (model, atoms_adapter); older return model alone
    if isinstance(result, tuple):
        model, atoms_adapter = result[0], result[1]
    else:
        model, atoms_adapter = result, None

    if atoms_adapter is None:
        return ORBCalculator(model, device=config.DEVICE)
    try:
        return ORBCalculator(model, atoms_adapter=atoms_adapter, device=config.DEVICE)
    except TypeError:
        return ORBCalculator(model, device=config.DEVICE)


def new_calculator(model_key: str = None, with_d3: bool = False):
    """A fresh calculator. Never reuse one across different systems.

    model_key indexes config.MODELS; None uses config.DEFAULT_MODEL, so
    existing single-model callers keep working unchanged.
    """
    spec = model_spec(model_key)
    backend = spec["backend"]

    if backend == "mace":
        return _build_mace(spec, with_d3)      # handles its own dispersion

    if backend == "fairchem":
        base = _build_fairchem(spec)
    elif backend == "orb":
        base = _build_orb(spec)
    else:
        raise ValueError(f"Unknown backend '{backend}' for model '{model_key}'")

    if not with_d3:
        return base

    if spec.get("d3") == "none":
        # Asking for D3 on a model whose reference functional already
        # includes dispersion would double-count it. Say so rather than
        # silently returning the bare model.
        raise ValueError(
            f"{model_key or config.DEFAULT_MODEL} was trained against a "
            "dispersion-inclusive functional; adding D3 would double-count it."
        )

    return SumCalculator([base, _d3_calculator(spec)])
