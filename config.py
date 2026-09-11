"""
Settings for the SBH10 agentic benchmark.

Everything tunable lives here. Nothing in src/ should hardcode a path,
a threshold, or a model name.
"""

import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "data" / "hf_cache"))

# --- paths ------------------------------------------------------------

ROOT = Path(__file__).parent
WORK_DIR = ROOT / "work"          # scratch: structures, trajectories
OUTPUT_DIR = ROOT / "outputs"     # results, logs

WORK_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# --- the MLIPs --------------------------------------------------------
#
# training_domain and in_domain_for_surfaces are not decoration. OC20 is
# adsorbates on slabs; OMat24 is bulk inorganic crystals; OMol25 is
# isolated molecules at a molecular level of theory. Running a molecular
# model on a periodic slab is out of domain by construction, and the
# results table has to say so or it reads as a fair fight when it isn't.
#
# d3_xc is per functional, not per taste. D3 parameters are fitted to a
# specific functional - RPBE for OC20, PBE for OMat24. Getting this wrong
# invalidates every dispersion comparison in the benchmark.
DEVICE = os.environ.get("MLIP_DEVICE", os.environ.get("UMA_DEVICE", "cpu"))
MODELS = {
    "uma-s-1p1": {
        "backend": "fairchem",
        "path": os.environ.get("UMA_CHECKPOINT", str(ROOT / "data" / "uma-s-1p1.pt")),
        "task": "oc20",
        "d3": "torch_dftd",
        "d3_xc": "rpbe",
        "d3_damping": "zero",
        "training_domain": "OC20 + OMat24 + OMol25 + ODAC23 + OMC25",
        "in_domain_for_surfaces": True,
        "noise_floor_eV": 0.3,
        "model_resolution_eV": 0.05,
        "reproducibility_eV": None,
        "gated": True,
    },
    "mace-mh-1": {
        "backend": "mace",
        "path": str(ROOT / "data" / "mace-mh-1" / "mace-mh-1.model"),         
        "head": "oc20_usemppbe",
        "d3": "builtin",              # mace_mp takes dispersion=True itself
        "d3_xc": "pbe",
        "training_domain": "OMat24 pretrain + multi-head (OC20, OMol, MPTraj)",
        "in_domain_for_surfaces": True,
        "noise_floor_eV": 0.3,
        "gated": False,
        "license_note": "ASL - academic/non-commercial only",
    },
    "mace-mpa-0": {
        "backend": "mace",
        "path": str(ROOT / "data" / "mace-mpa-0" / "mace-mpa-0-medium.model"),
        "head": None,
        "d3": "builtin",
        "d3_xc": "pbe",
        "training_domain": "Materials Project + Alexandria (bulk crystals)",
        "in_domain_for_surfaces": False,
        "noise_floor_eV": 0.3,
        "gated": False,
    },
    "orb-v3-cons-inf-omat": {
        "backend": "orb",
        "path": "orb_v3_conservative_inf_omat",
        "d3": "torch_dftd",
        "d3_xc": "pbe",
        "d3_damping": "bj",
        "training_domain": "OMat24 (bulk inorganic crystals)",
        "in_domain_for_surfaces": False,
        "noise_floor_eV": 0.3,
        "gated": False,
    },
    "esen-sm-cons-omol": {
        "backend": "fairchem",
        "path": os.environ.get("ESEN_CHECKPOINT", str(ROOT / "data" / "esen_sm_conserving_all.pt")),
        "task": "omol",
        "d3": "none",                 # omega-B97M-V already includes dispersion
        "d3_xc": None,
        "training_domain": "OMol25 (isolated molecules, wB97M-V)",
        "in_domain_for_surfaces": False,
        "noise_floor_eV": 0.3,
        "gated": True,
    },
}

# noise_floor_eV conflated two different failures and so measured
# neither. They are now separate fields.
#
#   model_resolution_eV  - the model's own energy resolution. Below this a
#       number is not distinguishable from zero however tightly the
#       optimiser converged. A property of the checkpoint.
#
#   reproducibility_eV   - the spread of the whole pipeline over repeat
#       runs from perturbed starting geometries. A property of the harness,
#       dominated by which basin the fragments land in, and in practice
#       much larger than the model resolution. None means NOT YET MEASURED,
#       and any claim about a difference smaller than it is unsupported.
#
# Both default to the legacy noise_floor_eV where a model has not been
# measured, so nothing silently becomes more confident than it was.
for _spec in MODELS.values():
    _spec.setdefault("model_resolution_eV", _spec.get("noise_floor_eV"))
    _spec.setdefault("reproducibility_eV", None)

DEFAULT_MODEL = os.environ.get("MLIP_MODEL", "uma-s-1p1")

# --- relaxation and NEB ----------------------------------------------

# Every check the run must pass before it can finish. validation_summary
# reports against this and exit_gate enforces it, so the two can never
# disagree about what "validated" means. A check that exists but is not
# listed here is optional, and an optional check is one the agent can skip
# on the run where it would have failed.
REQUIRED_CHECKS = (
    "convergence",
    "noise_floor",
    "dispersion",
    "dispersion_consistent",
    "gas_reference",
    "fragments",
    "geometry",
    "reaction_consistency",
    "endpoints_distinct",
    "path_resolved",
    "zpe",
)

FMAX = 0.02              # eV/A for endpoint relaxations
FMAX_NEB = 0.10          # eV/A for the band
MAX_STEPS = 300
N_IMAGES = 10            # intermediate images; band is N_IMAGES + 2
NEB_SPRING_K = 0.3

# --- physical sanity thresholds --------------------------------------

# A physisorbed species sits roughly 3.2-3.6 A above the surface plane.
# Much further out means there is no binding well at all, which for a
# dispersion-dominated system usually means a missing vdW term.
MAX_PHYSISORPTION_HEIGHT = 4.2

# Below this, an adsorbate atom is unphysically close to the metal.
MIN_CONTACT = 1.5

# --- the agent --------------------------------------------------------

LANGSIM_MODEL = os.environ.get("LANGSIM_MODEL", "claude-sonnet-4-6")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

MAX_ATTEMPTS = 10        # supervisor turns before the run is abandoned
RECURSION_LIMIT = 200

def check_checkpoint(model_key: str = None):
    """Fail early and clearly if a local checkpoint is missing."""
    spec = MODELS[model_key or DEFAULT_MODEL]

    if spec["backend"] == "orb":
        return                        # orb resolves its own weights by tag

    if not Path(spec["path"]).exists():
        if spec["backend"] == "fairchem":
            raise FileNotFoundError(
                f"Checkpoint for {model_key or DEFAULT_MODEL} not found at {spec['path']}\n"
                "FAIR Chemistry weights are gated: request access at "
                "https://huggingface.co/facebook/UMA (or /facebook/OMol25), agree to "
                "the licence, then place the file in data/ or set the env var."
            )
        raise FileNotFoundError(
            f"Checkpoint for {model_key or DEFAULT_MODEL} not found at {spec['path']}\n"
            f"Download it: hf download mace-foundations/{model_key or DEFAULT_MODEL} "
            f"--local-dir data/{model_key or DEFAULT_MODEL}, then check the actual "
            "filename with ls and update MODELS[...]['path'] to match."
        )

def as_dict():
    """The subset the graph needs."""
    return {
        "LANGSIM_MODEL": LANGSIM_MODEL,
        "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
        "working_directory": str(WORK_DIR),
    }
