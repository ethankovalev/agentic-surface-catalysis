"""
Settings for the SBH10 agentic benchmark.

Everything tunable lives here. Nothing in src/ should hardcode a path,
a threshold, or a model name.
"""

import os
from pathlib import Path

# --- paths ------------------------------------------------------------

ROOT = Path(__file__).parent
WORK_DIR = ROOT / "work"          # scratch: structures, trajectories
OUTPUT_DIR = ROOT / "outputs"     # results, logs

WORK_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# --- the MLIP ---------------------------------------------------------

# Gated checkpoint. Request access at huggingface.co/facebook/UMA,
# then either put the file in data/ or set UMA_CHECKPOINT.
MODEL_PATH = os.environ.get("UMA_CHECKPOINT", str(ROOT / "data" / "uma-s-1p2.pt"))
TASK_NAME = "oc20"
DEVICE = os.environ.get("UMA_DEVICE", "cpu")   # "cuda" if you have one

# UMA's benchmarked mean absolute error against its reference DFT.
# Any energy difference smaller than this is not distinguishable
# from zero, whatever the optimiser reports.
NOISE_FLOOR_EV = 0.3 # highest approximate bound of the range 0.1 eV and 0.3 eV for UMA's benchmarked MAE against its reference DFT.

# --- dispersion -------------------------------------------------------

# OC20 is trained on RPBE, which has no dispersion term. Use RPBE
# parameters for D3, not PBE - they are fitted per functional.
D3_DAMPING = "zero"
D3_XC = "rpbe"

# --- relaxation and NEB ----------------------------------------------

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


def check_checkpoint():
    """Fail early and clearly if the model weights are missing."""
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"UMA checkpoint not found at {MODEL_PATH}\n"
            "Request access at https://huggingface.co/facebook/UMA, then "
            "place uma-s-1p2.pt in data/ or set UMA_CHECKPOINT."
        )


def as_dict():
    """The subset the graph needs."""
    return {
        "LANGSIM_MODEL": LANGSIM_MODEL,
        "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
        "working_directory": str(WORK_DIR),
    }
