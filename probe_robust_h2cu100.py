"""
Test refine_saddle_robust against a real, documented failure: H2_Cu100's
~23 meV second imaginary mode, stable across the original refine_saddle
across both dispersion settings and multiple attempts.

    python probe_robust_h2cu100.py --model uma-s-1p1 --d3 off
    python probe_robust_h2cu100.py --model uma-s-1p1 --d3 off --compare

No agent, no API. GPU only. Mirrors run_seeded.py's own setup exactly
(store.put block copied from there) so this is a fair comparison, not a
different pipeline that happens to call a different function.

--compare runs the ORIGINAL refine_saddle first, confirms it reproduces
the known 23 meV ridge, resets, then runs refine_saddle_robust on the
identical starting state. Costs one extra Sella call to buy that
confirmation; worth it the first time this is run, since a "fix" tested
only against a failure it never independently reproduced is not really
tested.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from ase.io import read, write

import config
from src import store
from src.tools import new_calculator, refine_saddle, refine_saddle_robust
# adsorbate_indices and build_asymptotic are run_seeded.py's own local
# functions, not part of src.tools - imported from there directly so
# this probe uses the exact same asymptotic-state convention rather
# than a second, possibly-diverging copy of it.
from run_seeded import adsorbate_indices, build_asymptotic

SEED_FILE = Path("work_seeds") / "H2_Cu100.traj"


def setup_and_probe(model_key, with_d3, use_robust):
    """Exact mirror of run_seeded.py's setup, up to the refine call."""
    store.reset("H2_Cu100")
    work = config.WORK_DIR

    seed = read(str(SEED_FILE))
    ads = adsorbate_indices(seed)
    if not ads:
        raise SystemExit(f"{SEED_FILE}: no atoms tagged 2.")

    write(str(work / "peak.traj"), seed)

    asym, e_asym, species = build_asymptotic(
        seed, ads, "H2_Cu100", model_key, with_d3, work)

    store.put("initial_relaxed", {
        "energy_eV": e_asym, "converged": True,
        "note": "asymptotic state, mirrors run_seeded.py"})
    store.put("gasref_relaxed", {
        "energy_eV": e_asym, "converged": True, "species": species})

    probe = seed.copy()
    probe.calc = new_calculator(model_key, with_d3=with_d3)
    e_seed = float(probe.get_potential_energy())
    store.put("neb", {
        "barrier_eV": e_seed - e_asym, "with_d3": bool(with_d3),
        "skipped": True, "reason": "seeded track"})

    fn = refine_saddle_robust if use_robust else refine_saddle
    kwargs = {"model_key": model_key, "with_d3": with_d3, "scope": "adsorbate"}
    if use_robust:
        kwargs["max_attempts"] = 3

    msg = fn.invoke(kwargs)
    saddle = store.get("saddle") or {}
    return msg, saddle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="uma-s-1p1")
    ap.add_argument("--d3", choices=["on", "off"], default="off")
    ap.add_argument("--compare", action="store_true",
                    help="also run the original refine_saddle first, "
                         "to confirm it reproduces the known failure")
    args = ap.parse_args()
    with_d3 = args.d3 == "on"

    if not SEED_FILE.exists():
        raise SystemExit(f"{SEED_FILE} not found. Run "
                         "build_seeds.py first.")

    if args.compare:
        print("=" * 70)
        print("STAGE 1: original refine_saddle, confirming the known failure")
        print("=" * 70)
        msg, saddle = setup_and_probe(args.model, with_d3, use_robust=False)
        print(msg)
        imag = saddle.get("imaginary_modes_meV") or []
        print(f"\nimaginary modes: {imag}")
        if len(imag) == 1:
            print("\nNOTE: this attempt found a clean saddle immediately.")
            print("The documented 23 meV ridge did not reproduce this run.")
            print("MLIP relaxations are not perfectly deterministic run to")
            print("run; proceeding to Stage 2 regardless, but read this")
            print("result as 'nothing to recover from today', not as")
            print("refine_saddle_robust's own outcome.")
        else:
            print(f"\nConfirmed: {len(imag)} imaginary modes, matching the")
            print("documented ridge failure. Proceeding to Stage 2.")
        print()

    print("=" * 70)
    print("STAGE 2: refine_saddle_robust")
    print("=" * 70)
    msg, saddle = setup_and_probe(args.model, with_d3, use_robust=True)
    print(msg)

    attempts = store.get("saddle_attempts") or []
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"attempts made: {len(attempts)}")
    print(f"final first_order_saddle: {saddle.get('first_order_saddle')}")
    print(f"final imaginary modes: {saddle.get('imaginary_modes_meV')}")
    print(f"recovered (needed >1 attempt and succeeded): "
          f"{saddle.get('recovered')}")
    if saddle.get("first_order_saddle"):
        print(f"barrier: {saddle.get('barrier_eV'):.3f} eV "
              f"(reference: 0.740 eV)")


if __name__ == "__main__":
    main()
