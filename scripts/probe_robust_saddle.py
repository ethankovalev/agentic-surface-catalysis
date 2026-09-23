"""
Test refine_saddle_robust against a seeded reaction's saddle failure.

    python scripts/probe_robust_saddle.py --reaction CH4_Ru0001 --d3 off --compare
    python scripts/probe_robust_saddle.py --reaction H2_Cu100   --d3 off

No agent, no API, GPU only. Mirrors run_seeded.py's own setup exactly
(the store.put block is copied from there), so this is a fair comparison
against the documented track record rather than a different pipeline
that happens to call a different function.

Generalises the earlier H2_Cu100-only probe, which resolved that
reaction's ~23 meV ridge once the displacement went from 0.1 A (too
small, 3 attempts, still 2 modes) to 0.3 A (2 attempts, 1 mode,
0.743 eV against a 0.740 reference).

--compare runs the ORIGINAL refine_saddle first, to confirm the
documented failure reproduces today before crediting any recovery. MLIP
relaxations are not bit-identical run to run.

ON THE RIDGE DISPLACEMENT
--------------------------
refine_saddle_robust takes no displacement argument. The step size is a
literal in src/tools.py:

    displacement = 0.3 * second

To try a different value, edit that line. This script deliberately does
NOT accept a --displacement flag: an earlier draft did, and passed it
through as a kwarg the tool does not accept, which would have crashed
on the first real run. A flag that cannot work is worse than no flag.

ON REFERENCES
-------------
Reference barriers are read from src.benchmark.SBH10, not copied here.
An earlier draft hardcoded them, which would have silently diverged the
moment the open SBH17 question on CH4_Ni111_step (0.80 against 0.699)
is settled. This script is not a tool and is not reachable by the agent,
so importing benchmark is consistent with the blind-evaluation rule,
which constrains tools rather than scripted probes.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ase.io import read, write

import config
from src import store
from src.benchmark import SBH10
from src.tools import new_calculator, refine_saddle, refine_saddle_robust
# adsorbate_indices and build_asymptotic are run_seeded.py's own local
# functions, not part of src.tools. Imported from there directly so this
# probe uses the identical asymptotic-state convention rather than a
# second copy that could drift from it.
from run_seeded import (adsorbate_indices, build_asymptotic,
                       connectivity_by_bond_displacement)


def reference_eV(reaction):
    return SBH10[reaction].get("reference_eV")


def setup_and_probe(reaction, model_key, with_d3, use_robust):
    """Exact mirror of run_seeded.py's setup, up to the refine call."""
    seed_file = Path("work_seeds") / f"{reaction}.traj"
    if not seed_file.exists():
        raise SystemExit(
            f"{seed_file} not found. Run:\n"
            f"  python build_seeds.py data_sbh10_si/poscars.txt --out work_seeds/")

    store.reset(reaction)
    work = config.WORK_DIR

    seed = read(str(seed_file))
    ads = adsorbate_indices(seed)
    if not ads:
        raise SystemExit(
            f"{seed_file}: no atoms tagged 2. build_seeds.py writes those "
            "tags; this trajectory did not come from it.")

    write(str(work / "peak.traj"), seed)

    asym, e_asym, species = build_asymptotic(
        seed, ads, reaction, model_key, with_d3, work)

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

    kwargs = {"model_key": model_key, "with_d3": with_d3,
              "scope": "adsorbate"}
    if use_robust:
        kwargs["max_attempts"] = 3
        msg = refine_saddle_robust.invoke(kwargs)
    else:
        msg = refine_saddle.invoke(kwargs)

    saddle = store.get("saddle") or {}
    connectivity = None
    if saddle.get("first_order_saddle"):
        try:
            connectivity = connectivity_by_bond_displacement(
                reaction, model_key, with_d3, work)
        except Exception as exc:
            connectivity = {"connects": None,
                            "error": f"{type(exc).__name__}: {exc}"}

    return msg, saddle, (e_seed - e_asym), connectivity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reaction", required=True, choices=sorted(SBH10))
    ap.add_argument("--model", default="uma-s-1p1")
    ap.add_argument("--d3", choices=["on", "off"], default="off")
    ap.add_argument("--compare", action="store_true",
                    help="run the original refine_saddle first, to confirm "
                         "the documented failure reproduces today")
    args = ap.parse_args()

    with_d3 = args.d3 == "on"
    ref = reference_eV(args.reaction)

    if args.compare:
        print("=" * 70)
        print(f"STAGE 1: original refine_saddle on {args.reaction}")
        print("=" * 70)
        msg, saddle, _, _ = setup_and_probe(
            args.reaction, args.model, with_d3, use_robust=False)
        print(msg)
        imag = [round(float(m), 1)
                for m in (saddle.get("imaginary_modes_meV") or [])]
        print(f"\nimaginary modes: {imag}")
        if len(imag) == 1:
            print("\nNOTE: a clean saddle was found on the first attempt, so")
            print("the documented failure did not reproduce this run. Read")
            print("Stage 2 as 'nothing to recover from today' rather than as")
            print("evidence about refine_saddle_robust either way.")
        elif len(imag) == 0:
            print("\nZero imaginary modes: this fell into a minimum, which is")
            print("basin collapse rather than a ridge. Stage 2 will retry")
            print("with a smaller trust radius, not a displacement.")
        else:
            print(f"\nConfirmed: {len(imag)} imaginary modes, matching the")
            print("documented failure. Proceeding.")
        print()

    print("=" * 70)
    print(f"STAGE 2: refine_saddle_robust on {args.reaction}")
    print("=" * 70)
    msg, saddle, at_reference, connectivity = setup_and_probe(
        args.reaction, args.model, with_d3, use_robust=True)
    print(msg)
    if connectivity is not None:
        print()
        print(f"connectivity (bond displacement): {connectivity}")

    attempts = store.get("saddle_attempts") or []
    imag = [round(float(m), 1)
            for m in (saddle.get("imaginary_modes_meV") or [])]

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"reaction:                 {args.reaction}")
    print(f"model / dispersion:       {args.model} / D3 {args.d3}")
    print(f"attempts made:            {len(attempts)}")
    print(f"first order saddle:       {saddle.get('first_order_saddle')}")
    print(f"imaginary modes (meV):    {imag}")
    print(f"recovered:                {saddle.get('recovered')}")
    print(f"barrier at published TS:  {at_reference:+.3f} eV")

    if saddle.get("first_order_saddle"):
        barrier = saddle.get("barrier_eV")
        if ref is None:
            print(f"refined barrier:          {barrier:.3f} eV "
                  "(no reference recorded for this reaction)")
        else:
            print(f"refined barrier:          {barrier:.3f} eV   "
                  f"reference {ref:.3f}   error {barrier - ref:+.3f}")
    else:
        print("\nNot resolved. The ridge displacement is a literal in "
              "src/tools.py:")
        print("    displacement = 0.3 * second")
        print("H2_Cu100 needed 0.3 after 0.1 proved too small, so a larger")
        print("value is the next thing to try. If the modes barely move")
        print("across attempts, that is evidence the second mode is a real")
        print("feature of this model's surface rather than a symmetry")
        print("artifact, which is itself worth recording.")


if __name__ == "__main__":
    main()
