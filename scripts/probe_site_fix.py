"""
Test yesterday's fcc/hcp hypothesis on CH4_Ru0001, directly, no agent.

    python scripts/probe_site_fix.py --model uma-s-1p1 --d3 off

Uses .invoke() on the tools directly, the same pattern as cell_test.py
and the manual CH4_Ru0001 diagnostic from earlier this week. NO CALL
TOUCHES THE ANTHROPIC API. Every dollar this costs is GPU time only.

WHAT THIS ANSWERS
------------------
Yesterday: CH4_Ru0001's endpoint search puts both fragments on HCP
hollows; SBH10's Table S1 specifies FCC for both. Hypothesis: the
endpoint was never a genuine local minimum because it wasn't started in
the right basin, which is consistent with the observed instability
(re-relaxation drifting 3.9 A back toward recombination, a persistent
8-11 meV second imaginary mode at the saddle).

This script tests that hypothesis the cheap way first, then the
expensive way only if the cheap answer justifies it:

  STAGE 1 (cheap, ~a minute): build both a HCP and an FCC endpoint,
      relax each, compare final energies. If FCC is NOT lower, the
      hypothesis is wrong on energy grounds alone and there is no
      reason to spend more budget chasing it - stop and report that.

  STAGE 2 (expensive, most of the GPU spend): only runs if FCC won
      Stage 1. Takes the FCC endpoint through the full chain - NEB,
      refine_saddle, connectivity, ZPE - and reports whether it now
      produces a confirmed first-order saddle, which yesterday's HCP
      endpoint could not.

A COST CHECKPOINT IS PRINTED BETWEEN THE TWO STAGES. Read it. Stage 2
is the part that burns the bulk of a small GPU budget (NEB + two
Hessians). Do not run this unattended if funds are tight - watch the
pod's cost as Stage 1 finishes and decide before Stage 2 starts.

WHY NOT USE THE BLIND SITE_TYPE PREFERENCE FROM SBH10_PREFERRED_SITE
----------------------------------------------------------------------
That dict encodes SBH10's OWN answer (Table S1), so using it inside the
autonomous pipeline would be a reference leak - the point of the
autonomous track is that no literature geometry informs it. This script
instead makes the choice on PURE ENERGY: build both candidates, relax
both, keep whichever is actually lower on this model's own PES. That is
blind by construction and, unlike the literature dict, generalizes to
every reaction with more than one hollow type - not just the one this
was checked on.

If FCC turns out lower here, that is also independent evidence UMA's
own PES agrees with SBH10's site assignment, which is worth reporting
regardless of what Stage 2 finds.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src import store
from src.tools import (
    build_slab, place_adsorbate, build_dissociated_endpoint,
    relax_structure, check_endpoints_are_minima, build_gas_reference,
    run_neb, refine_saddle, check_saddle_connects,
    compute_gas_referenced_barrier,
)
from src.zpe import compute_zpe_correction


def stage1_compare_sites(model_key, with_d3):
    """Build both site types, relax the final endpoint for each, compare.

    Deliberately runs hcp FIRST and fcc SECOND, so that on return the
    work/ directory and the store are left holding the RELAXED fcc
    structures. Stage 2 then continues from them instead of rebuilding
    and re-relaxing, which on a tight GPU budget is two whole relaxations
    saved for free.

    Each relaxed final.traj is ALSO copied to final_<kind>.traj before the
    next iteration overwrites it. Without this, two relaxations that
    happen to converge to the same energy to many decimal places cannot
    be told apart from a caching bug that never actually ran a second,
    independent relaxation - the only file left on disk would be
    whichever ran last, with nothing to compare it against.
    """
    import shutil
    results = {}
    for kind in ("hcp", "fcc"):
        print(f"\n{'=' * 60}\n  candidate site: {kind}\n{'=' * 60}")
        store.reset(f"probe_{kind}")

        print(build_slab.invoke({"metal": "Ru", "facet": "0001",
                                 "nx": 3, "ny": 3, "layers": 4}))
        print(place_adsorbate.invoke({"species": "CH4", "site": "ontop"}))
        print(build_dissociated_endpoint.invoke({"site_type": (kind, kind)}))

        print(relax_structure.invoke({
            "structure": "initial", "model_key": model_key, "with_d3": with_d3}))
        msg = relax_structure.invoke({
            "structure": "final", "model_key": model_key, "with_d3": with_d3})
        print(msg)

        final = store.get("final_relaxed") or {}
        results[kind] = {
            "converged": final.get("converged"),
            "energy_eV": final.get("energy_eV"),
        }

        src = Path(config.WORK_DIR) / "final.traj"
        dst = Path(config.WORK_DIR) / f"final_{kind}.traj"
        if src.exists():
            shutil.copy(src, dst)
            print(f"  saved a copy for later comparison: {dst.name}")

    return results


def stage2_full_chain(model_key, with_d3, n_images, rebuild):
    """Take the FCC endpoint through NEB, refinement, connectivity, ZPE.

    rebuild=False (the normal path): stage 1 has just run, so work/ holds
        the relaxed fcc initial and final states and the store holds their
        energies. Continue straight from there - rebuilding would discard
        two relaxations that were just paid for.

    rebuild=True (--stage2-only): nothing has run in this process, so the
        structures have to be built and relaxed first.
    """
    if rebuild:
        store.reset("probe_fcc_full")
        print(build_slab.invoke({"metal": "Ru", "facet": "0001",
                                 "nx": 3, "ny": 3, "layers": 4}))
        print(place_adsorbate.invoke({"species": "CH4", "site": "ontop"}))
        print(build_dissociated_endpoint.invoke({"site_type": ("fcc", "fcc")}))
        print(relax_structure.invoke({
            "structure": "initial", "model_key": model_key, "with_d3": with_d3}))
        print(relax_structure.invoke({
            "structure": "final", "model_key": model_key, "with_d3": with_d3}))
    else:
        print("continuing from stage 1's relaxed fcc structures "
              "(not rebuilding - that would repeat two relaxations)")

    print(build_gas_reference.invoke({}))
    print(relax_structure.invoke({
        "structure": "gasref", "model_key": model_key, "with_d3": with_d3}))
    print(check_endpoints_are_minima.invoke({
        "model_key": model_key, "with_d3": with_d3}))

    print(f"\n--- run_neb, n_images={n_images} ---")
    print(run_neb.invoke({"n_images": n_images, "model_key": model_key,
                          "with_d3": with_d3}))

    print("\n--- refine_saddle ---")
    print(refine_saddle.invoke({"model_key": model_key, "with_d3": with_d3,
                                "scope": "adsorbate"}))

    print("\n--- check_saddle_connects ---")
    print(check_saddle_connects.invoke({"model_key": model_key, "with_d3": with_d3}))

    print("\n--- compute_gas_referenced_barrier ---")
    print(compute_gas_referenced_barrier.invoke({}))

    print("\n--- compute_zpe_correction ---")
    print(compute_zpe_correction.invoke({"model_key": model_key, "with_d3": with_d3}))

    saddle = store.get("saddle") or {}
    return {
        "saddle_converged": saddle.get("converged"),
        "first_order_saddle": saddle.get("first_order_saddle"),
        "imaginary_modes_meV": saddle.get("imaginary_modes_meV"),
        "connects": (store.get("saddle_connectivity") or {}).get("connects"),
        "barrier_eV": store.get("barrier_eV"),
        "dzpe_eV": (store.get("zpe") or {}).get("dzpe_eV"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--d3", choices=["on", "off"], default="off")
    ap.add_argument("--n-images", type=int, default=12)
    ap.add_argument("--stage2-only", action="store_true",
                    help="skip the comparison, assume fcc already won it")
    args = ap.parse_args()

    model_key = args.model or config.DEFAULT_MODEL
    with_d3 = args.d3 == "on"

    if not args.stage2_only:
        print("STAGE 1: which site is actually lower in energy on this PES?")
        print("(this part is cheap - two CH4 relaxations, a minute or two)\n")
        cmp = stage1_compare_sites(model_key, with_d3)

        print(f"\n{'=' * 60}\nSTAGE 1 RESULT\n{'=' * 60}")
        for kind, r in cmp.items():
            print(f"  {kind}: converged={r['converged']}  "
                  f"energy={r['energy_eV']}")

        hcp_e, fcc_e = cmp["hcp"]["energy_eV"], cmp["fcc"]["energy_eV"]
        if hcp_e is None or fcc_e is None:
            print("\nOne of the two relaxations did not report an energy. "
                  "Stop here and look at the log above before spending more "
                  "budget - do not proceed to Stage 2 on an unclear result.")
            return 1

        fcc_wins = fcc_e < hcp_e
        print(f"\nfcc {'IS' if fcc_wins else 'is NOT'} lower "
              f"(fcc={fcc_e:.4f} eV, hcp={hcp_e:.4f} eV, "
              f"delta={fcc_e - hcp_e:+.4f} eV)")

        print(f"\n{'=' * 60}\nCOST CHECKPOINT\n{'=' * 60}")
        print("Stage 1 is done. Stage 2 (NEB + saddle refinement + two")
        print("Hessians) is the expensive part of this script - most of")
        print("today's GPU budget, if you have limited funds, goes here.")
        print("Check your RunPod balance now before continuing.\n")

        if not fcc_wins:
            print("fcc did NOT come out lower on this model's own PES.")
            print("The site-search hypothesis is not supported by energy")
            print("alone. Stage 2 would still be informative (a corrected")
            print("site could still resolve connectivity even without being")
            print("the global minimum) but is a weaker bet - decide with")
            print("that in mind rather than running it automatically.")
            print("\nRerun with --stage2-only to proceed anyway.")
            return 0

        print("fcc IS lower. Proceeding to Stage 2 automatically.")
        print("(Ctrl+C now if you want to stop and check budget first -")
        print(" nothing expensive has run yet.)\n")

    print(f"\n{'=' * 60}\nSTAGE 2: full chain on the FCC endpoint\n{'=' * 60}")
    result = stage2_full_chain(model_key, with_d3, args.n_images,
                               rebuild=args.stage2_only)

    print(f"\n{'=' * 60}\nFINAL RESULT\n{'=' * 60}")
    for k, v in result.items():
        print(f"  {k}: {v}")

    if result.get("first_order_saddle"):
        print("\nCONFIRMED FIRST-ORDER SADDLE on the fcc-corrected endpoint.")
        print("This is real evidence the site fix resolves the instability -")
        print("compare against yesterday's hcp result: 2 imaginary modes,")
        print("8-11 meV second mode, endpoint drifting 3.9 A on re-relaxation.")
    else:
        print("\nStill not a confirmed first-order saddle. The site fix")
        print("alone did not resolve this reaction - worth logging as a")
        print("negative result: the fcc/hcp mismatch was real, but was not")
        print("(or was not the only) cause of the instability.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
