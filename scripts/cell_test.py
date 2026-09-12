"""
Controlled cell-sensitivity test.

    python scripts/cell_test.py --reaction H2_Cu111 --d3 off
    python scripts/cell_test.py --reaction H2_Cu111 --d3 off --cells 3x3x4

WHAT QUESTION THIS ANSWERS
--------------------------
The seeded track runs in SBH10's cell (2x2x6, 1/4 ML). The autonomous
track runs in 3x3x4 (1/9 ML). Their barriers differ, but that difference
currently confounds three things at once:

    1. lateral coverage      1/4 ML vs 1/9 ML
    2. slab thickness        6 layers vs 4
    3. geometry source       published BEEF-vdW saddle vs NEB search

This script holds 3 fixed and varies 1 and 2. Everything else - the same
deterministic tool sequence, the same model, the same dispersion setting,
the same thresholds - is identical between the two runs. Whatever gap
remains is the cell.

WHY NOT JUST COMPARE AGAINST THE EXISTING AGENT RESULT
------------------------------------------------------
The existing 3x3x4 barrier came from the LangGraph agent, which chooses
its own site, image count and retry behaviour. Comparing an agent run at
one cell against a scripted run at another would fold agent variability
into a number meant to isolate cell size. So this script runs BOTH cells
itself, with no agent in the loop. The agent's number stays a separate
data point about the agent, not about the cell.

INTERPRETING THE RESULT
-----------------------
Small gap (under ~0.05 eV): cell choice is not what separates the two
    tracks, every existing 3x3x4 result stands, and the seeded/autonomous
    difference is mostly geometry source. Do not migrate the pipeline.

Large gap: MLIP surface barriers are cell-sensitive at a magnitude that
    matters, which is worth reporting in its own right - most published
    MLIP benchmarks do not state their cell, and this would show why they
    should.

Either way the answer is a measurement, not a guess, and either way it is
cheaper than migrating the pipeline and re-running everything to find out.

NOTE ON THE REFERENCE CONVENTION
--------------------------------
Barriers here are gas-referenced and zero-point corrected, the same
convention as the seeded track and as SBH10 itself. The reference values
are experimental (or SRP-DFT fitted to experiment), which are
low-coverage quantities: 3x3 at 1/9 ML is CLOSER to that limit than 2x2
at 1/4 ML, not further. Matching SBH10's cell matches their methodology,
not the physics their reference encodes. Worth keeping in mind when
reading the gap.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src import store
from src.benchmark import SBH10
from src.tools import (
    build_dissociated_endpoint,
    build_gas_reference,
    build_slab,
    check_endpoints_are_minima,
    check_saddle_connects,
    compute_gas_referenced_barrier,
    place_adsorbate,
    refine_saddle,
    relax_structure,
    run_neb,
)
from src.zpe import compute_zpe_correction

RESULTS_DIR = Path(os.environ.get(
    "CELL_TEST_RESULTS_DIR",
    "/workspace/agentic-surface-catalysis/results/cell_test"))

# (nx, ny, layers, n_fixed_layers)
#   3x3x4 - the autonomous pipeline's default, 1/9 ML, top 2 layers free
#   2x2x6 - SBH10's cell, 1/4 ML, top 2 layers free
# Both leave exactly two layers mobile, so slab flexibility is held
# constant and only coverage and total thickness change.
CELLS = {
    "3x3x4": (3, 3, 4, 2),
    "2x2x6": (2, 2, 6, 4),
}


def coverage(nx, ny):
    """Adsorbate per surface metal atom, as a fraction of a monolayer."""
    return 1.0 / (nx * ny)


def step(label, message):
    print(f"\n--- {label}")
    print(message)
    if message.startswith("FAILED") or "DID NOT CONVERGE" in message:
        raise RuntimeError(f"{label}: {message}")
    return message


def run_cell(reaction, cell_name, model_key, with_d3, n_images):
    spec = SBH10[reaction]
    nx, ny, layers, n_fixed = CELLS[cell_name]

    store.reset(f"{reaction}__cell{cell_name}")
    store.put("provenance", "autonomous_scripted")
    store.put("cell", {"nx": nx, "ny": ny, "layers": layers,
                       "n_fixed_layers": n_fixed,
                       "coverage_ML": coverage(nx, ny)})

    print(f"\n{'=' * 72}")
    print(f"{reaction}  cell={cell_name}  ({coverage(nx, ny):.3f} ML)  "
          f"model={model_key}  d3={'on' if with_d3 else 'off'}")
    print('=' * 72)

    if spec.get("site_type") == "step":
        raise RuntimeError(
            f"{reaction} is a step reaction. build_stepped_slab has its own "
            "geometry conventions and is not a drop-in for build_slab here; "
            "run this test on a terrace reaction.")

    step("build_slab", build_slab.invoke({
        "metal": spec["metal"], "facet": spec["facet"],
        "nx": nx, "ny": ny, "layers": layers, "n_fixed_layers": n_fixed}))

    step("place_adsorbate", place_adsorbate.invoke({
        "species": spec["molecule"], "site": "ontop"}))

    step("build_dissociated_endpoint", build_dissociated_endpoint.invoke({}))

    for structure in ("initial", "final"):
        step(f"relax_structure({structure})", relax_structure.invoke({
            "structure": structure, "model_key": model_key, "with_d3": with_d3}))

    step("build_gas_reference", build_gas_reference.invoke({}))
    step("relax_structure(gasref)", relax_structure.invoke({
        "structure": "gasref", "model_key": model_key, "with_d3": with_d3}))

    step("check_endpoints_are_minima", check_endpoints_are_minima.invoke({
        "model_key": model_key, "with_d3": with_d3}))

    # Not wrapped in step(): an unconverged band is expected often enough
    # that refine_saddle exists precisely to rescue it, so it must not
    # abort the run.
    print("\n--- run_neb")
    print(run_neb.invoke({"n_images": n_images, "model_key": model_key,
                          "with_d3": with_d3}))

    print("\n--- refine_saddle")
    print(refine_saddle.invoke({"model_key": model_key, "with_d3": with_d3,
                                "scope": "adsorbate"}))

    print("\n--- check_saddle_connects")
    print(check_saddle_connects.invoke({"model_key": model_key,
                                        "with_d3": with_d3}))

    print("\n--- compute_gas_referenced_barrier")
    print(compute_gas_referenced_barrier.invoke({}))

    print("\n--- compute_zpe_correction")
    print(compute_zpe_correction.invoke({"model_key": model_key,
                                         "with_d3": with_d3}))

    saddle = store.get("saddle") or {}
    conn = store.get("saddle_connectivity") or {}
    zpe = store.get("zpe") or {}
    neb = store.get("neb") or {}

    return {
        "reaction": reaction,
        "cell": cell_name,
        "nx": nx, "ny": ny, "layers": layers, "n_fixed_layers": n_fixed,
        "coverage_ML": coverage(nx, ny),
        "provenance": "autonomous_scripted",
        "model_key": model_key,
        "with_d3": bool(with_d3),
        "n_images": n_images,
        "reference_eV": spec.get("reference_eV"),
        "neb_barrier_eV": neb.get("barrier_eV"),
        "neb_converged": neb.get("converged"),
        "saddle_converged": saddle.get("converged"),
        "first_order_saddle": saddle.get("first_order_saddle"),
        "imaginary_modes_meV": saddle.get("imaginary_modes_meV"),
        "connects": conn.get("connects"),
        "barrier_classical_eV": store.get("barrier_classical_eV"),
        "dzpe_eV": zpe.get("dzpe_eV"),
        "barrier_eV": store.get("barrier_eV"),
        "barrier_convention": store.get("barrier_convention"),
        "validation": store.validation(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reaction", default="H2_Cu111")
    ap.add_argument("--cells", default="3x3x4,2x2x6",
                    help="comma-separated, from " + ",".join(CELLS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--d3", choices=["on", "off"], default="off")
    ap.add_argument("--n-images", type=int, default=12)
    args = ap.parse_args()

    if args.reaction not in SBH10:
        raise SystemExit(f"unknown reaction {args.reaction!r}. "
                         f"Known: {sorted(SBH10)}")
    cells = [c.strip() for c in args.cells.split(",")]
    for c in cells:
        if c not in CELLS:
            raise SystemExit(f"unknown cell {c!r}. Known: {sorted(CELLS)}")

    model_key = args.model or config.DEFAULT_MODEL
    with_d3 = args.d3 == "on"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.reaction}_{model_key}_d3{args.d3}.json"
    results = json.loads(out.read_text()) if out.exists() else {}

    for cell_name in cells:
        try:
            results[cell_name] = run_cell(
                args.reaction, cell_name, model_key, with_d3, args.n_images)
        except Exception as exc:
            print(f"\nFAILED in {cell_name}: {type(exc).__name__}: {exc}")
            results[cell_name] = {
                "reaction": args.reaction, "cell": cell_name,
                "model_key": model_key, "with_d3": with_d3,
                "barrier_eV": None,
                "status": f"{type(exc).__name__}: {exc}",
            }
        out.write_text(json.dumps(results, indent=2))

    print(f"\n{'=' * 72}")
    print(f"CELL SENSITIVITY: {args.reaction}, {model_key}, d3={args.d3}")
    print('=' * 72)
    ref = SBH10[args.reaction].get("reference_eV")
    print(f"{'cell':8s} {'ML':>6s} {'barrier':>9s} {'classical':>10s} "
          f"{'dZPE':>7s} {'vs ref':>8s}  saddle")
    for cell_name in cells:
        r = results.get(cell_name, {})
        def fmt(key, w, p=3):
            v = r.get(key)
            return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else f"{'--':>{w}}"
        b = r.get("barrier_eV")
        err = f"{b - ref:+8.3f}" if isinstance(b, (int, float)) and ref else f"{'--':>8}"
        ok = ("1st-order" if r.get("first_order_saddle")
              else r.get("status", "not confirmed"))
        print(f"{cell_name:8s} {fmt('coverage_ML', 6, 3)} {fmt('barrier_eV', 9)} "
              f"{fmt('barrier_classical_eV', 10)} {fmt('dzpe_eV', 7)} {err}  {ok}")

    got = [results[c].get("barrier_eV") for c in cells
           if isinstance(results.get(c, {}).get("barrier_eV"), (int, float))]
    if len(got) == len(cells) and len(got) > 1:
        gap = max(got) - min(got)
        print(f"\ncell gap: {gap:.3f} eV")
        if gap < 0.05:
            print("Under 0.05 eV. Cell choice is not what separates the seeded "
                  "and autonomous tracks; the existing 3x3x4 results stand and "
                  "there is no case for migrating the pipeline.")
        else:
            print("Above 0.05 eV. Cell size materially moves the barrier - "
                  "report it as a result, and state the cell alongside every "
                  "barrier in the write-up.")
    else:
        print("\nNot every cell produced a barrier, so no gap can be quoted. "
              "Fix the failing cell before drawing a conclusion from this.")

    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
