"""
Seeded track: literature transition state -> MLIP barrier.

    python scripts/run_seeded.py --seeds work_seeds/ --model uma-s-1p1 --d3 on
    python scripts/run_seeded.py --seeds work_seeds/ --only H2_Cu111

WHAT THIS MEASURES, AND WHAT IT DOES NOT
----------------------------------------
Every geometry here is a BEEF-vdW transition state computed by the authors
of SBH10 and published in their Supporting Information. Starting from it
and refining answers one question:

    given a converged transition state, is the MLIP's energy there correct?

That is a property of the potential energy surface. It is NOT evidence
that this engine can find a transition state, which is the actual product
claim and which only the autonomous pipeline can support. A customer with
a novel catalyst has no SBH10 entry to start from.

Every result written by this script carries provenance="seeded". Nothing
downstream may average a seeded barrier together with an autonomous one,
or report either without the label. The two answer different questions.

WHY THE NUMBERS WILL NOT MATCH THE AUTONOMOUS TRACK
---------------------------------------------------
These cells are 2x2 with 6 layers (Ni(211): 3x3x4). The autonomous
pipeline builds 3x3 with 4 layers. That is 1/4 ML coverage against
1/9 ML, plus a different amount of metal free to relax. Each track is
separately comparable to the SBH10 reference - the reference was computed
in the cell used here - but a seeded-minus-autonomous difference is mostly
coverage, not model error. Do not correct one onto the other.

BARRIER CONVENTION
------------------
SBH10 defines the barrier as

    E_b = E(transition state) - E(asymptotic state)

with both computed in the SAME supercell, the slab held in the same
geometry, and the molecule in the asymptotic state at its equilibrium
geometry far from the surface. This script reproduces that exactly: the
asymptotic state reuses the seed's own slab, atom for atom, and places the
relaxed molecule 8 A above it. Building a separate gas-phase calculation
instead would throw away the error cancellation the convention is designed
to exploit.

That barrier is then zero-point corrected, because SBH10's reference
values are, and a classical barrier compared against them is wrong by
0.03-0.15 eV in a fixed direction on every reaction.

CONNECTIVITY
------------
check_saddle_connects needs an initial/final pair to relax toward and
there is none here. Instead this script stretches and compresses the
breaking bond and relaxes both, requiring one side to fall to the intact
molecule and the other to dissociated fragments. It is recorded as
"bond_displacement_connectivity", not as check_saddle_connects, because it
is the weaker test: it confirms the saddle sits between reactant and
product along the bond coordinate, not that the reaction mode itself
connects them. Reporting it under the stronger name would overstate it.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase.build import molecule as build_molecule
from ase.constraints import FixAtoms
from ase.data import covalent_radii
from ase.io import read, write
from ase.optimize import BFGS

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src import store
from src.calculators import new_calculator
from src.tools import refine_saddle
from src.zpe import compute_zpe_correction

ADSORBATE_ELEMENTS = {"H", "C", "N", "O"}

# Same pattern as scripts/run_grid.py's RESULTS_DIR: an env var, not a
# config.py attribute (config.py has no such attribute - it exposes
# WORK_DIR and OUTPUT_DIR only). Written to the network volume, not the
# repo, for the same reason as the grid track: the repo's local disk is
# wiped when the pod goes away.
#
# Deliberately a DIFFERENT directory from the grid track's default
# (results/seeded, not results/grid). These are two different questions
# with two different answers - see the module docstring - and giving them
# the same directory would invite exactly the accidental blending this
# script exists to prevent.
SEEDED_RESULTS_DIR = Path(os.environ.get(
    "SEEDED_RESULTS_DIR",
    "/workspace/agentic-surface-catalysis/results/seeded"))

# Height of the molecule above the top metal layer in the asymptotic
# state. SBH10 used at least 10 A of vacuum; 8 A of separation puts the
# molecule outside any chemical interaction while staying inside the cell
# these geometries were built with.
ASYMPTOTIC_HEIGHT = 8.0

# Displacement applied to the breaking bond for the connectivity test.
CONNECTIVITY_PUSH = 0.35

GAS_FMAX = 0.02
GAS_STEPS = 200
CONNECTIVITY_STEPS = 60


def adsorbate_indices(atoms):
    tags = atoms.get_tags()
    return [i for i in range(len(atoms)) if tags[i] == 2]


def breaking_bond(atoms, ads):
    """Anchor and terminal atom of the dissociating bond at a saddle.

    The 2.2x reach matters: at a transition state the breaking bond is
    already stretched past equilibrium - CH4/Ni(100) sits at 1.91 A
    against a 1.09 A normal C-H - so an intact-molecule cutoff excludes
    exactly the bond being looked for and silently returns a spectator.
    """
    anchor = max(ads, key=lambda k: covalent_radii[atoms[k].number])
    r_a = covalent_radii[atoms[anchor].number]
    neighbours = [
        k for k in ads
        if k != anchor
        and atoms.get_distance(anchor, k, mic=True)
        < 2.2 * (r_a + covalent_radii[atoms[k].number])
    ]
    if not neighbours:
        raise ValueError("adsorbate has no bonded pair at the saddle")
    terminal = max(neighbours,
                   key=lambda k: atoms.get_distance(anchor, k, mic=True))
    return anchor, terminal, atoms.get_distance(anchor, terminal, mic=True)


def infer_molecule(atoms, ads):
    """Which gas-phase molecule the adsorbate fragments came from."""
    counts = {}
    for i in ads:
        counts[atoms[i].symbol] = counts.get(atoms[i].symbol, 0) + 1
    known = {
        ("H", 2): "H2",
        ("N", 2): "N2",
    }
    if len(counts) == 1:
        (sym, n), = counts.items()
        if (sym, n) in known:
            return known[(sym, n)]
    if counts == {"C": 1, "H": 4}:
        return "CH4"
    raise ValueError(
        f"cannot identify the gas-phase molecule from adsorbate composition "
        f"{counts}. Add it to infer_molecule() rather than guessing.")


def build_asymptotic(seed, ads, name, model_key, with_d3, work):
    """Seed's slab, unchanged, plus the relaxed molecule 8 A above it.

    The slab is copied atom for atom from the transition state rather than
    rebuilt. SBH10's convention holds the metal geometry identical between
    the two states so the barrier is a difference of like with like; a
    freshly relaxed slab would put a real energy difference into the
    reference that has nothing to do with the reaction.
    """
    metal = [i for i in range(len(seed)) if i not in ads]
    asym = seed[metal]

    species = infer_molecule(seed, ads)
    mol = build_molecule(species)
    if len(mol) != len(ads):
        raise ValueError(
            f"{name}: seed has {len(ads)} adsorbate atoms but {species} has "
            f"{len(mol)}")

    top_z = max(asym.positions[:, 2])
    centre = asym.cell[0][:2] / 2 + asym.cell[1][:2] / 2
    mol.positions -= mol.positions.mean(axis=0)
    mol.positions[:, 0] += centre[0]
    mol.positions[:, 1] += centre[1]
    mol.positions[:, 2] += top_z + ASYMPTOTIC_HEIGHT

    if mol.positions[:, 2].max() > asym.cell[2][2] - 1.0:
        raise ValueError(
            f"{name}: the molecule at {ASYMPTOTIC_HEIGHT} A would sit within "
            "1 A of the cell top and interact with its own periodic image. "
            "The seed's vacuum gap is too small for this convention.")

    asym += mol
    tags = [1 if abs(asym.positions[i, 2] - top_z) < 0.8 else 0
            for i in range(len(metal))]
    tags += [2] * len(mol)
    asym.set_tags(tags)

    # Freeze every metal atom: only the molecule's internal geometry is
    # being relaxed here. Letting the slab move would relax it away from
    # the transition state's slab and break the cancellation above.
    asym.set_constraint(FixAtoms(indices=list(range(len(metal)))))
    asym.calc = new_calculator(model_key, with_d3=with_d3)

    opt = BFGS(asym, logfile="-")
    converged = bool(opt.run(fmax=GAS_FMAX, steps=GAS_STEPS))
    energy = float(asym.get_potential_energy())

    if not converged:
        raise ValueError(
            f"{name}: the molecule did not relax to fmax {GAS_FMAX} eV/A in "
            f"{GAS_STEPS} steps. The asymptotic energy is the zero of the "
            "barrier, so an unconverged one shifts every number derived from "
            "it.")

    write(str(work / "gasref.traj"), asym)
    return asym, energy, species


def connectivity_by_bond_displacement(name, model_key, with_d3, work):
    """Push the breaking bond both ways and see where each side falls.

    Weaker than following the imaginary mode, and named accordingly. What
    it rules out is a saddle that sits somewhere other than between the
    intact molecule and the dissociated fragments - which is exactly the
    failure that produced a confirmed first-order saddle for subsurface H
    penetration on H2/Cu(111) earlier in this project.
    """
    saddle = read(str(work / "saddle.traj"))
    ads = adsorbate_indices(saddle)
    anchor, terminal, r_saddle = breaking_bond(saddle, ads)
    intact = covalent_radii[saddle[anchor].number] + covalent_radii[saddle[terminal].number]

    axis = saddle.positions[terminal] - saddle.positions[anchor]
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        raise ValueError("breaking bond has zero length")
    axis = axis / norm

    results = {}
    for label, sign in (("compressed", -1.0), ("stretched", +1.0)):
        trial = saddle.copy()
        trial.positions[terminal] += sign * CONNECTIVITY_PUSH * axis
        trial.calc = new_calculator(model_key, with_d3=with_d3)
        BFGS(trial, logfile="-").run(fmax=0.05, steps=CONNECTIVITY_STEPS)
        results[label] = float(trial.get_distance(anchor, terminal, mic=True))

    # Compressed should fall back toward a normal bond, stretched should
    # run away from it. Thresholds are relative to the intact bond length
    # so they mean the same thing for H-H, N-N and C-H.
    recombined = results["compressed"] < intact * 1.35
    dissociated = results["stretched"] > max(r_saddle, intact) * 1.25
    connects = recombined and dissociated

    record = {
        "method": "bond_displacement",
        "connects": connects,
        "bond_at_saddle_A": round(r_saddle, 3),
        "intact_bond_A": round(intact, 3),
        "after_compression_A": round(results["compressed"], 3),
        "after_stretch_A": round(results["stretched"], 3),
        "push_A": CONNECTIVITY_PUSH,
        "note": ("weaker than mode-following connectivity; confirms the "
                 "saddle lies between reactant and product along the bond "
                 "coordinate only"),
    }
    if not connects:
        why = []
        if not recombined:
            why.append(
                f"compressing the bond left it at {results['compressed']:.2f} A "
                f"instead of falling back toward {intact:.2f} A")
        if not dissociated:
            why.append(
                f"stretching the bond left it at {results['stretched']:.2f} A "
                "instead of running away to dissociation")
        record["failure"] = "; ".join(why)
    return record


def run_one(name, seed_file, model_key, with_d3, work):
    store.reset(f"{name}__seeded")
    store.put("provenance", "seeded")
    store.put("seed_source", "SBH10 SI, BEEF-vdW transition state geometry")
    store.put("seed_file", str(seed_file))

    seed = read(str(seed_file))
    ads = adsorbate_indices(seed)
    if not ads:
        raise ValueError(
            f"{name}: no atoms tagged 2. build_seeds.py writes the tags; this "
            "trajectory was not produced by it.")

    anchor, terminal, r_b = breaking_bond(seed, ads)

    # refine_saddle starts from peak.traj. Here the "peak" is the published
    # transition state rather than a band's highest image.
    write(str(work / "peak.traj"), seed)

    asym, e_asym, species = build_asymptotic(
        seed, ads, name, model_key, with_d3, work)

    # refine_saddle measures its barrier from initial_relaxed.energy_eV.
    # Setting that to the asymptotic energy makes the refined barrier the
    # gas-referenced barrier directly, in SBH10's convention, with no
    # physisorption-well correction needed - there is no physisorbed state
    # in this convention to correct from.
    store.put("initial_relaxed", {
        "energy_eV": e_asym,
        "converged": True,
        "note": "asymptotic state: seed slab + relaxed molecule at "
                f"{ASYMPTOTIC_HEIGHT} A (SBH10 E_asym convention)",
    })
    store.put("gasref_relaxed", {
        "energy_eV": e_asym, "converged": True, "species": species})
    store.put("neb", {
        "barrier_eV": None,
        "with_d3": bool(with_d3),
        "skipped": True,
        "reason": "seeded track: the transition state is given, not searched for",
    })

    saddle_msg = refine_saddle.invoke({
        "model_key": model_key, "with_d3": with_d3, "scope": "adsorbate"})

    saddle = store.get("saddle") or {}
    result = {
        "reaction": name,
        "provenance": "seeded",
        "model_key": model_key,
        "with_d3": bool(with_d3),
        "molecule": species,
        "seed_r_b_A": round(r_b, 3),
        "asymptotic_energy_eV": e_asym,
        "saddle_message": saddle_msg,
        "saddle_converged": saddle.get("converged"),
        "first_order_saddle": saddle.get("first_order_saddle"),
        "imaginary_modes_meV": saddle.get("imaginary_modes_meV"),
        "shift_from_seed_eV": saddle.get("shift_from_neb_peak_eV"),
    }

    if not (saddle.get("converged") and saddle.get("first_order_saddle")):
        result["barrier_eV"] = None
        result["status"] = "no confirmed first-order saddle"
        return result

    barrier_classical = saddle["barrier_eV"]
    store.put("barrier_gas_eV", barrier_classical)
    result["barrier_classical_eV"] = barrier_classical

    # Drift from the published geometry is itself a measurement: a large
    # shift means the MLIP's saddle is somewhere the reference functional's
    # is not, which is a model difference worth reporting rather than
    # absorbing silently.
    refined = read(str(work / "saddle.traj"))
    try:
        _, _, r_b_refined = breaking_bond(refined, adsorbate_indices(refined))
        result["refined_r_b_A"] = round(r_b_refined, 3)
        result["r_b_drift_A"] = round(r_b_refined - r_b, 3)
    except ValueError as exc:
        result["refined_r_b_A"] = None
        result["r_b_drift_note"] = str(exc)

    try:
        conn = connectivity_by_bond_displacement(name, model_key, with_d3, work)
    except Exception as exc:
        conn = {"method": "bond_displacement", "connects": False,
                "failure": f"{type(exc).__name__}: {exc}"}
    store.put("saddle_connectivity", conn)
    result["connectivity"] = conn

    zpe_msg = compute_zpe_correction.invoke({
        "model_key": model_key, "with_d3": with_d3})
    zpe = store.get("zpe") or {}
    result["zpe_message"] = zpe_msg
    result["dzpe_eV"] = zpe.get("dzpe_eV")
    result["barrier_eV"] = store.get("barrier_zpe_eV")
    result["barrier_convention"] = store.get("barrier_convention")

    if result["barrier_eV"] is None:
        result["status"] = "saddle found but zero-point correction failed"
    elif not conn.get("connects"):
        result["status"] = "barrier computed but connectivity not established"
    else:
        result["status"] = "ok"
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=Path, default=Path("work_seeds"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--d3", choices=["on", "off"], default="on")
    ap.add_argument("--only", default=None,
                    help="run a single reaction by name")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    model_key = args.model or config.DEFAULT_MODEL
    with_d3 = args.d3 == "on"
    work = Path(config.WORK_DIR)
    work.mkdir(parents=True, exist_ok=True)

    seeds = sorted(args.seeds.glob("*.traj"))
    if args.only:
        seeds = [s for s in seeds if s.stem == args.only]
        if not seeds:
            raise SystemExit(
                f"no seed named {args.only!r} in {args.seeds}/. Available: "
                f"{sorted(s.stem for s in args.seeds.glob('*.traj'))}")
    if not seeds:
        raise SystemExit(
            f"no .traj files in {args.seeds}/. Run build_seeds.py first.")

    out = args.out or (SEEDED_RESULTS_DIR
                       / f"seeded_{model_key}_d3{args.d3}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    if out.exists():
        results = json.loads(out.read_text())
        print(f"resuming: {len(results)} already done in {out}")

    for seed_file in seeds:
        name = seed_file.stem
        if name in results and results[name].get("barrier_eV") is not None:
            print(f"skip {name} (already has a barrier)")
            continue
        print(f"\n{'=' * 70}\n{name}  [{model_key}, d3={args.d3}]\n{'=' * 70}")
        try:
            results[name] = run_one(name, seed_file, model_key, with_d3, work)
        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")
            results[name] = {
                "reaction": name, "provenance": "seeded",
                "model_key": model_key, "with_d3": with_d3,
                "barrier_eV": None,
                "status": f"crashed: {type(exc).__name__}: {exc}",
            }
        out.write_text(json.dumps(results, indent=2))
        r = results[name]
        print(f"-> {r.get('status')}  barrier={r.get('barrier_eV')}  "
              f"dZPE={r.get('dzpe_eV')}")

    print(f"\n{'=' * 70}")
    print(f"{'reaction':22s} {'barrier':>9s} {'classical':>10s} "
          f"{'dZPE':>7s} {'r_b drift':>10s}  status")
    for name, r in sorted(results.items()):
        def fmt(key, w, p=3):
            v = r.get(key)
            return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else f"{'--':>{w}}"
        print(f"{name:22s} {fmt('barrier_eV', 9)} {fmt('barrier_classical_eV', 10)} "
              f"{fmt('dzpe_eV', 7)} {fmt('r_b_drift_A', 10)}  {r.get('status')}")
    print(f"\nwritten to {out}")
    print("provenance=seeded on every entry: these are PES-accuracy results "
          "at published saddles, not transition-state search results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
