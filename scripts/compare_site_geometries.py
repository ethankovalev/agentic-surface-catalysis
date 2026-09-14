"""
Compare the two relaxed CH4/Ru(0001) endpoints saved by probe_site_fix.py.

Run AFTER probe_site_fix.py's Stage 1 has completed (it needs the patched
version that saves final_hcp.traj and final_fcc.traj). Pure CPU, no
calculator, no GPU, no API - this only reads two trajectory files already
on disk.

    python scripts/compare_site_geometries.py

WHAT IT SETTLES
----------------
Stage 1 reported both relaxations converging to the identical energy,
-316.0989878029285 eV, to 13 significant figures. That precision is too
exact to be two genuinely different local minima by coincidence - real
distinct minima on a PES essentially never agree past 4-6 figures. Two
explanations:

  A. Physically real: the fragment migrated during relaxation from its
     hcp starting point to the same basin fcc would have reached. Not a
     bug - a real statement that these two hollows are not separate
     minima for this species on this surface.

  B. A bug: something served a stale/cached result instead of running an
     independent second relaxation.

This script tells them apart by comparing the ACTUAL ATOMIC POSITIONS of
the two saved endpoints, not just their energies. If A is true, the
anchor atoms should have ended up at (near enough) the same xy
coordinates regardless of which hollow they started from. If B is true,
the starting geometries logged in Stage 1 were already different (they
were, verified separately) but the "relaxed" positions would either be
identical to those different starting points (relaxation silently did
nothing) or otherwise inconsistent with a real BFGS trajectory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from ase.io import read
from ase.data import covalent_radii

import config

ADS = {"H", "C", "N", "O"}


def anchor_positions(atoms):
    """xy of the CH3 carbon anchor and the lone H, by element."""
    syms = atoms.get_chemical_symbols()
    ads = [i for i in range(len(atoms)) if syms[i] in ADS]
    c = [i for i in ads if syms[i] == "C"][0]
    # the departed hydrogen: the H atom furthest from the carbon
    hs = [i for i in ads if syms[i] == "H"]
    h = max(hs, key=lambda i: atoms.get_distance(c, i, mic=True))
    return {"C (CH3 anchor)": atoms.positions[c, :2],
            "H (departed)": atoms.positions[h, :2]}


def main():
    work = Path(config.WORK_DIR)
    hcp_path = work / "final_hcp.traj"
    fcc_path = work / "final_fcc.traj"

    for p in (hcp_path, fcc_path):
        if not p.exists():
            raise SystemExit(
                f"{p} not found. Rerun the patched probe_site_fix.py "
                "Stage 1 first - the version that saves per-site copies.")

    hcp = read(str(hcp_path))
    fcc = read(str(fcc_path))

    print(f"hcp: {hcp_path}  ({len(hcp)} atoms)")
    print(f"fcc: {fcc_path}  ({len(fcc)} atoms)\n")

    hp = anchor_positions(hcp)
    fp = anchor_positions(fcc)

    print(f"{'atom':20s} {'hcp xy':>18s} {'fcc xy':>18s} {'lateral delta':>14s}")
    max_delta = 0.0
    for label in hp:
        d = float(np.linalg.norm(hp[label] - fp[label]))
        max_delta = max(max_delta, d)
        print(f"{label:20s} {str(np.round(hp[label], 3)):>18s} "
              f"{str(np.round(fp[label], 3)):>18s} {d:14.3f}")

    print()
    if max_delta < 0.15:
        print("SAME GEOMETRY (within 0.15 A). This supports explanation A:")
        print("the fragment migrated during relaxation to a common basin.")
        print("Not a bug. This IS the finding: fcc and hcp are not separate")
        print("minima for this species on this surface, on this PES.")
    else:
        print(f"DIFFERENT GEOMETRIES (max lateral delta {max_delta:.3f} A).")
        print("The two relaxations reached genuinely different structures")
        print("despite reporting the same energy to 13 figures. That")
        print("combination - different geometry, identical energy - would")
        print("itself be extremely unusual and is worth showing me before")
        print("trusting either number.")


if __name__ == "__main__":
    main()
