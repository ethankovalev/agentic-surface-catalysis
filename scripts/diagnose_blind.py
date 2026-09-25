"""
Break down one blind run: what every step did, why it failed, and what
the saved structures actually look like.

    python scripts/diagnose_blind.py N2_Ru0001_step
    python scripts/diagnose_blind.py N2_Ru0001_step --model uma-s-1p1 --d3 off

No GPU, no API, changes nothing. Reads the result file and the
structures run_blind.py saved beside it.

WHAT IT PRINTS
--------------
1. The timeline: every tool call in order, with its output.
2. The saddle candidates: each band peak, whether it refined to a first
   order saddle, and whether that saddle connected.
3. Every failed check, with its full message.
4. The saved structures: for each of initial, final, peak and saddle,
   the breaking bond length, and for every adsorbate atom its nearest
   metal distance and how many metal atoms it touches.

The structure table is the part that answers "is this geometry sane".
An N atom bound in a hollow sits about 1.9 to 2.1 A from three metal
atoms. An N atom with zero metal neighbours has desorbed; one with six
or more is inside the slab.
"""

import argparse
import json
from pathlib import Path

from ase.data import covalent_radii
from ase.io import read

RESULTS = Path("/workspace/agentic-surface-catalysis/results/blind")
STRUCTURES = ("initial", "final", "peak", "saddle")
TOUCHING_A = 2.4


def breaking_bond(atoms):
    """(anchor, terminal, length): the farthest adsorbate atom from the
    heaviest one. At a dissociation, that is the bond being broken."""
    tags = atoms.get_tags()
    ads = [i for i in range(len(atoms)) if tags[i] == 2]
    if len(ads) < 2:
        return None
    anchor = max(ads, key=lambda k: covalent_radii[atoms[k].number])
    others = [k for k in ads if k != anchor]
    terminal = max(others, key=lambda k: atoms.get_distance(anchor, k, mic=True))
    return anchor, terminal, atoms.get_distance(anchor, terminal, mic=True)


def describe(atoms):
    """One line per adsorbate atom: nearest metal, and how many it touches."""
    tags = atoms.get_tags()
    ads = [i for i in range(len(atoms)) if tags[i] == 2]
    metal = [i for i in range(len(atoms)) if tags[i] != 2]
    lines = []
    for a in ads:
        distances = [atoms.get_distance(a, m, mic=True) for m in metal]
        nearest = min(distances)
        touching = sum(1 for d in distances if d < TOUCHING_A)
        if touching == 0:
            verdict = "DESORBED"
        elif touching >= 6:
            verdict = "INSIDE THE SLAB?"
        else:
            verdict = "on the surface"
        lines.append(f"      {atoms[a].symbol} atom {a:3d}: nearest metal "
                     f"{nearest:.2f} A, touching {touching}   {verdict}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reaction")
    ap.add_argument("--model", default="uma-s-1p1")
    ap.add_argument("--d3", choices=["on", "off"], default="off")
    ap.add_argument("--results", type=Path, default=RESULTS)
    args = ap.parse_args()

    tag = f"{args.reaction}_{args.model}_d3{args.d3}"
    path = args.results / f"{tag}.json"
    if not path.exists():
        raise SystemExit(f"{path} not found")
    r = json.loads(path.read_text())

    print("=" * 72)
    print(f"{tag}")
    print(f"barrier {r.get('computed_eV')}   reference {r.get('reference_eV')}   "
          f"validated {r.get('validated')}")
    print(f"stopped at {r.get('stopped_at')}   run error {r.get('run_error')}")
    print("=" * 72)

    print("\n1. TIMELINE")
    for n, step in enumerate(r.get("steps", []), 1):
        text = " ".join(str(step.get("output", "")).split())
        print(f"  {n:2d}. {step.get('tool')}  ({step.get('seconds', '?')} s)")
        print(f"      {text[:300]}")

    print("\n2. SADDLE CANDIDATES")
    candidates = r.get("saddle_candidates") or []
    if not candidates:
        print("  none recorded")
    for c in candidates:
        print(f"  {c.get('band')}: band barrier {c.get('band_barrier_eV')}, "
              f"first order {c.get('first_order')}, connects {c.get('connects')}, "
              f"barrier {c.get('barrier_eV')}")
    print(f"  kept: {r.get('saddle_chosen_from')}")

    print("\n3. FAILED CHECKS")
    detail = r.get("validation_detail") or {}
    failed = [k for k, v in (r.get("validation") or {}).items() if not v]
    if not failed:
        print("  none")
    for k in failed:
        print(f"  {k}:")
        print(f"      {detail.get(k, '')}")
    for name, problems in (r.get("structural_problems") or {}).items():
        for p in problems:
            print(f"  structural ({name}): {p}")
    for flag in r.get("energy_flags") or []:
        print(f"  energy flag: {flag}")

    print("\n4. SAVED STRUCTURES")
    folder = args.results / tag
    for name in STRUCTURES:
        f = folder / f"{name}.traj"
        if not f.exists():
            print(f"  {name}: not saved")
            continue
        atoms = read(str(f))
        bond = breaking_bond(atoms)
        bond_text = "n/a" if bond is None else (
            f"{atoms[bond[0]].symbol}-{atoms[bond[1]].symbol} {bond[2]:.2f} A")
        print(f"  {name}: breaking bond {bond_text}")
        for line in describe(atoms):
            print(line)

    print("\n5. SAVED BANDS")
    any_band = False
    for label in ("band1", "band2"):
        f = folder / f"band_{label}.traj"
        if not f.exists():
            continue
        any_band = True
        images = read(str(f), index=":")
        print(f"  {label}: {len(images)} images")
        first = None
        for n, image in enumerate(images):
            try:
                energy = image.get_potential_energy()
                first = energy if first is None else first
                e_text = f"{energy - first:+.3f} eV"
            except Exception:
                e_text = "   n/a   "
            bond = breaking_bond(image)
            b_text = "n/a" if bond is None else f"{bond[2]:.2f} A"
            flags = [line.split()[-1] for line in describe(image)
                     if not line.endswith("on the surface")]
            note = f"   {', '.join(flags)}" if flags else ""
            print(f"    image {n:2d}   E {e_text}   breaking bond {b_text}{note}")
    if not any_band:
        print("  none saved (runs before patch_save_bands.py kept no bands)")


if __name__ == "__main__":
    main()
