"""
Structural sanity checks: catch unphysical structures, not just
inaccurate barriers.

    python patch_structure_sanity.py --check src/tools.py
    python patch_structure_sanity.py src/tools.py

WHY
---
An inaccurate barrier on a correct structure is off by some amount. A
wrong structure is not the reaction at all, and MLIPs produce wrong
structures with ordinary-looking energies and no warning. That is the
larger failure mode, and until now nothing looked for most of it.

check_geometry already caught three things: an adsorbate atom inside a
bond length of the metal, a barrier peak at a band endpoint, and an
adsorbate buried in the slab. This adds three more, all reference-free,
so none of them needs to know what the answer should be:

  DESORPTION   a fragment that has drifted off the surface. Thresholds
               differ by structure: the initial state is legitimately
               physisorbed (CH4 on Ni sat at a 3.68 A closest contact),
               so it is only flagged past 5.0 A. In the final state and
               at the saddle the fragments must be chemically bound, so
               3.0 A. The gas reference is deliberately far from the
               surface and is never checked.

  FUSED ATOMS  two adsorbate atoms closer than 0.6 x their covalent
               radius sum. No bond is that short; it means atoms have
               been pushed into each other.

  DAMAGED SLAB a metal atom whose nearest metal neighbour is more than
               15 percent off the slab's median nearest neighbour
               distance, or one with fewer than three metal neighbours.
               Distances only, never heights: stepped slabs are vicinal
               cells whose surface is a tilted staircase, and a height
               test flagged every atom of a perfectly ordinary published
               N2/Ru(0001) step as pulled out of the slab. Real surfaces relax by a few
               percent. An out-of-domain model can wreck the slab
               itself, and no existing check looked at the metal at all.

WHERE IT RUNS
-------------
In check_geometry, over initial, final and saddle, so it is enforced by
the existing gate through REQUIRED_CHECKS.

And after every call to relax_structure, as a warning in the tool's
return string, so the agent sees the problem at the step that caused it
rather than several steps later at validation.

LIMITS
------
These catch failures that look the same on every system. They cannot
anticipate a failure specific to one chemistry. Passing them rules out
these particular ways of being unphysical, nothing more.
"""

import sys
from pathlib import Path


OLD_HELPER = '''@tool
def check_geometry() -> str:'''

NEW_HELPER = '''# Closest approach of any atom in a fragment to any metal atom, beyond
# which the fragment counts as desorbed. The initial state is allowed to
# be physisorbed; the final state and saddle must be chemically bound.
DESORBED_BEYOND_A = {"initial": 5.0, "final": 3.0, "saddle": 3.0}

# Two adsorbate atoms closer than this multiple of their covalent radius
# sum have been pushed into each other. No bond is that short.
FUSED_BELOW_RATIO = 0.6

# A surface metal atom whose nearest metal neighbour is this far off the
# slab's median is not a relaxed surface. Real surfaces relax by a few
# percent.
SURFACE_NN_TOLERANCE = 0.15

# A metal atom with fewer metal neighbours than this has been pulled out
# of the slab. Coordination rather than height, because a stepped slab is
# a vicinal cell whose surface is a tilted staircase: on the published
# N2/Ru(0001) step, heights "above the surface layer" climbed to 13 A
# across perfectly ordinary terraces. A top-layer atom on a close packed
# face has nine metal neighbours, a step edge atom seven, an adatom three.
EXTRACTED_BELOW_COORDINATION = 3


def _fragments(atoms, ads):
    """Group adsorbate atoms into bonded fragments."""
    groups = []
    unassigned = list(ads)
    while unassigned:
        group = [unassigned.pop()]
        grew = True
        while grew:
            grew = False
            for k in list(unassigned):
                for g in group:
                    limit = 1.3 * (covalent_radii[atoms[k].number]
                                   + covalent_radii[atoms[g].number])
                    if atoms.get_distance(k, g, mic=True) < limit:
                        group.append(k)
                        unassigned.remove(k)
                        grew = True
                        break
        groups.append(group)
    return groups


def _structural_problems(atoms, name):
    """Reference-free list of ways this structure is unphysical.

    name is one of initial, final, saddle, gasref. gasref is never
    checked: its molecule is placed far from the surface on purpose.
    """
    if name == "gasref":
        return []

    problems = []
    tags = atoms.get_tags()
    ads = [i for i in range(len(atoms)) if tags[i] == 2]
    metal = [i for i in range(len(atoms)) if tags[i] != 2]
    if not metal:
        return problems

    # desorption
    limit = DESORBED_BEYOND_A.get(name)
    if limit is not None and ads:
        for frag in _fragments(atoms, ads):
            closest = min(atoms.get_distance(a, m, mic=True)
                          for a in frag for m in metal)
            if closest > limit:
                syms = "".join(sorted(atoms[a].symbol for a in frag))
                problems.append(
                    f"{name}: fragment {syms} is {closest:.2f} A from the "
                    f"nearest metal atom, past the {limit} A limit, so it "
                    f"has desorbed")

    # fused adsorbate atoms
    for x in range(len(ads)):
        for y in range(x + 1, len(ads)):
            a, b = ads[x], ads[y]
            d = atoms.get_distance(a, b, mic=True)
            ref = covalent_radii[atoms[a].number] + covalent_radii[atoms[b].number]
            if d < FUSED_BELOW_RATIO * ref:
                problems.append(
                    f"{name}: {atoms[a].symbol}-{atoms[b].symbol} is "
                    f"{d:.2f} A, below {FUSED_BELOW_RATIO} x the {ref:.2f} A "
                    f"covalent sum, so the atoms have been pushed together")

    # damaged slab. Both tests use only distances between metal atoms, so
    # neither depends on how the cell is oriented.
    if len(metal) >= 4:
        dist = {i: sorted(atoms.get_distance(i, j, mic=True)
                          for j in metal if j != i) for i in metal}
        median = float(np.median([d[0] for d in dist.values()]))

        for i in metal:
            nearest = dist[i][0]
            off = nearest / median - 1.0
            if abs(off) > SURFACE_NN_TOLERANCE:
                problems.append(
                    f"{name}: {atoms[i].symbol} atom {i} has its nearest "
                    f"metal neighbour at {nearest:.2f} A, {off:+.0%} off the "
                    f"slab median {median:.2f} A, so the slab is damaged")

            coordination = sum(1 for d in dist[i] if d < 1.2 * median)
            if coordination < EXTRACTED_BELOW_COORDINATION:
                problems.append(
                    f"{name}: {atoms[i].symbol} atom {i} has only "
                    f"{coordination} metal neighbours, so it has been pulled "
                    f"out of the slab")

    return problems


@tool
def check_geometry() -> str:'''


OLD_GEOM = '''        for _, symbol, height in buried:
            problems.append(
                f"{name}: {symbol} sits {height:+.2f} A relative to the top "
                f"metal layer, so it is inside the slab rather than bonded "
                f"to the surface")'''

NEW_GEOM = '''        for _, symbol, height in buried:
            problems.append(
                f"{name}: {symbol} sits {height:+.2f} A relative to the top "
                f"metal layer, so it is inside the slab rather than bonded "
                f"to the surface")
        try:
            problems.extend(_structural_problems(read(str(f)), name))
        except Exception as exc:
            problems.append(
                f"{name}: structural check could not run: "
                f"{type(exc).__name__}: {exc}")'''


OLD_RELAX = '''    status = "converged" if converged else "DID NOT CONVERGE"
    return (f"Relaxed {structure} ({status}). Energy {energy:.4f} eV, "
            f"closest adsorbate-metal contact {contact:.2f} Å, "
            f"D3 {'on' if with_d3 else 'OFF'}.")'''

NEW_RELAX = '''    status = "converged" if converged else "DID NOT CONVERGE"
    message = (f"Relaxed {structure} ({status}). Energy {energy:.4f} eV, "
               f"closest adsorbate-metal contact {contact:.2f} Å, "
               f"D3 {'on' if with_d3 else 'OFF'}.")

    # Flag an unphysical structure at the step that produced it, not
    # several steps later at validation. check_geometry enforces the same
    # checks; this only makes them visible sooner.
    try:
        found = _structural_problems(atoms, structure)
    except Exception as exc:
        found = [f"structural check could not run: {type(exc).__name__}"]
    if found:
        message += (" WARNING, unphysical structure: " + "; ".join(found)
                    + ". Do not build on this structure.")
    return message'''


EDITS = [
    ("structural helpers", OLD_HELPER, NEW_HELPER),
    ("check_geometry wiring", OLD_GEOM, NEW_GEOM),
    ("relax_structure warning", OLD_RELAX, NEW_RELAX),
]


def main():
    check_only = "--check" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(args[0]) if args else Path("src/tools.py")

    if not path.exists():
        print(f"FAILED: {path} does not exist. Run from the repo root.")
        return 1

    text = path.read_text()
    if "_structural_problems" in text:
        print("Already patched. Nothing to do.")
        return 0

    for name, old, _ in EDITS:
        found = text.count(old)
        if found != 1:
            print(f"FAILED: the {name} anchor appears {found} times, "
                  "expected exactly 1. Nothing written.")
            return 1

    if check_only:
        print(f"All {len(EDITS)} anchors found. Patch would apply cleanly.")
        return 0

    backup = Path(str(path) + ".pre_structure_sanity")
    if not backup.exists():
        backup.write_text(text)

    for name, old, new in EDITS:
        text = text.replace(old, new, 1)
        print(f"  patched: {name}")
    path.write_text(text)
    print(f"Patched {path}. Backup: {backup.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
