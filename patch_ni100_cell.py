"""
Give CH4_Ni100 a 4x4 cell, so its dissociated state can exist at all.

    python patch_ni100_cell.py --check
    python patch_ni100_cell.py

Run from the repo root. Edits src/benchmark.py and
scripts/check_structures.py.

WHY
---
patch_site_selection.py keeps the two dissociated fragments on separate
surface metal atoms, because adjacent fragments sharing a metal atom are
the repulsive arrangement rather than the product minimum (Chorkendorff
and Niemantsverdriet 6.5.3.2).

Ni(100) at the default 3x3 has nine fourfold hollows, each touching four
of the nine surface atoms, and no pair of them shares zero metal atoms:
0 of 36. The builder therefore refuses, correctly, and
check_structures.py - step 1 of the workflow in CLAUDE.md - has failed on
CH4_Ni100 since that patch went in. Confirmed on the pod: it is the only
failing reaction. At 4x4, 56 of 120 site pairs are disjoint and the
endpoint builds with the fragments 5.01 A apart.

HOW
---
Cell size is problem specification, like site_type: it says what
calculation to set up, not what the answer is. So it goes in the same
place site_type does, the SBH10 spec, and reaches the agent the same way
site_type does, as an instruction in run_one's task prompt. The agent
still never sees the reference barrier.

Only reactions that need it carry a "cell" entry. Everything else keeps
building at the tool default, so no existing result changes.

WHAT THIS CHANGES SCIENTIFICALLY
---------------------------------
Coverage for CH4_Ni100 drops from 1/9 to 1/16 of a monolayer in the
autonomous track, and it is the only reaction at that size. The cell
sensitivity test on H2_Cu111 found a 0.034 eV gap between 3x3x4 and
2x2x6, but that is one reaction and a different pair of sizes, so it
does not establish that 3x3 and 4x4 agree for CH4 on Ni(100). The lower
coverage is closer to the low-coverage limit the experimental reference
describes, so it should be no worse; it is not tested.
"""

import sys
from pathlib import Path


BENCH = Path("src/benchmark.py")
CHECK = Path("scripts/check_structures.py")


BENCH_SPEC_OLD = '''    "CH4_Ni100": {
        "molecule": "CH4", "metal": "Ni", "facet": "100",
        "site_type": "terrace",'''

BENCH_SPEC_NEW = '''    "CH4_Ni100": {
        "molecule": "CH4", "metal": "Ni", "facet": "100",
        "site_type": "terrace",
        # At 3x3, no two of Ni(100)'s fourfold hollows avoid sharing a
        # surface Ni atom, so the dissociated fragments cannot sit on
        # separate metal atoms and the endpoint builder refuses. 4x4 has
        # 56 disjoint pairs of 120. See patch_ni100_cell.py.
        "cell": [4, 4],'''


BENCH_TASK_OLD = '''    task = (
        f"Compute the dissociation barrier for {spec['molecule']} on "
        f"{spec['metal']}({spec['facet']}).'''

BENCH_TASK_NEW = '''    # Cell size is problem specification, like site_type: it says what to
    # set up, not what the answer is.
    cell = spec.get("cell")
    if cell:
        site_instruction += (
            f" Build the slab with nx={cell[0]} and ny={cell[1]}. A smaller "
            f"cell cannot hold the dissociated fragments on separate metal "
            f"atoms for this surface, and the endpoint builder will refuse.")

    task = (
        f"Compute the dissociation barrier for {spec['molecule']} on "
        f"{spec['metal']}({spec['facet']}).'''


CHECK_OLD = '''        msg = build_slab.invoke(
            {"metal": spec["metal"], "facet": spec["facet"]})'''

CHECK_NEW = '''        args = {"metal": spec["metal"], "facet": spec["facet"]}
        if spec.get("cell"):
            args["nx"], args["ny"] = spec["cell"]
        msg = build_slab.invoke(args)'''


EDITS = [
    (BENCH, "CH4_Ni100 spec", BENCH_SPEC_OLD, BENCH_SPEC_NEW),
    (BENCH, "run_one task prompt", BENCH_TASK_OLD, BENCH_TASK_NEW),
    (CHECK, "check_structures slab build", CHECK_OLD, CHECK_NEW),
]


def main():
    check_only = "--check" in sys.argv

    for path in (BENCH, CHECK):
        if not path.exists():
            print(f"FAILED: {path} does not exist. Run from the repo root.")
            return 1

    if '"cell": [4, 4]' in BENCH.read_text():
        print("Already patched. Nothing to do.")
        return 0

    for path, name, old, _ in EDITS:
        found = path.read_text().count(old)
        if found != 1:
            print(f"FAILED: the {name} anchor appears {found} times in "
                  f"{path}, expected exactly 1. Nothing written.")
            return 1

    if check_only:
        print(f"All {len(EDITS)} anchors found. Patch would apply cleanly.")
        return 0

    for path in (BENCH, CHECK):
        backup = Path(str(path) + ".pre_ni100_cell")
        if not backup.exists():
            backup.write_text(path.read_text())

    for path, name, old, new in EDITS:
        path.write_text(path.read_text().replace(old, new, 1))
        print(f"  patched: {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
