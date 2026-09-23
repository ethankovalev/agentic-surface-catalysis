"""
Fix scripts/check_structures.py: two checks that predate recent commits.

Run:  python patch_check_structures.py --check scripts/check_structures.py
      python patch_check_structures.py scripts/check_structures.py

WHAT WAS WRONG
--------------
1. "fragments separated" compared against a flat MIN_FRAGMENT_SEPARATION =
   3.0 A. That constant predates commit 41ae8ca ("Target the nearest real
   adjacent site... 2.0x recombination-safe margin"), which deliberately
   snaps dissociated fragments onto the nearest genuine adjacent site
   rather than an arbitrary continuous distance - NOTES.md records the
   real H2/Cu(111) product at 2.03 A. Every reaction now legitimately
   produces separations in the 1.5-2.7 A range, so the flat 3.0 A check
   fails all ten by construction and has been failing them since that
   commit landed, independent of anything patched today.

2. "endpoints distinct" measured the distance between ads_i[0] and
   ads_i[1] - literally the first two adsorbate atoms by index - not the
   bond that actually broke. For a diatomic (H2, N2) those are the same
   two atoms, so this was invisible. For CH4 ([C, H, H, H, H] in ASE's
   ordering), atoms 0 and 1 are the carbon and a hydrogen that STAYS in
   the CH3 fragment; that bond is untouched by dissociation and reads the
   same length in initial.traj and final.traj by definition. Before the
   orientation fix, build_dissociated_endpoint happened to also pick
   (0, 1) as the breaking pair, so the two bugs cancelled by coincidence.
   The orientation fix makes the breaking pair depend on which hydrogen
   points at the surface, which is correct chemistry, and exposed that
   this check was never looking at the right bond.

THE FIX
-------
Both checks now call breaking_bond() (from src/tools.py, added alongside
the CH4 orientation fix) to identify the actual bond that dissociates,
instead of guessing from fixed indices or a hand-picked distance:

  - "fragments separated" requires the anchor-to-anchor distance to be at
    least RECOMBINATION_MARGIN (2.0x, matching the constant already
    enforced inside build_dissociated_endpoint) times the INTACT bond
    length of the specific molecule being tested, not a single number
    borrowed across nine different bond types.
  - "endpoints distinct" measures the identified breaking bond, in both
    initial.traj and final.traj, not a hardcoded (0, 1).

This is the same fix pattern as patch_ch4_orientation.py: the geometry
logic in src/tools.py was already correct, and the checker just hadn't
been told what changed.
"""

import sys
from pathlib import Path


OLD_CONSTANTS = '''MIN_FRAGMENT_SEPARATION = 3.0
BOND_TOLERANCE = 0.45
STEP_PROXIMITY = 3.2
MIN_INITIAL_CLEARANCE = 1.8'''

NEW_CONSTANTS = '''# Matches the invariant build_dissociated_endpoint already enforces via
# min_safe_separation in src/tools.py: a fragment closer than this to its
# partner could relax straight back into the intact molecule. A single
# flat distance across nine different bond types (H-H, N-N, C-H on three
# different metals) does not make physical sense - a real adjacent-site
# product legitimately sits anywhere from 1.5 to 2.7 A away depending on
# the lattice, which is why the old MIN_FRAGMENT_SEPARATION = 3.0 A failed
# every reaction regardless of whether the geometry was actually correct.
RECOMBINATION_MARGIN = 2.0
BOND_TOLERANCE = 0.45
STEP_PROXIMITY = 3.2
MIN_INITIAL_CLEARANCE = 1.8'''


OLD_IMPORT = '''from src.tools import (  # noqa: E402
    build_dissociated_endpoint,
    build_slab,
    build_stepped_slab,
    place_adsorbate,
)'''

NEW_IMPORT = '''from src.tools import (  # noqa: E402
    breaking_bond,
    build_dissociated_endpoint,
    build_slab,
    build_stepped_slab,
    place_adsorbate,
)'''


OLD_TAIL = '''    a0, a1 = _anchor(final, ff[0]), _anchor(final, ff[1])
    sep = final.get_distance(a0, a1, mic=True)
    add("fragments separated", sep >= MIN_FRAGMENT_SEPARATION,
        f"{sep:.2f} A between anchors, minimum {MIN_FRAGMENT_SEPARATION}")

    for n, anc in enumerate((a0, a1)):
        ideal = r_metal + covalent_radii[final[anc].number]
        nearest = min(final.get_distance(anc, m, mic=True) for m in metal_f)
        ok = abs(nearest - ideal) <= BOND_TOLERANCE
        add(f"fragment {n + 1} bonded",
            ok, f"{final[anc].symbol} at {nearest:.2f} A, "
                f"covalent {ideal:.2f} A, height {final.positions[anc, 2] - top_z:.2f} A")

    if stepped:
        edge = slab_rec.get("step_edge_atoms") or []
        if edge:
            near = [min(final.get_distance(anc, e, mic=True) for e in edge)
                    for anc in (a0, a1)]
            add("fragments at the step", min(near) <= STEP_PROXIMITY,
                f"closest fragment {min(near):.2f} A from an edge atom")

    bond_i = initial.get_distance(ads_i[0], ads_i[1], mic=True)
    bond_f = final.get_distance(ads_f[0], ads_f[1], mic=True)
    add("endpoints distinct", abs(bond_f - bond_i) > 1.0,
        f"adsorbate pair {bond_i:.2f} A -> {bond_f:.2f} A")

'''

NEW_TAIL = '''    # Identify the bond that actually dissociates, rather than guessing
    # from fixed indices (wrong for anything bigger than a diatomic) or a
    # single borrowed distance (wrong across different bond types). Uses
    # the initial, still-intact geometry, so this is the same pair
    # build_dissociated_endpoint itself chose to break.
    try:
        break_a, break_b, intact_bond = breaking_bond(initial, ads_i)
    except ValueError as exc:
        add("fragments separated", False,
            f"could not identify the breaking bond: {exc}")
        return results

    a0, a1 = _anchor(final, ff[0]), _anchor(final, ff[1])
    sep = final.get_distance(a0, a1, mic=True)
    required = RECOMBINATION_MARGIN * intact_bond
    add("fragments separated", sep >= required,
        f"{sep:.2f} A between anchors, minimum {required:.2f} A "
        f"({RECOMBINATION_MARGIN:.1f}x the intact "
        f"{initial[break_a].symbol}-{initial[break_b].symbol} bond, "
        f"{intact_bond:.2f} A)")

    for n, anc in enumerate((a0, a1)):
        ideal = r_metal + covalent_radii[final[anc].number]
        nearest = min(final.get_distance(anc, m, mic=True) for m in metal_f)
        ok = abs(nearest - ideal) <= BOND_TOLERANCE
        add(f"fragment {n + 1} bonded",
            ok, f"{final[anc].symbol} at {nearest:.2f} A, "
                f"covalent {ideal:.2f} A, height {final.positions[anc, 2] - top_z:.2f} A")

    if stepped:
        edge = slab_rec.get("step_edge_atoms") or []
        if edge:
            near = [min(final.get_distance(anc, e, mic=True) for e in edge)
                    for anc in (a0, a1)]
            add("fragments at the step", min(near) <= STEP_PROXIMITY,
                f"closest fragment {min(near):.2f} A from an edge atom")

    # Same pair, both trajectories. ads_i and ads_f are the same atoms in
    # the same order - build_dissociated_endpoint only moves positions, it
    # never reorders or relabels - so break_a/break_b index the breaking
    # bond correctly in both files.
    #
    # Pass criterion is the same `required` used for "fragments separated",
    # not an independent flat +1.0 A. A hardcoded absolute change does not
    # scale with bond length: H2's 0.74 A bond reaching a real, physically
    # correct 1.56 A adjacent-site separation is only a 0.82 A change and
    # failed here while H2/Cu(100) at 1.80 A (0.74 A bond, same chemistry)
    # passed at 1.06 A purely by chance of which site the search landed on.
    # In this benchmark break_a/break_b always coincide with the two
    # fragment anchors used above (a single molecule splits into exactly
    # two groups, so each group's heaviest-atom anchor IS the breaking
    # pair), so this check and "fragments separated" measure the same
    # physical distance before any relaxation has run - they should not be
    # allowed to disagree because one uses a scaled criterion and the
    # other does not.
    bond_i = initial.get_distance(break_a, break_b, mic=True)
    bond_f = final.get_distance(break_a, break_b, mic=True)
    add("endpoints distinct", bond_f >= required,
        f"breaking bond {initial[break_a].symbol}-{initial[break_b].symbol} "
        f"{bond_i:.2f} A -> {bond_f:.2f} A, minimum {required:.2f} A")

'''


def apply(path: Path, check_only: bool) -> int:
    text = path.read_text()
    problems = []

    for name, old in (("constants", OLD_CONSTANTS),
                      ("tools import", OLD_IMPORT),
                      ("check tail", OLD_TAIL)):
        if old not in text and (
            (name == "constants" and NEW_CONSTANTS not in text) or
            (name == "tools import" and NEW_IMPORT not in text) or
            (name == "check tail" and NEW_TAIL not in text)
        ):
            problems.append(f"  - could not find the original {name} block")

    if problems:
        print("PATCH DID NOT APPLY:")
        print("\n".join(problems))
        print("The file has moved on since this patch was written; apply by hand.")
        return 1

    if NEW_CONSTANTS in text and NEW_IMPORT in text and NEW_TAIL in text:
        print("Already patched. Nothing to do.")
        return 0

    text = text.replace(OLD_CONSTANTS, NEW_CONSTANTS, 1)
    text = text.replace(OLD_IMPORT, NEW_IMPORT, 1)
    text = text.replace(OLD_TAIL, NEW_TAIL, 1)

    if check_only:
        print("All three blocks found. Patch would apply cleanly.")
        return 0

    backup = path.with_suffix(path.suffix + ".pre_check_fix")
    if not backup.exists():
        backup.write_text(path.read_text())
    path.write_text(text)
    print(f"Patched {path}. Original saved to {backup.name}.")
    print("Now run: python scripts/check_structures.py")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    target = Path(args[0]) if args else Path("scripts/check_structures.py")
    if not target.exists():
        raise SystemExit(f"{target} does not exist. Pass the path to check_structures.py.")
    raise SystemExit(apply(target, "--check" in sys.argv))
