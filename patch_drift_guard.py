"""
Fix two defects in refine_saddle_robust that let a collapsed geometry
pass as a transition state.

    python patch_drift_guard.py --check src/tools.py
    python patch_drift_guard.py src/tools.py

FOUND 2026-09-19, ON REAL DATA
-------------------------------
CH4_Ni100 and CH4_Ni111_step both reported "first order saddle: True"
while sitting on a near-intact molecule:

    CH4_Ni100       final C-H 1.125 A, seed 1.91 A, mode 41 meV
    CH4_Ni111_step  final C-H 1.100 A,              mode 47 meV

An intact C-H is 1.07 A. Both had fallen all the way back to the
reactant, which is the documented failure for these two reactions, and
both were accepted.

DEFECT 1: THE DRIFT GUARD MEASURED A SPECTATOR BOND
-----------------------------------------------------
_breaking_bond_length called src.tools.breaking_bond, which finds
neighbours with _bonded_to, an intact-molecule cutoff. At the seed
geometry the breaking C-H is stretched to 1.91 A and falls outside that
cutoff entirely, so the function returned an untouched spectator C-H
instead. Comparing a spectator before against a spectator after gives
+0.012 A, which looks reassuring and means nothing.

run_seeded.py's own breaking_bond already documents this exact trap:
the 2.2x reach exists because "an intact-molecule cutoff excludes
exactly the bond being looked for and silently returns a spectator".
This patch uses that reach.

DEFECT 2: THE DRIFT WAS COMPUTED AND NEVER USED
-------------------------------------------------
refine_saddle_robust accepted on imaginary-mode count alone. A collapsed
geometry with one soft mode passes that test. The drift was recorded in
the attempt log and never consulted, so the decision was made without
the one number that would have caught this.

THE GUARD
---------
An attempt is a transition state only if it has exactly one imaginary
mode AND its breaking bond is still stretched. "Still stretched" is
r >= COLLAPSE_RATIO x (covalent radius sum), a physical statement rather
than a tuned constant:

    C-H  covalent sum 1.07 A   collapsed at 1.125 A  ratio 1.05  REJECT
    C-H  covalent sum 1.07 A   collapsed at 1.100 A  ratio 1.03  REJECT
    H-H  covalent sum 0.62 A   genuine saddle ~1.0 A ratio 1.6   keep

plus a plain distance guard: more than MAX_DRIFT_A from the seed means
the optimiser has left the region it was asked about, whatever the mode
count says.

A rejected attempt is treated as basin collapse, so the existing
recovery (smaller trust radius) runs, rather than the run stopping.
"""

import sys
from pathlib import Path


OLD = '''def _breaking_bond_length(atoms):
    """Length of the bond being broken, or None if it cannot be found."""
    try:
        tags = atoms.get_tags()
        ads = [i for i in range(len(atoms)) if tags[i] == 2]
        if len(ads) < 2:
            return None
        a, b, _ = breaking_bond(atoms, ads)
        return float(atoms.get_distance(a, b, mic=True))
    except Exception:
        return None'''

NEW = '''# A saddle whose breaking bond has relaxed back to this multiple of the
# covalent radius sum has fallen into the reactant basin, whatever its
# mode count says. 1.15 sits above the two measured collapses (ratios
# 1.05 and 1.03 for C-H) and well below a genuine saddle (about 1.6 for
# H-H on Cu(100)).
COLLAPSE_RATIO = 1.15

# And a bond this far from where the search started has left the region
# it was asked about, in either direction.
MAX_DRIFT_A = 0.5


def _breaking_bond(atoms):
    """(anchor, terminal, length) of the bond being broken, or None.

    Uses a 2.2x reach rather than an intact-molecule cutoff, for the
    reason run_seeded.py's own breaking_bond documents: at a transition
    state the breaking bond is already stretched well past equilibrium -
    CH4/Ni(100) sits at 1.91 A against a 1.09 A normal C-H - so a normal
    cutoff excludes exactly the bond being looked for and silently
    returns a spectator. src.tools.breaking_bond uses _bonded_to, which
    is that normal cutoff, and using it here made the drift guard
    compare one untouched C-H against another and report +0.012 A while
    the real breaking bond had collapsed by 0.79 A.
    """
    try:
        tags = atoms.get_tags()
        ads = [i for i in range(len(atoms)) if tags[i] == 2]
        if len(ads) < 2:
            return None
        anchor = max(ads, key=lambda k: covalent_radii[atoms[k].number])
        others = [k for k in ads if k != anchor]
        if not others:
            return None
        # The FARTHEST adsorbate atom from the anchor, with no distance
        # cutoff at all. At a dissociation saddle the breaking bond is by
        # construction the longest anchor-to-fragment distance, so a
        # cutoff can only ever exclude the right answer. run_seeded.py
        # uses a 2.2x reach for this, which is wide enough for a 1.91 A
        # stretched C-H but still silently returns a 1.09 A spectator
        # once the bond passes 2.35 A - the same failure this patch
        # exists to fix, at the other end of the range.
        terminal = max(
            others, key=lambda k: atoms.get_distance(anchor, k, mic=True))
        return (anchor, terminal,
                float(atoms.get_distance(anchor, terminal, mic=True)))
    except Exception:
        return None


def _breaking_bond_length(atoms):
    """Length of the bond being broken, or None if it cannot be found."""
    found = _breaking_bond(atoms)
    return None if found is None else found[2]


def _geometry_is_still_a_saddle(atoms, r_seed):
    """Has the breaking bond stayed stretched, or fallen back?

    Returns (ok, reason). One imaginary mode is necessary but not
    sufficient: a geometry that has relaxed back to a near-intact
    molecule can still show a single soft mode, and on 2026-09-19 two
    reactions did exactly that and were accepted as transition states.
    """
    found = _breaking_bond(atoms)
    if found is None:
        return True, "breaking bond could not be identified, guard skipped"

    anchor, terminal, r_now = found
    intact = (covalent_radii[atoms[anchor].number]
              + covalent_radii[atoms[terminal].number])
    ratio = r_now / intact if intact > 0 else float("inf")

    if r_seed is not None and r_now - r_seed > MAX_DRIFT_A:
        return False, (
            f"breaking bond stretched to {r_now:.3f} A, {r_now - r_seed:+.3f} "
            f"A past the {r_seed:.3f} A starting geometry: the fragments have "
            "separated, so this is the product side rather than a saddle")

    if ratio < COLLAPSE_RATIO:
        return False, (
            f"breaking bond relaxed to {r_now:.3f} A, only {ratio:.2f}x the "
            f"{intact:.2f} A covalent sum: this is a near-intact molecule, "
            "not a transition state")

    if r_seed is not None and r_seed - r_now > MAX_DRIFT_A:
        return False, (
            f"breaking bond fell {r_now - r_seed:+.3f} A from the "
            f"{r_seed:.3f} A starting geometry: the optimiser has left the "
            "region it was asked about")

    return True, ""'''


CALL_OLD = '''        if len(imag) == 1:
            break

        if attempt == max_attempts:
            break'''

CALL_NEW = '''        geometry_ok, geometry_reason = _geometry_is_still_a_saddle(
            atoms, r_peak)
        attempts[-1]["geometry_ok"] = geometry_ok
        if geometry_reason:
            attempts[-1]["geometry_reason"] = geometry_reason

        if len(imag) == 1 and geometry_ok:
            break

        if attempt == max_attempts:
            break'''


STRATEGY_OLD = '''        # choose the next strategy from how this attempt failed
        if len(imag) == 0:'''

STRATEGY_NEW = '''        # choose the next strategy from how this attempt failed. A
        # geometry that collapsed is basin collapse even when the mode
        # count looks right, so it takes the same recovery.
        if len(imag) == 1 and not geometry_ok:
            delta0 = 0.005
            displacement = None
            strategy = ("retry with a smaller trust radius after the "
                        "geometry collapsed despite a single mode")
        elif len(imag) == 0:'''


FINAL_OLD = '''    final = attempts[-1]
    imag = final["imaginary_modes_meV"]
    first_order = len(imag) == 1'''

FINAL_NEW = '''    final = attempts[-1]
    imag = final["imaginary_modes_meV"]
    first_order = len(imag) == 1 and final.get("geometry_ok", True)'''


HEAD_OLD = '''    if first_order:
        head = (f"Refined to a first-order saddle after {len(attempts)} "'''

HEAD_NEW = '''    if len(imag) == 1 and not final.get("geometry_ok", True):
        head = (f"NOT A TRANSITION STATE after {len(attempts)} attempt(s): "
                f"one imaginary mode at {imag[0]:.1f} meV, but "
                f"{final.get('geometry_reason', 'the geometry has collapsed')}"
                ". A single mode is necessary but not sufficient.")
    elif first_order:
        head = (f"Refined to a first-order saddle after {len(attempts)} "'''


EDITS = [
    ("_breaking_bond_length + guards", OLD, NEW),
    ("accept decision", CALL_OLD, CALL_NEW),
    ("strategy selection", STRATEGY_OLD, STRATEGY_NEW),
    ("first_order definition", FINAL_OLD, FINAL_NEW),
    ("report head", HEAD_OLD, HEAD_NEW),
]


def main():
    check_only = "--check" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(args[0]) if args else Path("src/tools.py")

    if not path.exists():
        print(f"FAILED: {path} does not exist. Run from the repo root.")
        return 1

    text = path.read_text()
    if "_geometry_is_still_a_saddle" in text:
        print("Already patched. Nothing to do.")
        return 0
    if "refine_saddle_robust" not in text:
        print("FAILED: refine_saddle_robust not found. Apply "
              "patch_robust_saddle.py first.")
        return 1

    for name, old, _ in EDITS:
        found = text.count(old)
        if found != 1:
            print(f"FAILED: the {name} anchor appears {found} times, "
                  "expected exactly 1. Nothing written.")
            return 1

    if check_only:
        print(f"All {len(EDITS)} anchors found. Patch would apply cleanly.")
        return 0

    backup = Path(str(path) + ".pre_drift_guard")
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
