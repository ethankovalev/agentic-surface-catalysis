"""
Remove the D3-off bias from two checks, and stop the subsurface check
refusing atoms bound to the lower terrace of a stepped slab.

    python patch_d3off_and_steps.py --check
    python patch_d3off_and_steps.py

Run from the repo root. Edits src/tools.py. Apply after
patch_dispersion_truth.py and patch_structure_sanity.py.

FOUND ON THE BLIND SWEEP OF 2026-09-22
--------------------------------------

1. check_dispersion_relevance penalised deliberate D3-off runs.
   passed = not (too_far and not with_d3). The same far contact passed
   with D3 on ("may genuinely be unbound") and failed with D3 off
   ("rerun with with_d3=true"). The agent validated N2/Ru(0001) terrace
   at a 4.70 A contact only because its record falsely said D3 was on.
   The blind run reached the same 4.70 A, recorded honestly, and failed
   on this check alone.

   Without dispersion, a physisorbed molecule is weakly held and floats
   further out. That is the model without D3 behaving as expected, not
   an error. And the scored barrier is referenced to the FREE molecule,
   so an initial state drifted outward is effectively the reference
   state already: N2's well depth was 9 meV. Once the gas-referenced
   barrier exists, a far contact under D3-off no longer bears on the
   number, so the check passes and says why. Before the gas reference
   exists it still fails, as before.

2. The initial-state desorption check had the same bias.
   H2/Pt(111), D3 off, was refused because its physisorbed molecule sat
   5.14 A out against a 5.0 A limit. Same reasoning: for a gas-referenced
   barrier the initial state may be anywhere between physisorbed and
   free. The initial-state limit is removed. The final-state and saddle
   limits stay: those are the ones that caught CH4/Ni(100)'s CH3
   fragment 5.48 A from any metal atom at the saddle, a real failure.

3. The subsurface check refused surface-bound atoms on stepped slabs.
   It compared each adsorbate atom's height against the highest metal
   atom. A stepped slab is a vicinal cell, a staircase, so an atom on
   the lower terrace sits below the step atoms. Both stepped reactions
   were refused as "inside the slab", at -0.38 and -1.00 A, but each
   flagged atom had exactly three metal neighbours within 2.3 A, nearest
   1.91 to 2.13 A: textbook threefold hollow binding on the surface.
   CH4/Ni(111) step, refused this way, had a barrier of 0.743 eV against
   a 0.80 eV reference.

   The replacement asks whether the atom is surrounded by metal or
   sitting on it. The vectors from the atom to every metal atom within
   3.2 A are averaged. For an atom on a surface, all the metal is on one
   side and the average points firmly away from it. For an atom inside
   the slab, metal surrounds it and the average nearly cancels. This
   does not depend on which way the cell is oriented.

   The case the original check was written for, an H atom 1.69 A inside
   the slab on H2/Cu(111), is surrounded by metal and is still caught.
"""

import sys
from pathlib import Path

TOOLS = Path("src/tools.py")

DISP_OLD = '''    too_far = contact > config.MAX_PHYSISORPTION_HEIGHT
    passed = not (too_far and not with_d3)

    if passed and too_far:
        detail = (f"contact {contact:.2f} Å is large but D3 was on: "
                  "may genuinely be unbound")
    elif passed:
        detail = f"contact {contact:.2f} Å, D3 {'on' if with_d3 else 'off'}"
    else:
        detail = (f"contact {contact:.2f} Å with D3 OFF: rerun the "
                  "relaxation with with_d3=true")'''

DISP_NEW = '''    too_far = contact > config.MAX_PHYSISORPTION_HEIGHT
    # For a barrier referenced to the free molecule, a physisorbed state
    # that has drifted outward is already close to the reference state,
    # so the drift does not bear on the scored number. Without D3 a
    # molecule is weakly held and floats further out; that is the model
    # behaving as expected, not an error to rerun away.
    well_depth = store.get("well_depth_eV")
    gas_referenced = store.get("barrier_gas_eV") is not None
    passed = not (too_far and not with_d3) or gas_referenced

    if too_far and not with_d3 and gas_referenced:
        detail = (f"contact {contact:.2f} Å with D3 off: the molecule is "
                  f"weakly held without dispersion, which does not affect a "
                  f"gas-referenced barrier (well depth {well_depth:.3f} eV)")
    elif passed and too_far:
        detail = (f"contact {contact:.2f} Å is large but D3 was on: "
                  "may genuinely be unbound")
    elif passed:
        detail = f"contact {contact:.2f} Å, D3 {'on' if with_d3 else 'off'}"
    else:
        detail = (f"contact {contact:.2f} Å with D3 off and no gas-referenced "
                  "barrier yet: either compute the gas-referenced barrier, or "
                  "relax again with with_d3=true")'''

DESORB_OLD = '''DESORBED_BEYOND_A = {"initial": 5.0, "final": 3.0, "saddle": 3.0}'''
DESORB_NEW = '''DESORBED_BEYOND_A = {"final": 3.0, "saddle": 3.0}
# No limit for the initial state. The scored barrier is referenced to the
# free molecule, so the initial state may sit anywhere from physisorbed
# to effectively free without affecting it. A 5.0 A limit here refused
# H2/Pt(111) with D3 off, where the molecule is weakly held by design.'''

SUB_OLD = '''    top_z = max(atoms.positions[m, 2] for m in metal)
    return [(i, atoms[i].symbol, float(atoms.positions[i, 2] - top_z))
            for i in ads
            if atoms.positions[i, 2] - top_z < MIN_ADSORBATE_HEIGHT]'''

SUB_NEW = '''    top_z = max(atoms.positions[m, 2] for m in metal)
    buried = []
    for i in ads:
        near = [m for m in metal
                if atoms.get_distance(i, m, mic=True) < SUBSURFACE_SHELL_A]
        if len(near) < SUBSURFACE_MIN_NEIGHBOURS:
            continue
        # Average of the vectors from the atom to its metal neighbours.
        # On a surface they all point one way; inside the slab they cancel.
        vectors = [atoms.get_distance(i, m, mic=True, vector=True)
                   for m in near]
        offset = float(np.linalg.norm(np.mean(vectors, axis=0)))
        if offset < SURROUNDED_BELOW_A:
            buried.append((i, atoms[i].symbol,
                           float(atoms.positions[i, 2] - top_z)))
    return buried'''

CONST_OLD = '''def _subsurface_adsorbates(atoms):'''
CONST_NEW = '''# Inside-the-slab test. A height comparison against the highest metal
# atom refused surface-bound atoms on the lower terrace of stepped slabs,
# which are vicinal cells shaped like a staircase. Instead: average the
# vectors from the atom to its metal neighbours within the shell. For an
# atom on a surface the metal is all on one side and the average is
# large; for an atom inside the slab the metal surrounds it and the
# average nearly cancels.
SUBSURFACE_SHELL_A = 3.2
SUBSURFACE_MIN_NEIGHBOURS = 4
SURROUNDED_BELOW_A = 0.6


def _subsurface_adsorbates(atoms):'''


EDITS = [
    ("dispersion relevance", DISP_OLD, DISP_NEW),
    ("initial desorption limit", DESORB_OLD, DESORB_NEW),
    ("subsurface criterion", SUB_OLD, SUB_NEW),
    ("subsurface constants", CONST_OLD, CONST_NEW),
]


def main():
    if not TOOLS.exists():
        print(f"FAILED: {TOOLS} not found. Run from the repo root.")
        return 1
    text = TOOLS.read_text()
    if "SURROUNDED_BELOW_A" in text:
        print("Already patched. Nothing to do.")
        return 0
    problems = [f"{name}: anchor found {text.count(old)} times"
                for name, old, _ in EDITS if text.count(old) != 1]
    if problems:
        print("FAILED, nothing written:")
        for p in problems:
            print(f"  {p}")
        return 1
    if "--check" in sys.argv:
        print(f"All {len(EDITS)} anchors found. Patch would apply cleanly.")
        return 0
    backup = Path(str(TOOLS) + ".pre_d3off_and_steps")
    if not backup.exists():
        backup.write_text(text)
    for name, old, new in EDITS:
        text = text.replace(old, new, 1)
    TOOLS.write_text(text)
    print("Patched src/tools.py: dispersion relevance, initial desorption "
          "limit, subsurface test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
