"""
Make both connectivity tests measure the right bond, and judge where a
relaxation landed against the intact molecule rather than against
wherever the final state happens to sit.

    python patch_connectivity_intact.py --check
    python patch_connectivity_intact.py

Run from the repo root. Edits src/tools.py. Apply after
patch_irc_constraints.py and patch_d3off_and_steps.py.

FOUND ON THE BLIND SWEEP OF 2026-09-22
--------------------------------------
H2/Cu(111): refine_saddle_robust confirmed a first-order saddle, one mode
at 72.6 meV, and its IRC said the saddle connects. check_saddle_connects
then said it does not, and compute_zpe_correction refused on that basis,
so the run failed validation with a correct saddle in hand.

Following the mode, one side relaxed to 0.75 A (intact H2) and the other
to 2.02 A. A 2.02 A H-H distance is not a molecule: it is two H atoms on
adjacent sites, immediately after dissociation. The final endpoint sits
at 4.42 A, the same atoms diffused apart onto separate metal atoms. That
is the two-stage mechanism in Chorkendorff and Niemantsverdriet 6.5.3.2:
dissociation first reaches adjacent sites, then a small diffusion barrier
separates the fragments. The saddle belongs to the dissociation step.

DEFECT 1: THE MIDPOINT RULE MOVES WITH THE FINAL STATE
------------------------------------------------------
check_saddle_connects labelled each side by which side of the midpoint
between the initial and final separations it fell. Its own comment
records this exact case from an earlier session: the IRC relaxed to
2.03 A, the final state was 3.89 A, the midpoint was 2.3 A, and the
saddle was correctly accepted.

patch_site_selection.py then moved the final state to 4.42 A, as it
should, because the fragments now sit on separate metal atoms. That
pushed the midpoint to 2.59 A, past 2.02 A. A correct fix silently broke
a previously fixed case.

The rule now asks what actually matters: did this side fall back to the
intact molecule, or break it? Recombined means within 1.35 times the bond
measured in the initial state. Dissociated means beyond 1.8 times it.
Anything between is reported as neither, and does not count as
connecting. The comment's own reasoning, "2.7 times the 0.74 A bond",
is now the rule.

DEFECT 2: THE PAIR WAS THE FIRST TWO ATOMS BY INDEX
---------------------------------------------------
pair() measured ads[0] to ads[1]. For H2 that is the breaking bond. For
CH4 it is the carbon and a hydrogen that stays attached, the same
spectator bug check_structures.py had weeks ago. On every CH4 reaction
both endpoints read about 1.09 A, and the tool refused on "the endpoints
differ by only 0.02 A": the README lists that as an open limitation on
stepped slabs. It was this. The pair now comes from the cutoff-free
breaking bond finder, measured on the saddle and applied to every
structure by index.

DEFECT 3: THE INTACT BOND WAS A COVALENT RADIUS SUM
----------------------------------------------------
The two connectivity helpers inside refine_saddle_robust judged
recombination against the covalent radius sum. For H-H that is 0.62 A,
while a real H2 bond is 0.74 A, so the 1.35 limit allowed barely 0.1 A
of stretch. H2/Ru(0001) was rejected because its IRC's molecular end
stopped at 0.94 A. Both helpers now use the bond measured in the relaxed
initial state, falling back to the covalent sum only if that state is
missing or is itself not intact.
"""

import sys
from pathlib import Path

TOOLS = Path("src/tools.py")

PAIR_OLD = '''    def pair(atoms):
        return float(atoms.get_distance(ads[0], ads[1], mic=True))'''
PAIR_NEW = '''    # The breaking bond, found on the saddle and applied to every
    # structure by index. ads[0] to ads[1] was the carbon and a spectator
    # hydrogen for every CH4 reaction.
    found = _breaking_bond(saddle)
    if found is None:
        return "FAILED: the breaking bond could not be identified at the saddle."
    anchor, terminal, _ = found

    def pair(atoms):
        return float(atoms.get_distance(anchor, terminal, mic=True))'''

CLASSIFY_OLD = '''        midpoint = 0.5 * (d_initial + d_final)
        landed[label] = (d, "initial" if d < midpoint else "final")'''
CLASSIFY_NEW = '''        #
        # A midpoint between the two endpoint separations moves with the
        # final state. Once fragments were kept on separate metal atoms,
        # the H2/Cu(111) final state moved to 4.42 A, the midpoint to
        # 2.59 A, and the correct 2.02 A adjacent-site product fell on the
        # "initial" side. Judge against the intact bond instead.
        recombined_below = 1.35 * d_initial
        dissociated_above = 1.8 * d_initial
        if d < recombined_below:
            where = "initial"
        elif d > dissociated_above:
            where = "final"
        else:
            where = "neither"
        landed[label] = (d, where)'''

STORE_OLD = '''    store.put("saddle_connectivity", {
        "connects": fwd_where != bwd_where,'''
STORE_NEW = '''    connects = {fwd_where, bwd_where} == {"initial", "final"}
    store.put("saddle_connectivity", {
        "connects": connects,'''

MSG_OLD = '''    if fwd_where == bwd_where:
        return (f"SADDLE DOES NOT CONNECT THE ENDPOINTS: following the "
                f"imaginary mode both ways falls to the {fwd_where} state "'''
MSG_NEW = '''    if not connects:
        return (f"SADDLE DOES NOT CONNECT THE ENDPOINTS: following the "
                f"imaginary mode lands forward on {fwd_where} and backward "
                f"on {bwd_where}, where one must recombine the molecule and "
                f"the other break it "'''

INTACT_BLOCK = '''    anchor, terminal, r_saddle = found
    intact = (covalent_radii[atoms[anchor].number]
              + covalent_radii[atoms[terminal].number])'''
INTACT_NEW = '''    anchor, terminal, r_saddle = found
    intact = _intact_bond_length(atoms, anchor, terminal)'''

HELPER_OLD = '''def _connectivity_by_bond(atoms, model_key, with_d3):'''
HELPER_NEW = '''def _intact_bond_length(atoms, anchor, terminal):
    """The breaking bond as it is in the relaxed initial state.

    The covalent radius sum is a poor stand-in for H-H: 0.62 A against a
    real 0.74 A bond, which left almost no room for a molecular end to be
    recognised as recombined. Falls back to the covalent sum only when
    initial.traj is missing, or when that state is itself not intact.
    """
    covalent = (covalent_radii[atoms[anchor].number]
                + covalent_radii[atoms[terminal].number])
    path = Path(_path("initial.traj"))
    if not path.exists():
        return covalent
    try:
        measured = float(read(str(path)).get_distance(anchor, terminal, mic=True))
    except Exception:
        return covalent
    if measured > 1.5 * covalent:
        return covalent
    return measured


def _connectivity_by_bond(atoms, model_key, with_d3):'''


def main():
    if not TOOLS.exists():
        print(f"FAILED: {TOOLS} not found. Run from the repo root.")
        return 1
    text = TOOLS.read_text()
    if "def _intact_bond_length" in text:
        print("Already patched. Nothing to do.")
        return 0

    checks = [("check_saddle_connects pair", PAIR_OLD, 1),
              ("check_saddle_connects classification", CLASSIFY_OLD, 1),
              ("check_saddle_connects record", STORE_OLD, 1),
              ("check_saddle_connects message", MSG_OLD, 1),
              ("intact bond in the two connectivity helpers", INTACT_BLOCK, 2),
              ("intact bond helper location", HELPER_OLD, 1)]
    problems = [f"{name}: found {text.count(old)}, expected {n}"
                for name, old, n in checks if text.count(old) != n]
    if problems:
        print("FAILED, nothing written:")
        for p in problems:
            print(f"  {p}")
        return 1
    if "--check" in sys.argv:
        print("All anchors found. Patch would apply cleanly.")
        return 0

    backup = Path(str(TOOLS) + ".pre_connectivity_intact")
    if not backup.exists():
        backup.write_text(text)
    text = text.replace(PAIR_OLD, PAIR_NEW, 1)
    text = text.replace(CLASSIFY_OLD, CLASSIFY_NEW, 1)
    text = text.replace(STORE_OLD, STORE_NEW, 1)
    text = text.replace(MSG_OLD, MSG_NEW, 1)
    text = text.replace(INTACT_BLOCK, INTACT_NEW)
    text = text.replace(HELPER_OLD, HELPER_NEW, 1)
    TOOLS.write_text(text)
    print("Patched check_saddle_connects (pair and classification) and both "
          "connectivity helpers (measured intact bond).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
