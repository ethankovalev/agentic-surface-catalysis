"""
THE 7 eV CH4 BUG. Root cause, fix, and proof.

Run this file to apply the patch:  python patch_ch4_orientation.py path/to/src/tools.py
Run with --check to verify without writing.

WHAT IS WRONG
-------------
Two functions in src/tools.py disagree about which bond breaks, and both
are wrong for a polyatomic.

1. `_orient_for_dissociation` returns anything that is not a diatomic
   unchanged. ASE's g2 CH4 comes back as

       C  ( 0.000,  0.000,  0.000)
       H  ( 0.629,  0.629,  0.629)     <- pointing UP
       H  (-0.629, -0.629,  0.629)     <- pointing UP
       H  ( 0.629, -0.629, -0.629)
       H  (-0.629,  0.629, -0.629)

   so the molecule is placed edge-down (two H toward the metal, two away).
   The reactive geometry for CH4 on a transition metal terrace is one C-H
   pointing at the surface.

2. `build_dissociated_endpoint` picks the bond to break as the LONGEST
   bonded pair, with a strict `d > best`. All four C-H bonds are 1.0897 A,
   so the strict comparison never fires again after the first, and the pair
   selected is deterministically (atom 0, atom 1) - which is the C and the
   FIRST H in the list, at z = +0.629. The H on the far side of the carbon
   from the metal.

   Verified, not assumed: replaying the exact selection loop on
   ase.build.molecule("CH4") returns pair (0, 1), d = 1.0897, terminal atom
   at z = +0.629 while the lowest atom in the molecule sits at z = -0.629.

CONSEQUENCE
-----------
The final state puts that H at a hollow site 2.7 A away. The band therefore
has to take a hydrogen from the top of the molecule, through or around the
carbon, down to the surface. Along that path the C-H bond is broken while
the hydrogen is still well clear of any metal atom, so the model is asked
for something very close to gas-phase homolysis of methane - about 4.5 eV -
plus the strain of dragging it past the CH3 umbrella, plus a path that is
not the reaction coordinate so nothing along it relaxes usefully.

That is the 7 eV. It is not a model error, a dispersion error, or a
convergence error. UMA is answering the question it was asked.

The real transition state has CH3 essentially where the methane was, over a
top site, with the breaking C-H stretched to roughly 1.5-1.6 A and the
hydrogen already interacting with a metal atom, which is what pays for the
bond being broken. The barrier is 0.8-1.0 eV because the surface does most
of the work. If the hydrogen never reaches the surface on the way, the
surface does none of it.

THE FIX
-------
Make the two functions agree, and make both of them aware of which way is
down.

  - `_orient_for_dissociation` gains a polyatomic branch: find the anchor
    (heaviest atom), find the terminal atoms bonded to it, and rotate so
    that one of those bonds points along -z, at the surface.
  - the pair selection in `build_dissociated_endpoint` chooses the bond
    whose terminal atom is LOWEST, instead of the first of several equal
    bonds. After the rotation above, that is the bond that was aimed at the
    metal.

Diatomic behaviour is untouched: H2 and N2 still go down parallel to the
surface, and with two atoms the lowest-terminal rule is a no-op.

API surface is untouched. Same function names, same signatures, same
decorators, same return strings.

NOT FIXED HERE, ON PURPOSE
--------------------------
The site CH4 is placed on. The methane TS on Ni and Ru sits over a top
site; `place_adsorbate` defaults to "ontop" and the agent usually chooses
it, but it is a choice, not a guarantee. Pin it in the task spec rather
than patching a preference into the builder - which site is being modelled
is part of the problem statement, the same argument already made for
terrace versus step.
"""

import re
import sys
from pathlib import Path


OLD_ORIENT = '''def _orient_for_dissociation(ads):
    """Lay a diatomic's bond parallel to the surface. Others are unchanged."""
    if len(ads) != 2:
        return ads
    axis = ads.positions[1] - ads.positions[0]
    if np.linalg.norm(axis) < 1e-6:
        return ads
    ads.rotate(axis, (1.0, 0.0, 0.0), center="COM")
    return ads'''


NEW_ORIENT = '''def _bonded_to(ads, anchor, scale=1.3):
    """Indices bonded to `anchor` by the covalent-radius criterion."""
    r_a = covalent_radii[ads[anchor].number]
    out = []
    for k in range(len(ads)):
        if k == anchor:
            continue
        reach = scale * (r_a + covalent_radii[ads[k].number])
        if np.linalg.norm(ads.positions[k] - ads.positions[anchor]) < reach:
            out.append(k)
    return out


def breaking_bond(atoms, indices):
    """Which bond dissociates: (anchor, terminal, length).

    The anchor is the heaviest adsorbate atom - the fragment that stays
    intact and binds to the surface. The terminal atom is the one bonded to
    it that sits LOWEST, i.e. the one aimed at the metal.

    Height, not bond length, is the discriminator. Methane's four C-H bonds
    are identical to five decimal places, so ranking by length picks
    whichever happens to come first in ASE's atom ordering, which is the
    hydrogen pointing away from the surface. Breaking that one forces the
    band through something very close to gas-phase homolysis and returns a
    barrier several eV too high.

    Raises rather than returning a guess: a molecule with nothing bonded to
    its anchor is not a molecule this function understands, and picking an
    arbitrary pair would hide that.
    """
    if len(indices) < 2:
        raise ValueError("fewer than two adsorbate atoms; nothing to dissociate")

    anchor = max(indices, key=lambda k: covalent_radii[atoms[k].number])
    neighbours = [k for k in _bonded_to(atoms, anchor) if k in indices]
    if not neighbours:
        raise ValueError(
            f"nothing is bonded to the anchor atom ({atoms[anchor].symbol}, "
            f"index {anchor}). The molecule is already dissociated, or the "
            "geometry is distorted past recognition.")

    terminal = min(neighbours, key=lambda k: atoms.positions[k, 2])
    length = float(np.linalg.norm(
        atoms.positions[terminal] - atoms.positions[anchor]))
    return anchor, terminal, length


def _orient_for_dissociation(ads):
    """Point the bond that will break at the surface.

    Diatomic: the bond goes parallel to the surface, which is the geometry
    both atoms need to reach their own site.

    Polyatomic AB_n: one A-B bond points straight down. For methane this is
    the difference between a barrier of 0.8 eV and one of 7 eV. ASE's g2
    geometry puts CH4 edge-down, two hydrogens toward the metal and two
    away, and nothing downstream rotates it, so the bond that breaks is
    chosen from a molecule that was never aimed at the surface.
    """
    if len(ads) < 2:
        return ads

    if len(ads) == 2:
        axis = ads.positions[1] - ads.positions[0]
        if np.linalg.norm(axis) < 1e-6:
            return ads
        ads.rotate(axis, (1.0, 0.0, 0.0), center="COM")
        return ads

    anchor, terminal, _ = breaking_bond(ads, list(range(len(ads))))
    axis = ads.positions[terminal] - ads.positions[anchor]
    if np.linalg.norm(axis) < 1e-6:
        raise ValueError(
            f"{ads.get_chemical_formula()}: the bond chosen to break has zero "
            "length. The input geometry is degenerate.")
    ads.rotate(axis, (0.0, 0.0, -1.0), center=ads.positions[anchor])
    return ads'''


OLD_PAIR = '''    # Two hydrogens on opposite sides of a carbon sit further apart than
    # any C-H bond, so the old "longest internal distance" rule split
    # CH4 into CH2 + H2 instead of CH3 + H.
    best, pair = -1.0, None
    for i in ads:
        for j in ads:
            if i >= j:
                continue
            d = atoms.get_distance(i, j, mic=True)
            r_i = covalent_radii[atoms[i].number]
            r_j = covalent_radii[atoms[j].number]
            bonded = d < 1.3 * (r_i + r_j)
            if bonded and d > best:
                best, pair = d, (i, j)

    if pair is None:
        return ("FAILED: no bonded pair found in the adsorbate. The molecule "
                "may already be dissociated, or the geometry is distorted.")

    a, b = pair'''


NEW_PAIR = '''    # Which bond breaks. Two hydrogens on opposite sides of a carbon sit
    # further apart than any C-H bond, so a "longest internal distance"
    # rule splits CH4 into CH2 + H2. Ranking bonded pairs by length instead
    # fixes that but replaces it with a worse failure: methane's four C-H
    # bonds are degenerate, so the strict comparison keeps whichever comes
    # first in ASE's ordering, which is the hydrogen pointing AWAY from the
    # metal. Breaking that one makes the band do gas-phase homolysis and
    # returned 7 eV against a 0.8 eV reference on CH4/Ru(0001).
    #
    # breaking_bond() ranks by the terminal atom's height instead, and
    # _orient_for_dissociation has already rotated one bond to point at the
    # surface, so the two agree by construction.
    try:
        a, b, best = breaking_bond(atoms, ads)
    except ValueError as exc:
        return f"FAILED: {exc}"

    # The terminal atom must actually be on the surface side. If it is not,
    # the adsorbate was rotated after placement, or placed by something that
    # bypassed _orient_for_dissociation, and the band is about to be asked
    # for homolysis again. Crash rather than produce a number.
    anchor_z = atoms.positions[a, 2]
    terminal_z = atoms.positions[b, 2]
    if len(ads) > 2 and terminal_z > anchor_z + 0.1:
        return (f"FAILED: the bond selected to break ({atoms[a].symbol}-"
                f"{atoms[b].symbol}) points away from the surface - the "
                f"{atoms[b].symbol} sits {terminal_z - anchor_z:.2f} A ABOVE "
                f"the {atoms[a].symbol} it is bonded to. Breaking it forces "
                "the band through gas-phase homolysis and returns a barrier "
                "several eV too high. Rebuild the initial state: "
                "place_adsorbate orients the breaking bond downward, so this "
                "means the geometry was modified after placement.")'''


def apply(path: Path, check_only: bool) -> int:
    text = path.read_text()
    problems = []

    for name, old in (("_orient_for_dissociation", OLD_ORIENT),
                      ("breaking-bond selection", OLD_PAIR)):
        if old not in text:
            problems.append(
                f"  - could not find the original {name} block. The file has "
                "moved on since this patch was written; apply it by hand.")

    if problems:
        print("PATCH DID NOT APPLY:")
        print("\n".join(problems))
        return 1

    text = text.replace(OLD_ORIENT, NEW_ORIENT)
    text = text.replace(OLD_PAIR, NEW_PAIR)

    if check_only:
        print("Both blocks found. Patch would apply cleanly.")
        return 0

    backup = path.with_suffix(path.suffix + ".pre_ch4_fix")
    backup.write_text(path.read_text())
    path.write_text(text)
    print(f"Patched {path}. Original saved to {backup.name}.")
    print("Now run: python scripts/check_structures.py")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    target = Path(args[0]) if args else Path("src/tools.py")
    if not target.exists():
        raise SystemExit(f"{target} does not exist. Pass the path to tools.py.")
    raise SystemExit(apply(target, "--check" in sys.argv))
