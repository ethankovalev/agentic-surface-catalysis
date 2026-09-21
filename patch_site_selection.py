"""
Choose the second fragment's site so the two fragments do not share a
surface metal atom, and rank the remaining choices by coordination.

    python patch_site_selection.py --check src/tools.py
    python patch_site_selection.py src/tools.py

Apply INSTEAD OF patch_shared_metal.py, not after it. That patch added
the same check as a hard failure at the end, which rejected all six
reaction families because nothing upstream was choosing disjoint sites.
This one fixes the choosing.

THE PHYSICS
-----------
Chorkendorff and Niemantsverdriet 6.5.3.2, on N2 over Ru(0001): after
dissociation the two N atoms sit in adjacent threefold sites touching
the SAME metal atom, which is repulsive. The minimum is reached only
once they diffuse onto separate metal atoms. Generally, adsorbate atoms
bound to different metal atoms are more stable than ones sharing.

Measured before this patch, 3x3x4 Ru(0001): the two N atoms came out
2.563 A apart, each bonded to three surface Ru, sharing one. Stage d.
It cleared the existing 2.0 x intact-bond distance guard, because that
guard is about recombination, not about shared metal atoms.

The cell is not the constraint. On that slab, 45 of 153 site pairs share
no metal atom, one of them 4.23 A apart. The search simply was not
looking for them.

COORDINATION RANKING, AND ITS LIMITS
-------------------------------------
Among the disjoint sites, higher coordination is preferred for atoms
that the literature shows bind strongly in hollows:

  C on Fe(100): hollow is 2.92 eV below ontop and 1.48 eV below bridge
    (Bromfield, Curulla Ferre and Niemantsverdriet, ChemPhysChem 2005,
     doi 10.1002/cphc.200400452)
  N on Rh(311): binds strongest at the hcp threefold terrace site
    (Inderwildi et al., ChemPhysChem 2005, doi 10.1002/cphc.200500222)
  O on Pt(111): fcc hollow preferred, bridge unstable and slides into it
    (Cahyanto et al., Surf Interface Anal 2016, doi 10.1002/sia.5936)

H and CH3 are deliberately NOT given the same preference, because the
metal decides for them:

  CH3 prefers hollow on Fe, Ni, Rh, Cu but TOP on Pt, Pd, Au, Ag
    (Wang et al., J Comput Chem 2005, doi 10.1002/jcc.20225)
  H prefers threefold hollow on most transition metals except Ir, and on
    Pt(111) all four sites lie within about 2 kJ/mol of each other
    (Ai et al., Angew Chem Int Ed 2020, doi 10.1002/anie.201915663;
     Wu, Xiong and Yang, ChemCatChem 2025, doi 10.1002/cctc.202500902)

So for H and CH3 the ranking falls back to distance alone, and the
coordination number is recorded rather than acted on. Encoding a
metal-dependent preference would need per-metal data this module does
not carry, and guessing it would be worse than not ranking.

WHAT THIS DOES NOT DO
-----------------------
_surface_sites generates hollows only. Bridge and top sites are not
enumerated at all, so "evaluate top and bridge for H and CH3" cannot be
implemented here without first extending the site search. That is real
work and is not attempted in this patch. The coordination number is
recorded on every placement so that when those sites do exist, the
ranking has something to rank.
"""

import sys
from pathlib import Path


OLD_HELPER = '''@tool
def build_dissociated_endpoint(separation: float = None,'''

NEW_HELPER = '''# Atoms whose binding is dominated by coordination number, so that the
# highest-coordination site available is the right default. See the
# module docstring of patch_site_selection.py for the sources.
COORDINATION_DRIVEN = {"C", "N", "O"}


def _surface_layer(atoms, metal):
    """Indices of the topmost metal layer."""
    if not metal:
        return []
    top_z = max(atoms.positions[i, 2] for i in metal)
    return [i for i in metal if atoms.positions[i, 2] > top_z - 0.5]


def _site_metal_neighbours(atoms, xy, z, surface, cutoff=2.8):
    """Which surface metal atoms a site at (xy, z) would touch.

    Distances are taken in the plane with the minimum image convention,
    because a site near a cell edge is adjacent to atoms on the far side.
    """
    cell2 = np.array(atoms.cell[:2, :2], dtype=float)
    inv2 = np.linalg.inv(cell2)
    found = set()
    for i in surface:
        d = (np.asarray(xy) - atoms.positions[i, :2]) @ inv2
        d -= np.round(d)
        lateral = np.linalg.norm(d @ cell2)
        dz = abs(z - atoms.positions[i, 2])
        if np.hypot(lateral, dz) < cutoff:
            found.add(i)
    return found


@tool
def build_dissociated_endpoint(separation: float = None,'''


OLD_CHOICE = '''                    by_distance = sorted(
                        free, key=lambda n: _mic_xy(sites[n][0], first_xy, cell2, inv2))
                    best_n = next('''

NEW_CHOICE = '''                    # Sites that share no surface metal atom with the
                    # first fragment. Adjacent threefold sites touching
                    # one metal atom are the repulsive arrangement, not
                    # the product minimum (Chorkendorff and
                    # Niemantsverdriet 6.5.3.2). Before this filter, all
                    # six SBH10 reaction families built endpoints that
                    # shared an atom.
                    surface = _surface_layer(atoms, metal)
                    first_nbrs = _site_metal_neighbours(
                        atoms, sites[used[0]][0], sites[used[0]][1], surface)
                    disjoint = [
                        n for n in free
                        if not (_site_metal_neighbours(
                            atoms, sites[n][0], sites[n][1], surface)
                            & first_nbrs)]
                    if not disjoint:
                        raise ValueError(
                            f"no site for the second fragment shares zero "
                            f"surface metal atoms with the first. Every "
                            f"available site would give the repulsive "
                            f"adjacent arrangement rather than the product "
                            f"minimum. The cell is too small to hold the "
                            f"dissociated state; widen it.")

                    # Among those, prefer high coordination for atoms whose
                    # binding is coordination driven. H and CH3 are left to
                    # distance alone: the preferred site for those depends
                    # on the metal, and this module has no per-metal data.
                    anchor_sym = atoms[anchor].symbol
                    if anchor_sym in COORDINATION_DRIVEN:
                        disjoint.sort(
                            key=lambda n: (
                                -len(_site_metal_neighbours(
                                    atoms, sites[n][0], sites[n][1], surface)),
                                _mic_xy(sites[n][0], first_xy, cell2, inv2)))
                        free = disjoint
                    else:
                        free = disjoint

                    by_distance = sorted(
                        free, key=lambda n: _mic_xy(sites[n][0], first_xy, cell2, inv2))
                    best_n = next('''


EDITS = [
    ("coordination helpers", OLD_HELPER, NEW_HELPER),
    ("second fragment site choice", OLD_CHOICE, NEW_CHOICE),
]


def main():
    check_only = "--check" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(args[0]) if args else Path("src/tools.py")

    if not path.exists():
        print(f"FAILED: {path} does not exist. Run from the repo root.")
        return 1

    text = path.read_text()
    if "_site_metal_neighbours" in text:
        print("Already patched. Nothing to do.")
        return 0
    if "_shares_metal_atom" in text:
        print("FAILED: patch_shared_metal.py is applied. Revert it first "
              "(src/tools.py.pre_shared_metal); this patch replaces it.")
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

    backup = Path(str(path) + ".pre_site_selection")
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
