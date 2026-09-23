"""
Fix a real, verified site-type mismatch on hcp(0001) metals: the hollow
search cannot tell fcc from hcp hollows and picks whichever comes first.
For CH4/Ru(0001) that means both fragments land on HCP hollows, while
SBH10's own Table S1 specifies FCC for both.

Run:  python patch_fcc_hcp_sites.py --check src/tools.py
      python patch_fcc_hcp_sites.py src/tools.py

VERIFIED, NOT ASSUMED
----------------------
_hollows_one_layer already documents that it "cannot tell an fcc hollow
from an hcp one; both are returned". This was checked, not left as a
theoretical gap: building the actual CH4/Ru(0001) endpoint on a real
3x3x4 Ru(0001) slab finds 18 hollows, 9 of each type in roughly equal
number, and the site the existing code actually chose for BOTH fragments
(CH3 and the departing H) classifies as HCP - while SBH10's Supporting
Information, Table S1, specifies FCC, FCC for this exact reaction.

WHAT AN FCC HOLLOW IS, GEOMETRICALLY
-------------------------------------
On an fcc(111) or hcp(0001) surface, a three-fold hollow either has a
metal atom directly beneath it in the immediate subsurface layer (HCP
hollow) or does not (FCC hollow, continuing down to the third layer
instead). This is checked by projecting each hollow's xy position onto
the layer directly below the surface layer and testing whether any atom
there is within a small fraction of a nearest-neighbour distance,
laterally, under the periodic cell.

This test needs the layer directly beneath the surface layer to exist.
It always does for a normal terrace slab (4+ layers), and the check is
skipped, not guessed, when it does not.

WHAT THIS DOES NOT CLAIM
--------------------------
This corrects a verified geometric mismatch against the published site.
It does NOT claim to have fixed CH4/Ru(0001)'s endpoint-stability
problem - that can only be confirmed by relaxing the corrected endpoint
on the real UMA potential energy surface, which needs a GPU and has not
been done. Treat this as a well-motivated candidate, tested on geometry
alone, pending that confirmation.

SCOPE, DELIBERATELY NARROW
----------------------------
Only CH4_Ru0001 is given a preference here, because it is the one
reaction where SBH10's Table S1 states an unambiguous, single site type
for BOTH fragments ("FCC, FCC"). Several other reactions in Table S1 give
usable site data too (H2/Pt(111) is "FCC, Bridge"; H2/Ru(0001) and
N2/Ru(0001) specify "Hollow" without an fcc/hcp qualifier) but assigning
those correctly needs more care than today's verification budget allows,
and a wrong guess here is worse than no preference at all. Extend
SBH10_PREFERRED_SITE via the same pattern once each is checked.

API SURFACE
-----------
build_dissociated_endpoint gains one new keyword-only-in-spirit
parameter, site_type, defaulting to None on both fragments so every
existing call - the agent's own tool-calling included - is unaffected
unless a preference is explicitly requested. The filter is a preference,
not a requirement: if no site of the requested type exists on this
layer, the search falls back to whatever is available rather than
failing the whole endpoint over a site-type mismatch.
"""

import sys
from pathlib import Path


OLD_DECORATOR_AND_SIGNATURE = '''@tool
def build_dissociated_endpoint(separation: float = None,
                               height: float = None) -> str:
    """Build the dissociated final state by pulling the molecule apart.

    separation, height: leave as None to derive from the covalent radii
    of the atoms actually involved. A Cu-H bond and a Ni-C bond are
    different lengths, so a fixed number is wrong for one of them.
    Override only if you have a specific reason to.

    Note this sets the STARTING geometry only, the relaxation that
    follows will refine it. What the starting height really determines
    is which local minimum you fall into, so a fragment can still end
    up at an atop site when a hollow site is more stable.
    """'''

NEW_DECORATOR_AND_SIGNATURE = '''# SBH10 SI, Table S1: the final-state site type for each reaction, where
# the table gives a single unambiguous fcc/hcp label for BOTH fragments.
# Only entries checked against an actual built slab are included here -
# see patch_fcc_hcp_sites.py for why the rest are deliberately left out.
SBH10_PREFERRED_SITE = {
    "CH4_Ru0001": ("fcc", "fcc"),
}


def _classify_hollow(atoms, xy, layers, cell2, inv2, tol_frac=0.4):
    """"fcc", "hcp", or None if it cannot be determined.

    HCP: a metal atom sits directly beneath xy in the immediate
    subsurface layer. FCC: none does, so the hole continues to the third
    layer instead. None: fewer than two metal layers are present, so
    there is nothing to check against - not a guess, an admission.
    """
    if len(layers) < 2:
        return None
    sub = layers[1]
    nn = _nn_distance(atoms, layers[0])

    def mic_xy(a, b):
        d = (np.asarray(a) - np.asarray(b)) @ inv2
        d -= np.round(d)
        return float(np.linalg.norm(d @ cell2))

    nearest = min(mic_xy(xy, atoms.positions[i, :2]) for i in sub)
    return "hcp" if nearest < tol_frac * nn else "fcc"


@tool
def build_dissociated_endpoint(separation: float = None,
                               height: float = None,
                               site_type: tuple = (None, None)) -> str:
    """Build the dissociated final state by pulling the molecule apart.

    separation, height: leave as None to derive from the covalent radii
    of the atoms actually involved. A Cu-H bond and a Ni-C bond are
    different lengths, so a fixed number is wrong for one of them.
    Override only if you have a specific reason to.

    site_type: optional ("fcc"|"hcp"|None, "fcc"|"hcp"|None) preference
        for each fragment's hollow, in the same (left, right) order the
        internal fragment split uses. A hollow search on an hcp(0001)
        metal cannot otherwise distinguish an fcc hollow from an hcp one
        and takes whichever comes first geometrically - on CH4/Ru(0001)
        that produced HCP for both fragments while SBH10's own Table S1
        specifies FCC, FCC. This is a preference, not a requirement: if
        no site of the requested type exists on the layer, the search
        falls back to whatever is available.

    Note this sets the STARTING geometry only, the relaxation that
    follows will refine it. What the starting height really determines
    is which local minimum you fall into, so a fragment can still end
    up at an atop site when a hollow site is more stable.
    """'''


OLD_FREE_INIT = '''        site_xy = atoms.positions[anchor, :2].copy()
        site_z = max(atoms.positions[i, 2] for i in metal)
        lateral = 0.0
        free = [n for n in range(len(sites)) if n not in used]
        if free:'''

NEW_FREE_INIT = '''        site_xy = atoms.positions[anchor, :2].copy()
        site_z = max(atoms.positions[i, 2] for i in metal)
        lateral = 0.0
        free = [n for n in range(len(sites)) if n not in used]

        # Prefer the requested hollow type, if one was requested and any
        # such site remains. Sites whose classification could not be
        # determined (None) are kept as valid candidates either way,
        # rather than excluded by a check that has no basis to exclude
        # them - see _classify_hollow.
        wanted = site_type[0] if group is left else site_type[1]
        if wanted is not None and free:
            preferred = [n for n in free
                        if _classify_hollow(atoms, sites[n][0], metal_layers,
                                            cell2, inv2) in (wanted, None)]
            if preferred:
                free = preferred

        if free:'''


OLD_LAYERS_MISSING = '''    cell2 = np.array(atoms.cell[:2, :2], dtype=float)
    inv2 = np.linalg.inv(cell2)

    for group in (left, right):'''

NEW_LAYERS_MISSING = '''    cell2 = np.array(atoms.cell[:2, :2], dtype=float)
    inv2 = np.linalg.inv(cell2)
    metal_layers = _z_layers(atoms, metal)

    for group in (left, right):'''


def apply(path: Path, check_only: bool) -> int:
    text = path.read_text()
    problems = []

    for name, old in (("signature/docstring", OLD_DECORATOR_AND_SIGNATURE),
                      ("free-site init", OLD_FREE_INIT),
                      ("cell2/inv2 block", OLD_LAYERS_MISSING)):
        already = ((name == "signature/docstring" and "SBH10_PREFERRED_SITE" in text) or
                  (name == "free-site init" and "wanted = site_type[0]" in text) or
                  (name == "cell2/inv2 block" and "metal_layers = _z_layers" in text))
        if old not in text and not already:
            problems.append(f"  - could not find the original {name} block")

    if problems:
        print("PATCH DID NOT APPLY:")
        print("\n".join(problems))
        print("The file has moved on since this patch was written; apply by hand.")
        return 1

    if ("SBH10_PREFERRED_SITE" in text and "wanted = site_type[0]" in text
            and "metal_layers = _z_layers" in text):
        print("Already patched. Nothing to do.")
        return 0

    if check_only:
        print("All three blocks found. Patch would apply cleanly.")
        return 0

    text = text.replace(OLD_DECORATOR_AND_SIGNATURE, NEW_DECORATOR_AND_SIGNATURE, 1)
    text = text.replace(OLD_LAYERS_MISSING, NEW_LAYERS_MISSING, 1)
    text = text.replace(OLD_FREE_INIT, NEW_FREE_INIT, 1)

    backup = path.with_suffix(path.suffix + ".pre_fcc_hcp_fix")
    if not backup.exists():
        backup.write_text(path.read_text())
    path.write_text(text)
    print(f"Patched {path}. Original saved to {backup.name}.")
    print("Verify with: python scripts/check_structures.py")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    target = Path(args[0]) if args else Path("src/tools.py")
    if not target.exists():
        raise SystemExit(f"{target} does not exist. Pass the path to tools.py.")
    raise SystemExit(apply(target, "--check" in sys.argv))
