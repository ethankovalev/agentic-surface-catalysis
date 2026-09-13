"""
Fix two verified defects in _classify_hollow, introduced by
patch_fcc_hcp_sites.py.

Run:  python patch_fcc_hcp_v2.py --check src/tools.py
      python patch_fcc_hcp_v2.py src/tools.py

Apply AFTER patch_fcc_hcp_sites.py. Neither defect changes any current
result - the only entry in SBH10_PREFERRED_SITE is CH4_Ru0001, a terrace
hcp(0001) reaction, which is exactly the case the original patch was
verified on. Both are latent: they would produce silently wrong site
labels the moment a preference was set for a stepped or 4-fold reaction.

DEFECT 1: WRONG SUBSURFACE LAYER ON STEPPED SLABS
--------------------------------------------------
_classify_hollow always compared against layers[1], no matter which layer
the site itself belonged to. _surface_sites searches up to two layers, so
on a stepped slab a site on layer 1 was compared against layer 1 - itself.
A hollow always sits between atoms of its own layer, so the nearest
same-layer atom is roughly the clearance (~1.5 A), comfortably past the
0.4 x nn threshold (~1.08 A), and every such site came back "fcc"
regardless of what it actually is.

Measured on a stepped Ru(0001) slab: layer 0 gives a healthy
{fcc, hcp} mix, layer 1 gives {fcc} for all 34 sites. That uniformity is
the tell.

Fixed by locating the site's OWN layer from its z, then using the layer
directly beneath that one. If no layer sits beneath it, the function
returns None - an admission, not a guess.

DEFECT 2: FCC/HCP IS MEANINGLESS FOR FOUR-FOLD HOLLOWS
-------------------------------------------------------
On fcc(100) every hollow is four-fold and the fcc/hcp distinction does
not exist - there is one kind of hollow. The original returned a
confident "fcc" for all nine sites on both Ni(100) and Cu(100).

Harmless in practice, because requesting "hcp" there simply finds no
preferred site and falls back. But a function that returns a confident
label for a property the surface does not have is exactly the kind of
plausible-looking wrong answer this project exists to catch.

Fixed by counting near-equidistant neighbours in the site's own layer: a
three-fold hollow has 3, a four-fold has 4. Four or more returns None.

WHAT IS STILL NOT HANDLED
--------------------------
Site classification on the lower terrace of a stepped slab is now
correct where a layer exists beneath it, and honest (None) where one does
not. It has NOT been verified against a published stepped-surface site
assignment, because SBH10 Table S1 gives "Hollow, Bridge" for
N2/Ru(0001)-step without an fcc/hcp qualifier. Do not add a stepped
reaction to SBH10_PREFERRED_SITE without checking it the same way
CH4_Ru0001 was checked.
"""

import sys
from pathlib import Path


OLD = '''def _classify_hollow(atoms, xy, layers, cell2, inv2, tol_frac=0.4):
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
    return "hcp" if nearest < tol_frac * nn else "fcc"'''


NEW = '''def _classify_hollow(atoms, xy, layers, cell2, inv2, tol_frac=0.4,
                     site_z=None, z_tol=0.6):
    """"fcc", "hcp", or None when the distinction does not apply.

    HCP: a metal atom sits directly beneath xy in the layer immediately
    below the site's own layer. FCC: none does, so the hole continues
    past that layer instead.

    None is returned, rather than a guess, in three cases:

    - the site's own layer cannot be identified from site_z
    - no layer exists beneath the site's own layer
    - the site is a four-fold hollow, where fcc and hcp are not distinct
      site types at all. On fcc(100) there is exactly one kind of hollow
      and an earlier version of this function confidently labelled all
      nine of them "fcc".

    site_z: the z of the layer the site belongs to, as carried in the
        third element of each _surface_sites entry. Required to pick the
        right subsurface layer. Without it the function falls back to
        layers[0]/layers[1], which is correct only on a single-terrace
        slab - on a stepped slab it compared layer-1 sites against layer
        1 itself and returned "fcc" for every one of them.
    """
    if not layers:
        return None

    def mic_xy(a, b):
        d = (np.asarray(a) - np.asarray(b)) @ inv2
        d -= np.round(d)
        return float(np.linalg.norm(d @ cell2))

    # Which layer does this site sit on, and which is directly beneath it?
    own = 0
    if site_z is not None:
        means = [float(np.mean(atoms.positions[l, 2])) for l in layers]
        gaps = [abs(site_z - mz) for mz in means]
        own = int(np.argmin(gaps))
        if gaps[own] > z_tol:
            return None                 # site belongs to no known layer
    if own + 1 >= len(layers):
        return None                     # nothing beneath it to check

    own_layer, sub = layers[own], layers[own + 1]
    if len(own_layer) < 3:
        return None
    nn = _nn_distance(atoms, own_layer)

    # Three-fold or four-fold? fcc/hcp only means something for three-fold
    # hollows. Count the site's near-equidistant neighbours in its own
    # layer, using the same 1.15x tolerance _hollows_one_layer uses to
    # decide a point is a hollow at all.
    same = sorted(mic_xy(xy, atoms.positions[i, :2]) for i in own_layer)
    if same and np.count_nonzero(np.asarray(same) < 1.15 * same[0]) >= 4:
        return None                     # four-fold: no fcc/hcp distinction

    nearest = min(mic_xy(xy, atoms.positions[i, :2]) for i in sub)
    return "hcp" if nearest < tol_frac * nn else "fcc"'''


OLD_CALL = '''            preferred = [n for n in free
                        if _classify_hollow(atoms, sites[n][0], metal_layers,
                                            cell2, inv2) in (wanted, None)]'''

NEW_CALL = '''            # sites[n] is (xy, z, clearance); z identifies the site's own
            # layer, without which a stepped slab is classified against
            # the wrong subsurface.
            preferred = [n for n in free
                        if _classify_hollow(atoms, sites[n][0], metal_layers,
                                            cell2, inv2,
                                            site_z=sites[n][1]) == wanted]'''


def apply(path: Path, check_only: bool) -> int:
    text = path.read_text()

    if "site_z=None, z_tol=0.6" in text and "site_z=sites[n][1]" in text:
        print("Already patched. Nothing to do.")
        return 0

    problems = []
    if OLD not in text:
        problems.append("  - could not find the original _classify_hollow")
    if OLD_CALL not in text:
        problems.append("  - could not find the preference filter call site")

    if problems:
        print("PATCH DID NOT APPLY:")
        print("\n".join(problems))
        print("Apply patch_fcc_hcp_sites.py first, then this one.")
        return 1

    if check_only:
        print("Both blocks found. Patch would apply cleanly.")
        return 0

    text = text.replace(OLD, NEW, 1).replace(OLD_CALL, NEW_CALL, 1)
    backup = path.with_suffix(path.suffix + ".pre_fcc_hcp_v2")
    if not backup.exists():
        backup.write_text(path.read_text())
    path.write_text(text)
    print(f"Patched {path}. Original saved to {backup.name}.")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    target = Path(args[0]) if args else Path("src/tools.py")
    if not target.exists():
        raise SystemExit(f"{target} does not exist.")
    raise SystemExit(apply(target, "--check" in sys.argv))
