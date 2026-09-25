"""
Keep every NEB band on the persistent volume, and show it in the
diagnostic.

    python patches/patch_save_bands.py --check
    python patches/patch_save_bands.py

Run from the repo root. Edits scripts/run_blind.py and
scripts/diagnose_blind.py.

WHY
---
config.WORK_DIR is the repository's own work/ folder, which sits on the
pod's container disk. When a pod is replaced, work/ is empty. The result
files and the five structures run_blind.py copies (initial, final,
gasref, peak, saddle) live on /workspace and survive; the band did not.

On 2026-09-24 that cost the diagnosis of N2/Ru(0001) step. Band 2's peak
held a near-intact N2 at 1.23 A and the next image was 0.94 eV lower, so
the bond breaks in the gap between them. Confirming that needs the band's
images, and they were in work/neb.traj on a pod that no longer existed.

WHAT CHANGES
------------
run_blind.py: after each band, its final images are written to
work/band_band1.traj or work/band_band2.traj, and copied into the result
folder with the other structures. run_neb's neb.traj holds either the
whole optimisation history (every image at every step) or, after a
walled retry, just the final band; in both cases the final band is the
last n_images frames, and n_images is taken from the NEB record rather
than assumed.

diagnose_blind.py: a fifth section prints each saved band image by
image: energy relative to the first image, breaking bond length, and
whether any adsorbate atom has left the surface. The step where a bond
breaks, and where an unsampled gap sits, can then be read directly.
"""

import sys
from pathlib import Path

BLIND = Path("scripts/run_blind.py")
DIAG = Path("scripts/diagnose_blind.py")

STRUCT_OLD = '''STRUCTURES = ("initial", "final", "gasref", "peak", "saddle")'''
STRUCT_NEW = '''STRUCTURES = ("initial", "final", "gasref", "peak", "saddle")
# Each band's final images. work/ is on the container disk and is lost
# when a pod is replaced; these are copied to the result folder with the
# structures above. See patches/patch_save_bands.py.
BANDS = ("band_band1", "band_band2")'''

KEEP_OLD = '''    dst = work / f"peak_{label}.traj"
    shutil.copy(src, dst)
    neb = neb or {}'''
KEEP_NEW = '''    dst = work / f"peak_{label}.traj"
    shutil.copy(src, dst)
    neb = neb or {}

    # The final band is the last n_images frames of neb.traj, whether that
    # file holds the whole optimisation history or only a walled retry's
    # final band.
    history = work / "neb.traj"
    n_band = neb.get("n_images")
    if history.exists() and n_band:
        try:
            frames = read(str(history), index=":")
            write(str(work / f"band_{label}.traj"), frames[-n_band:])
        except Exception as exc:
            print(f"  could not save band {label}: {type(exc).__name__}: {exc}")'''

COPY_OLD = '''            for name in STRUCTURES:
                src = work / f"{name}.traj"
                if src.exists():
                    shutil.copy(src, folder / f"{name}.traj")'''
COPY_NEW = '''            for name in STRUCTURES + BANDS:
                src = work / f"{name}.traj"
                if src.exists():
                    shutil.copy(src, folder / f"{name}.traj")'''

IMPORT_OLD = "from ase.io import read  # noqa: E402"
IMPORT_NEW = "from ase.io import read, write  # noqa: E402"

DIAG_OLD = '''        print(f"  {name}: breaking bond {bond_text}")
        for line in describe(atoms):
            print(line)


if __name__ == "__main__":'''
DIAG_NEW = '''        print(f"  {name}: breaking bond {bond_text}")
        for line in describe(atoms):
            print(line)

    print("\\n5. SAVED BANDS")
    any_band = False
    for label in ("band1", "band2"):
        f = folder / f"band_{label}.traj"
        if not f.exists():
            continue
        any_band = True
        images = read(str(f), index=":")
        print(f"  {label}: {len(images)} images")
        first = None
        for n, image in enumerate(images):
            try:
                energy = image.get_potential_energy()
                first = energy if first is None else first
                e_text = f"{energy - first:+.3f} eV"
            except Exception:
                e_text = "   n/a   "
            bond = breaking_bond(image)
            b_text = "n/a" if bond is None else f"{bond[2]:.2f} A"
            flags = [line.split()[-1] for line in describe(image)
                     if not line.endswith("on the surface")]
            note = f"   {', '.join(flags)}" if flags else ""
            print(f"    image {n:2d}   E {e_text}   breaking bond {b_text}{note}")
    if not any_band:
        print("  none saved (runs before patch_save_bands.py kept no bands)")


if __name__ == "__main__":'''

EDITS = [
    (BLIND, "band names", STRUCT_OLD, STRUCT_NEW),
    (BLIND, "save each band", KEEP_OLD, KEEP_NEW),
    (BLIND, "copy bands to the result folder", COPY_OLD, COPY_NEW),
    (BLIND, "write import", IMPORT_OLD, IMPORT_NEW),
    (DIAG, "band section", DIAG_OLD, DIAG_NEW),
]


def main():
    for p in (BLIND, DIAG):
        if not p.exists():
            print(f"FAILED: {p} not found. Run from the repo root.")
            return 1
    texts = {p: p.read_text() for p in (BLIND, DIAG)}
    if "BANDS = (" in texts[BLIND]:
        print("Already patched. Nothing to do.")
        return 0
    problems = [f"{p}: {name} anchor found {texts[p].count(old)} times"
                for p, name, old, _ in EDITS if texts[p].count(old) != 1]
    if problems:
        print("FAILED, nothing written:")
        for line in problems:
            print(f"  {line}")
        return 1
    if "--check" in sys.argv:
        print(f"All {len(EDITS)} anchors found. Patch would apply cleanly.")
        return 0
    for p in (BLIND, DIAG):
        backup = Path(str(p) + ".pre_save_bands")
        if not backup.exists():
            backup.write_text(texts[p])
    for p, name, old, new in EDITS:
        texts[p] = texts[p].replace(old, new, 1)
    for p in (BLIND, DIAG):
        p.write_text(texts[p])
    print("Patched scripts/run_blind.py (bands saved) and "
          "scripts/diagnose_blind.py (band section).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
