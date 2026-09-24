"""
Stop an NEB band routing adsorbate atoms through the metal, and never
report a barrier from a band the pipeline itself rejected.

    python patches/patch_band_wall.py --check
    python patches/patch_band_wall.py

Run from the repo root. Edits src/tools.py and scripts/run_blind.py.

WHAT WENT WRONG, N2/Ru(0001) STEP, 2026-09-24
---------------------------------------------
Both endpoints were genuine minima: an intact N2 at 1.17 A, and two N
atoms in threefold sites 2.93 A apart. Neither band converged. The first
band's peak had the N atoms 4.16 A apart, further apart than the product,
with one N touching six metal atoms; the geometry check put it 2.66 A
below the top metal layer, below the lower terrace, inside the metal.
Refining that peak gave a saddle for an N atom moving around in there:
both IRC ends stayed dissociated, at 3.41 and 4.40 A. The run reported
3.611 eV.

The day before, the same reaction's band avoided the pocket and reached
0.982 eV. An unconverged band settles into whichever route the optimiser
finds, so the result was not reproducible.

FIX 1: DETECT, THEN RETRY WITH A HARD GEOMETRIC WALL (src/tools.py)
-------------------------------------------------------------------
After the band, every interior image is checked with the same
surrounded-by-metal test the geometry check uses. If any image puts an
adsorbate atom inside the slab, the band is rebuilt from the endpoints
and optimised again with a wall 0.2 A below the lowest exposed metal
surface.

THE WALL ADDS NO ENERGY. It is a purely geometric constraint, like
ASE's own FixAtoms:
  - positions: an adsorbate atom that would move below the wall is
    placed on the wall instead;
  - forces: for an atom sitting on the wall, the component pushing it
    further in is removed, exactly as FixAtoms zeroes the forces on
    fixed atoms, so the optimiser can converge against the wall;
  - energy: nothing. ASE adds a constraint's energy only if the
    constraint defines adjust_potential_energy (Atoms.get_potential_energy
    checks for it by name). SurfaceWall does not define it, so every
    energy read while the wall is in place is the calculator's own
    number, exactly. An earlier draft used ASE's Hookean restraint,
    which DOES define it: a spring penalty that shaped the band while it
    optimised. That draft was never applied.

Above the wall it does nothing: positions, forces and energies are
untouched. A band that never enters the slab is not affected at all,
because the wall is never built.

The wall restricts where the band may go; it does not change the energy
of anywhere the band goes. It is removed before peak.traj is written, so
saddle refinement, connectivity and zero point correction all run with
no wall, and if refinement drives an atom into the slab anyway the
geometry check refuses the result.

The wall sits BELOW the surface plane, not at it: carbon and hydrogen in
fourfold hollows on (100) faces sit only 0.1 to 0.5 A above the plane,
and must not be touched.

SBH10 asks for SURFACE dissociation barriers. A route through the
subsurface is a different process, so excluding it from the path search
states what is being computed.

What the wall does not cover: the floor is the LOWEST exposed surface.
On a stepped slab that is the lower terrace, so an atom slipping under
the upper terrace would not meet it. The images are checked again after
the retry, and anything still inside the slab is recorded and reported.

FIX 2: NO BARRIER FROM A REJECTED BAND (scripts/run_blind.py)
-------------------------------------------------------------
The policy skipped band 2's peak as implausible (3.600 eV, above 3.0).
Nothing connected, so the gas-referenced barrier fell back to "the NEB
peak", and the NEB record in memory was band 2's: the one the policy had
rejected. That number went into the result and the paper tables.

A barrier is now recorded only when a connected saddle was found. Without
one, computed_eV is None, the reason is stated, and every band's barrier
is kept alongside for information.
"""

import sys
from pathlib import Path

TOOLS = Path("src/tools.py")
BLIND = Path("scripts/run_blind.py")

IMPORT_OLD = "from ase.constraints import FixAtoms"
IMPORT_NEW = "from ase.constraints import FixAtoms, FixConstraint"

HELPERS_OLD = "@tool\ndef run_neb("
HELPERS_NEW = '''# The band wall. See patches/patch_band_wall.py for the evidence.
WALL_MARGIN_A = 0.2      # below the lowest exposed metal surface


def _images_inside_slab(images):
    """Indices of interior band images with an adsorbate atom inside the slab.

    Uses the same surrounded-by-metal test as the geometry check, so a band
    is judged by exactly the rule the final result will be judged by.
    """
    inside = []
    for n, image in enumerate(images[1:-1], start=1):
        if _subsurface_adsorbates(image):
            inside.append(n)
    return inside


def _exposed_surface_floor(atoms):
    """Height of the lowest metal atom with nothing above it.

    A metal atom is exposed if no other metal atom sits over it, within
    1.5 A sideways and at least 0.5 A higher. Bottom layer atoms have atoms
    above them, so they never count. On a stepped slab the answer is the
    lower terrace.
    """
    tags = atoms.get_tags()
    metal = [i for i in range(len(atoms)) if tags[i] != 2]
    cell2 = np.array(atoms.cell[:2, :2], dtype=float)
    inv2 = np.linalg.inv(cell2)
    exposed = []
    for i in metal:
        covered = False
        for j in metal:
            if j == i or atoms.positions[j, 2] < atoms.positions[i, 2] + 0.5:
                continue
            d = (atoms.positions[j, :2] - atoms.positions[i, :2]) @ inv2
            d -= np.round(d)
            if np.linalg.norm(d @ cell2) < 1.5:
                covered = True
                break
        if not covered:
            exposed.append(atoms.positions[i, 2])
    return float(min(exposed)) if exposed else float(max(
        atoms.positions[i, 2] for i in metal))


class SurfaceWall(FixConstraint):
    """A hard, one-sided wall under the exposed surface. Geometric only.

    Like FixAtoms, it acts on positions and forces and never on energy:
      - an adsorbate atom that would move below wall_z is placed on it;
      - for an atom on the wall, the force component pushing it further
        in is removed, so the optimiser converges against the wall;
      - no adjust_potential_energy is defined, and ASE only adds a
        constraint's energy when that method exists. Every energy read
        with this wall in place is the calculator's energy, exactly.
    Above the wall it does nothing.

    It is never saved into a trajectory: ASE cannot read back a constraint
    it does not know, so it is always removed before anything is written.
    """

    def __init__(self, indices, wall_z):
        self.index = [int(i) for i in indices]
        self.wall_z = float(wall_z)

    def get_removed_dof(self, atoms):
        return 0

    def adjust_positions(self, atoms, new):
        for i in self.index:
            if new[i, 2] < self.wall_z:
                new[i, 2] = self.wall_z

    def adjust_forces(self, atoms, forces):
        for i in self.index:
            on_wall = atoms.positions[i, 2] <= self.wall_z + 1e-6
            if on_wall and forces[i, 2] < 0.0:
                forces[i, 2] = 0.0

    def copy(self):
        # The base class copies through ASE's constraint registry, which
        # does not know this class.
        return SurfaceWall(self.index, self.wall_z)

    def todict(self):
        return {"name": "SurfaceWall",
                "kwargs": {"indices": self.index, "wall_z": self.wall_z}}


def _add_wall(images, wall_z):
    """A SurfaceWall on the adsorbate atoms of every interior image. Any
    atom already below the wall is moved up onto it at once."""
    for image in images[1:-1]:
        tags = image.get_tags()
        ads = [i for i in range(len(image)) if tags[i] == 2]
        image.set_constraint(list(image.constraints) + [SurfaceWall(ads, wall_z)])
        image.set_positions(image.get_positions())


def _remove_wall(images):
    """Strip the wall, keeping the fixed layers."""
    for image in images:
        image.set_constraint([c for c in image.constraints
                              if not isinstance(c, SurfaceWall)])


def _optimise_band(start, end, n_images, model_key, with_d3, max_steps,
                   fmax, wall_z=None):
    """Build, interpolate and optimise one band. Returns (images, converged).

    Two passes with FIRE: climbing image off to let the band settle, then
    on with a fresh optimiser. The wall, if any, is added after
    interpolation so IDPP is unaffected, and is left on the images; it
    adds no energy, and the caller removes it before peak.traj is written.
    """
    images = [start.copy()]
    for _ in range(n_images):
        images.append(start.copy())
    images.append(end.copy())
    for image in images:
        image.calc = new_calculator(model_key, with_d3=with_d3)

    neb = NEB(images, climb=False, k=config.NEB_SPRING_K,
              method="improvedtangent")
    neb.interpolate(method="idpp")
    if wall_z is not None:
        _add_wall(images, wall_z)

    # Pass 1: rough path, climbing image off. Enabling climb before the
    # band has settled is a common cause of oscillation.
    FIRE(neb, logfile="-").run(fmax=0.2, steps=max_steps // 2)

    # Pass 2: fresh optimiser. Climbing image inverts the parallel force
    # on the peak, so pass 1's accumulated velocity now describes a
    # function that no longer exists.
    neb.climb = True
    # No step-by-step trajectory while the wall is on: ASE cannot read back
    # a file holding a constraint it does not know. run_neb writes the final
    # band to neb.traj once the wall is removed. Without a wall, neb.traj is
    # written exactly as before.
    trajectory = _path("neb.traj") if wall_z is None else None
    opt2 = FIRE(neb, trajectory=trajectory, logfile="-")
    converged = opt2.run(fmax=fmax, steps=max_steps)
    return images, bool(converged)


@tool
def run_neb('''

BAND_OLD = '''    start = read(_path("initial.traj"))
    end = read(_path("final.traj"))

    images = [start]
    for _ in range(n_images):
        img = start.copy()
        img.calc = new_calculator(model_key, with_d3=with_d3)
        images.append(img)
    images.append(end)

    start.calc = new_calculator(model_key, with_d3=with_d3)
    end.calc = new_calculator(model_key, with_d3=with_d3)

    neb = NEB(images, climb=False, k=config.NEB_SPRING_K,
              method="improvedtangent")
    neb.interpolate(method="idpp")

    # Pass 1: rough path, climbing image off. Enabling climb before the
    # band has settled is a common cause of oscillation.
    FIRE(neb, logfile="-").run(fmax=0.2, steps=max_steps // 2)

    # Pass 2: fresh optimiser. Climbing image inverts the parallel force
    # on the peak, so pass 1's accumulated velocity now describes a
    # function that no longer exists.
    neb.climb = True
    opt2 = FIRE(neb, trajectory=_path("neb.traj"), logfile="-")
'''

BAND_NEW = '''    start = read(_path("initial.traj"))
    end = read(_path("final.traj"))
'''

CONVERGE_OLD = '''    NEB_FMAX = 0.05
    converged = opt2.run(fmax=NEB_FMAX, steps=max_steps)
'''

CONVERGE_NEW = '''    NEB_FMAX = 0.05
    images, converged = _optimise_band(start, end, n_images, model_key,
                                       with_d3, max_steps, NEB_FMAX)

    # A band can route an adsorbate through the metal. On N2/Ru(0001) step
    # an unconverged band put an N atom below the lower terrace, and the
    # only saddle near its peak belonged to N moving inside the slab. If any
    # image does this, rebuild the band with a hard wall just below the
    # exposed surface. The wall is geometric and adds no energy; it is
    # removed before peak.traj is written, so refinement never sees it.
    inside_first = _images_inside_slab(images)
    wall_z = None
    inside_after = inside_first
    if inside_first:
        wall_z = _exposed_surface_floor(start) - WALL_MARGIN_A
        images, converged = _optimise_band(start, end, n_images, model_key,
                                           with_d3, max_steps, NEB_FMAX,
                                           wall_z=wall_z)
        _remove_wall(images)
        write(_path("neb.traj"), images)
        inside_after = _images_inside_slab(images)
'''

RECORD_OLD = '''        "profile_eV": [float(u) for u in uphill],
    })'''

RECORD_NEW = '''        "profile_eV": [float(u) for u in uphill],
        "images_inside_slab_first_pass": inside_first,
        "wall_applied": wall_z is not None,
        "wall_z_A": wall_z,
        "images_inside_slab_final": inside_after,
    })'''

MESSAGE_OLD = '''    return (f"NEB {status}. Barrier {barrier:.3f} eV, reaction energy "
            f"{reaction_energy:.3f} eV, peak at image {peak} of "
            f"{len(images) - 1}.{capped}")'''

MESSAGE_NEW = '''    wall_note = ""
    if wall_z is not None:
        wall_note = (f" The first band put an adsorbate inside the slab at "
                     f"image(s) {inside_first}, so it was rebuilt with a wall "
                     f"at z = {wall_z:.2f} A. The wall is geometric and adds "
                     f"no energy.")
        if inside_after:
            wall_note += (f" Image(s) {inside_after} are still inside the "
                          f"slab after the retry.")
    return (f"NEB {status}. Barrier {barrier:.3f} eV, reaction energy "
            f"{reaction_energy:.3f} eV, peak at image {peak} of "
            f"{len(images) - 1}.{capped}{wall_note}")'''

BLIND_OLD = '''            barrier = store.get("barrier_eV")
            ref = spec.get("reference_eV")'''

BLIND_NEW = '''            # A barrier only when a connected saddle was found. Otherwise
            # the gas-referenced barrier falls back to the NEB peak held in
            # memory, which can be a band the policy itself rejected: on
            # N2/Ru(0001) step that put 3.611 eV into the results.
            band_barriers = [s["band_barrier_eV"]
                             for s in (store.get("saddle_candidates") or [])]
            if store.get("saddle_chosen_from"):
                barrier = store.get("barrier_eV")
                barrier_note = "connected saddle"
            else:
                barrier = None
                barrier_note = ("no connected saddle was found, so no barrier "
                                "is reported; band barriers are kept for "
                                "information only")
            ref = spec.get("reference_eV")'''

BLIND_RECORD_OLD = '''                "saddle_candidates": store.get("saddle_candidates"),'''
BLIND_RECORD_NEW = '''                "saddle_candidates": store.get("saddle_candidates"),
                "barrier_note": barrier_note,
                "band_barriers_eV": band_barriers,'''


EDITS = [
    (TOOLS, "constraint import", IMPORT_OLD, IMPORT_NEW),
    (TOOLS, "band helpers", HELPERS_OLD, HELPERS_NEW),
    (TOOLS, "band construction", BAND_OLD, BAND_NEW),
    (TOOLS, "band optimisation and retry", CONVERGE_OLD, CONVERGE_NEW),
    (TOOLS, "neb record", RECORD_OLD, RECORD_NEW),
    (TOOLS, "neb message", MESSAGE_OLD, MESSAGE_NEW),
    (BLIND, "barrier only from a connected saddle", BLIND_OLD, BLIND_NEW),
    (BLIND, "record fields", BLIND_RECORD_OLD, BLIND_RECORD_NEW),
]


def main():
    check_only = "--check" in sys.argv
    for p in (TOOLS, BLIND):
        if not p.exists():
            print(f"FAILED: {p} not found. Run from the repo root.")
            return 1
    texts = {p: p.read_text() for p in (TOOLS, BLIND)}
    if "class SurfaceWall" in texts[TOOLS]:
        print("Already patched. Nothing to do.")
        return 0
    if "_images_inside_slab" in texts[TOOLS]:
        print("FAILED: the earlier Hookean spring draft of this patch is "
              "applied. Restore the originals first:")
        print("  cp src/tools.py.pre_band_wall src/tools.py")
        print("  cp scripts/run_blind.py.pre_band_wall scripts/run_blind.py")
        print("then run this again. Nothing written.")
        return 1

    problems = [f"{p}: {name} anchor found {texts[p].count(old)} times"
                for p, name, old, _ in EDITS if texts[p].count(old) != 1]
    if problems:
        print("FAILED, nothing written:")
        for line in problems:
            print(f"  {line}")
        return 1
    if check_only:
        print(f"All {len(EDITS)} anchors found. Patch would apply cleanly.")
        return 0

    for p in (TOOLS, BLIND):
        backup = Path(str(p) + ".pre_band_wall")
        if not backup.exists():
            backup.write_text(texts[p])
    for p, name, old, new in EDITS:
        texts[p] = texts[p].replace(old, new, 1)
    for p in (TOOLS, BLIND):
        p.write_text(texts[p])
    print("Patched src/tools.py (band wall) and scripts/run_blind.py "
          "(no barrier from a rejected band).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
