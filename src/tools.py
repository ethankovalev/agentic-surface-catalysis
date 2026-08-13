"""
Tools the agents can call.

Two rules that matter more than they look.

Docstrings are read by the model to decide what to call and with what
arguments. A vague docstring is a bug, not a style problem.

Failure is a return value, never a swallowed exception. A tool that
quietly returns a number when the optimiser did not converge defeats
the entire point of the validation layer.

Structures are passed between tools as filenames in the working
directory, not as objects. That keeps the interface JSON-serialisable
and leaves a trail on disk you can inspect when something goes wrong.
"""

import sys
from pathlib import Path

import numpy as np
from ase.build import add_adsorbate, fcc100, fcc110, fcc111, hcp0001, molecule
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.optimize import BFGS, FIRE
from langchain_core.tools import tool

try:
    from ase.mep import NEB, NEBTools
except ImportError:
    from ase.neb import NEB, NEBTools

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src import store
from src.calculators import new_calculator


BUILDERS = {
    "111": fcc111,
    "100": fcc100,
    "110": fcc110,
    "0001": hcp0001,
}


def _path(name: str) -> str:
    return str(config.WORK_DIR / name)


def _tag(atoms, n_metal):
    """OC20 convention: 0 = bulk, 1 = surface, 2 = adsorbate.

    The model reads these. Leave the layer numbering fcc111 supplies and
    it sees a system with no adsorbate at all.
    """
    layers = atoms.get_tags()
    top = max(layers[:n_metal]) if n_metal else 0
    tags = [0] * len(atoms)
    for i in range(n_metal):
        if layers[i] == 1:
            tags[i] = 1
    for i in range(n_metal, len(atoms)):
        tags[i] = 2
    atoms.set_tags(tags)
    return atoms


# ======================================================================
# Structure tools
# ======================================================================

@tool
def build_slab(metal: str, facet: str = "111", nx: int = 3, ny: int = 3,
               layers: int = 4, vacuum: float = 10.0,
               dopant: str = "none", n_fixed_layers: int = 2) -> str:
    """Build a clean or doped transition metal slab and save it.

    metal: chemical symbol, e.g. "Cu", "Ni", "Ru", "Pt", "Pd".
    facet: "111", "100", "110" for fcc metals, "0001" for hcp.
    nx, ny, layers: supercell size. 3x3x4 is a reasonable default.
    vacuum: vacuum padding above and below, in Angstrom.
    dopant: chemical symbol to substitute into the top layer, or "none"
            for a clean surface. SBH10 reactions are all clean surfaces.
    n_fixed_layers: bottom layers held fixed to mimic bulk.

    Saves work/slab.traj and returns a short description.
    """
    if facet not in BUILDERS:
        return f"FAILED: unknown facet {facet}. Use one of {list(BUILDERS)}."

    try:
        slab = BUILDERS[facet](metal, size=(nx, ny, layers), vacuum=vacuum)
    except Exception as exc:
        return f"FAILED: could not build {metal}({facet}): {exc}"

    layer_tags = slab.get_tags()
    top = [i for i in range(len(slab)) if layer_tags[i] == 1]
    bottom = [i for i in range(len(slab))
              if layer_tags[i] > layers - n_fixed_layers]

    if dopant.lower() not in ("none", "", "null"):
        for i in (top[0], top[len(top) // 2], top[-1]):
            slab[i].symbol = dopant

    slab.set_constraint(FixAtoms(indices=bottom))
    slab.pbc = True
    write(_path("slab.traj"), slab)

    store.put("slab", {"metal": metal, "facet": facet,
                       "size": [nx, ny, layers], "dopant": dopant,
                       "n_atoms": len(slab)})
    return (f"Built {metal}({facet}) {nx}x{ny}x{layers} slab, {len(slab)} atoms, "
            f"{len(bottom)} fixed. Saved to slab.traj.")


@tool
def place_adsorbate(species: str, height: float = 2.5,
                    site: str = "ontop") -> str:
    """Place a molecule above the slab and save the combined system.

    species: ASE g2-database name, e.g. "H2", "N2", "CH4", "O2", "CO".
    height: clearance in Angstrom between the molecule's lowest atom and
            the highest metal atom. 2.5 Å is a sensible physisorbed start.
            Note this is a true clearance - ASE's add_adsorbate measures
            to atom 0, which this tool corrects for.
    site: "ontop", "bridge", or "hollow". Approximate placement.

    Reads work/slab.traj, saves work/initial.traj.
    """
    slab_file = Path(_path("slab.traj"))
    if not slab_file.exists():
        return "FAILED: no slab.traj. Call build_slab first."

    slab = read(str(slab_file))
    n_metal = len(slab)

    try:
        ads = molecule(species)
    except Exception:
        return (f"FAILED: '{species}' is not in ASE's g2 database. "
                "Use a small molecule name like H2, N2, CH4, CO, O2.")

    top_z = max(slab.positions[:, 2])
    top_atoms = [i for i in range(n_metal)
                 if abs(slab.positions[i, 2] - top_z) < 0.1]
    a = slab.positions[top_atoms[0]]

    if site == "bridge" and len(top_atoms) > 1:
        b = slab.positions[top_atoms[1]]
        x, y = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    elif site == "hollow" and len(top_atoms) > 2:
        b, c = slab.positions[top_atoms[1]], slab.positions[top_atoms[2]]
        x, y = (a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3
    else:
        x, y = a[0], a[1]

    add_adsorbate(slab, ads, height=height, position=(float(x), float(y)))

    # add_adsorbate measures height to atom 0 of the molecule, which may
    # not be its lowest atom. Shift so `height` is a real clearance.
    z_metal = max(slab.positions[:n_metal, 2])
    z_low = min(slab.positions[n_metal:, 2])
    slab.positions[n_metal:, 2] += (z_metal + height) - z_low

    slab.center(axis=2)
    _tag(slab, n_metal)
    write(_path("initial.traj"), slab)

    contact = _closest_contact(slab, n_metal)
    store.put("initial", {"species": species, "height": height,
                          "site": site, "start_contact": contact})
    return (f"Placed {species} at {site}, {height:.2f} Å clearance. "
            f"Closest adsorbate-metal contact {contact:.2f} Å. "
            f"Saved to initial.traj.")


@tool
def build_dissociated_endpoint(separation: float = 3.0,
                               height: float = 1.5) -> str:
    """Build the dissociated final state by pulling the molecule apart.

    Splits the adsorbate into two fragments along its longest internal
    bond, slides them apart in the surface plane, and lowers both onto
    the metal so they can chemisorb.

    separation: target distance between the two fragment centres, in
                Angstrom. Keep it well under the in-plane cell vector or
                each fragment meets its own periodic image.
    height: height of each fragment above the surface plane. About 1.5 Å
            for atomic H, nearer 2.1 Å for a carbon-containing fragment.

    Reads work/initial.traj, saves work/final.traj.
    """
    init_file = Path(_path("initial.traj"))
    if not init_file.exists():
        return "FAILED: no initial.traj. Call place_adsorbate first."

    atoms = read(str(init_file))
    tags = atoms.get_tags()
    ads = [i for i in range(len(atoms)) if tags[i] == 2]
    metal = [i for i in range(len(atoms)) if tags[i] != 2]

    if len(ads) < 2:
        return "FAILED: fewer than two adsorbate atoms; nothing to dissociate."

    # Longest internal distance defines the bond being broken.
    best, pair = -1.0, (ads[0], ads[1])
    for i in ads:
        for j in ads:
            if i < j:
                d = atoms.get_distance(i, j, mic=True)
                if d > best:
                    best, pair = d, (i, j)
    a, b = pair

    left = [a] + [k for k in ads if k not in (a, b)
                  and atoms.get_distance(k, a, mic=True)
                  < atoms.get_distance(k, b, mic=True)]
    right = [k for k in ads if k not in left]

    direction = atoms.positions[b] - atoms.positions[a]
    direction[2] = 0.0
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = direction / norm

    shift = (separation - best) / 2.0
    atoms.positions[left] -= direction * shift
    atoms.positions[right] += direction * shift

    surface_z = max(atoms.positions[i, 2] for i in metal)
    for group in (left, right):
        lowest = min(atoms.positions[i, 2] for i in group)
        atoms.positions[group, 2] += (surface_z + height) - lowest

    write(_path("final.traj"), atoms)
    store.put("final", {"separation": separation, "height": height,
                        "fragments": [len(left), len(right)]})
    return (f"Built dissociated endpoint: fragments of {len(left)} and "
            f"{len(right)} atoms, {separation:.2f} Å apart, {height:.2f} Å "
            f"above the surface. Saved to final.traj.")


# ======================================================================
# Simulation tools
# ======================================================================

@tool
def relax_structure(structure: str, with_d3: bool = True,
                    fmax: float = 0.02, max_steps: int = 300) -> str:
    """Relax a saved structure to its nearest local minimum.

    structure: "initial" or "final".
    with_d3: include Grimme D3 dispersion in the relaxation loop. OC20
             is RPBE, which has no dispersion term, so leaving this off
             gives physically wrong geometries for weakly bound species
             that still converge cleanly.
    fmax: force convergence threshold in eV/A.

    Overwrites the structure file with the relaxed geometry. Reports the
    energy AND whether it converged - a non-converged energy is not a
    minimum and must not be used.
    """
    src = Path(_path(f"{structure}.traj"))
    if not src.exists():
        return f"FAILED: no {structure}.traj. Build it first."

    atoms = read(str(src))
    atoms.calc = new_calculator(with_d3=with_d3)

    opt = BFGS(atoms, logfile="-")
    converged = opt.run(fmax=fmax, steps=max_steps)
    energy = atoms.get_potential_energy()

    write(str(src), atoms)

    tags = atoms.get_tags()
    n_metal = sum(1 for t in tags if t != 2)
    contact = _closest_contact(atoms, n_metal)

    store.put(f"{structure}_relaxed", {
        "energy_eV": float(energy),
        "converged": bool(converged),
        "with_d3": bool(with_d3),
        "fmax_target": fmax,
        "closest_contact": contact,
    })

    status = "converged" if converged else "DID NOT CONVERGE"
    return (f"Relaxed {structure} ({status}). Energy {energy:.4f} eV, "
            f"closest adsorbate-metal contact {contact:.2f} Å, "
            f"D3 {'on' if with_d3 else 'OFF'}.")


@tool
def run_neb(n_images: int = 10, with_d3: bool = True,
            fmax: float = 0.10, max_steps: int = 400) -> str:
    """Find the transition state between the relaxed endpoints.

    Runs a two-pass climbing-image NEB with the FIRE optimiser. NEB
    forces are not the gradient of any single scalar function, so
    quasi-Newton methods build a Hessian from a false premise and go
    unstable near convergence - this is why FIRE, not BFGS.

    n_images: intermediate images. The band has n_images + 2 in total.
              Raise this if the energy profile shows a sharp spike
              between two neighbouring images.
    with_d3: match whatever the endpoint relaxations used.

    Reads initial.traj and final.traj, saves neb.traj. Reports the
    barrier AND whether the band converged.
    """
    for name in ("initial", "final"):
        if not Path(_path(f"{name}.traj")).exists():
            return f"FAILED: no {name}.traj. Relax both endpoints first."

    start = read(_path("initial.traj"))
    end = read(_path("final.traj"))

    images = [start]
    for _ in range(n_images):
        img = start.copy()
        img.calc = new_calculator(with_d3=with_d3)
        images.append(img)
    images.append(end)

    start.calc = new_calculator(with_d3=with_d3)
    end.calc = new_calculator(with_d3=with_d3)

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
    converged = opt2.run(fmax=fmax, steps=max_steps)

    barrier, reaction_energy = NEBTools(images).get_barrier()
    energies = [img.get_potential_energy() for img in images]
    uphill = [e - energies[0] for e in energies]
    peak = int(np.argmax(uphill))

    store.put("neb", {
        "barrier_eV": float(barrier),
        "reaction_energy_eV": float(reaction_energy),
        "converged": bool(converged),
        "peak_image": peak,
        "n_images": len(images),
        "profile_eV": [float(u) for u in uphill],
    })
    store.put("barrier_eV", float(barrier))

    status = "converged" if converged else "DID NOT CONVERGE"
    return (f"NEB {status}. Barrier {barrier:.3f} eV, reaction energy "
            f"{reaction_energy:.3f} eV, peak at image {peak} of "
            f"{len(images) - 1}.")


@tool
def read_results() -> str:
    """Report everything computed so far in this run.

    Use this to see the current state before deciding what to do next.
    """
    snap = store.snapshot()
    if not snap:
        return "Nothing computed yet."
    lines = [f"{k}: {v}" for k, v in snap.items() if k != "validation_detail"]
    return "\n".join(lines)


# ======================================================================
# Validation tools
# ======================================================================

@tool
def check_convergence() -> str:
    """Check that every calculation in this run actually converged.

    An energy from a non-converged optimisation is not a minimum and a
    barrier from a non-converged band is not a saddle point. Neither is
    usable, however plausible the number looks.
    """
    failures = []
    for key in ("initial_relaxed", "final_relaxed", "neb"):
        record = store.get(key)
        if record is None:
            failures.append(f"{key} was never run")
        elif not record.get("converged"):
            failures.append(f"{key} did not converge")

    passed = not failures
    detail = "all converged" if passed else "; ".join(failures)
    store.record_check("convergence", passed, detail)
    return f"convergence: {'PASS' if passed else 'FAIL'} - {detail}"


@tool
def check_noise_floor() -> str:
    """Check the barrier is larger than the model's own error bar.

    UMA's benchmarked MAE against its reference DFT is about 0.009 eV.
    A result smaller than that is not distinguishable from zero, no
    matter how tightly the optimiser converged.
    """
    barrier = store.get("barrier_eV")
    if barrier is None:
        store.record_check("noise_floor", False, "no barrier computed")
        return "noise_floor: FAIL - no barrier computed"

    passed = abs(barrier) > config.NOISE_FLOOR_EV
    detail = (f"barrier {barrier:.4f} eV vs noise floor "
              f"{config.NOISE_FLOOR_EV} eV")
    store.record_check("noise_floor", passed, detail)
    return f"noise_floor: {'PASS' if passed else 'FAIL'} - {detail}"


@tool
def check_dispersion_relevance() -> str:
    """Check whether a missing dispersion term explains the geometry.

    OC20 is trained on RPBE, which has no dispersion term. For weakly
    bound species this produces a structure that drifts away from the
    surface while still converging cleanly. If the adsorbate sits far
    out AND D3 was off, that is the likely cause and the fix is to relax
    again with with_d3 set to true.
    """
    record = store.get("initial_relaxed")
    if record is None:
        store.record_check("dispersion", False, "initial never relaxed")
        return "dispersion: FAIL - initial never relaxed"

    contact = record.get("closest_contact", 99.0)
    with_d3 = record.get("with_d3", False)

    too_far = contact > config.MAX_PHYSISORPTION_HEIGHT
    passed = not (too_far and not with_d3)

    if passed and too_far:
        detail = (f"contact {contact:.2f} Å is large but D3 was on - "
                  "may genuinely be unbound")
    elif passed:
        detail = f"contact {contact:.2f} Å, D3 {'on' if with_d3 else 'off'}"
    else:
        detail = (f"contact {contact:.2f} Å with D3 OFF - rerun the "
                  "relaxation with with_d3=true")

    store.record_check("dispersion", passed, detail)
    return f"dispersion: {'PASS' if passed else 'FAIL'} - {detail}"


@tool
def check_geometry() -> str:
    """Check the geometry is physically sensible.

    Catches atoms driven into the surface, and a transition state whose
    peak sits at one end of the band rather than in the middle - which
    means there is no barrier between the endpoints at all.
    """
    problems = []

    for key in ("initial_relaxed", "final_relaxed"):
        record = store.get(key)
        if record and record.get("closest_contact", 99) < config.MIN_CONTACT:
            problems.append(
                f"{key} contact {record['closest_contact']:.2f} Å is inside "
                "a bond length")

    neb = store.get("neb")
    if neb:
        peak, n = neb.get("peak_image", 0), neb.get("n_images", 1)
        if peak in (0, n - 1):
            problems.append(
                f"barrier peak is at endpoint image {peak}, so there is no "
                "hill between the endpoints")

    passed = not problems
    detail = "geometry sensible" if passed else "; ".join(problems)
    store.record_check("geometry", passed, detail)
    return f"geometry: {'PASS' if passed else 'FAIL'} - {detail}"


@tool
def check_barrier_magnitude() -> str:
    """Check the barrier is in a physically plausible range.

    Dissociative chemisorption barriers on transition metals are
    typically 0.1 to 2.5 eV. A near-zero barrier usually means the
    endpoints are not genuinely distinct; a very large one usually means
    the dissociated endpoint is badly constructed.
    """
    barrier = store.get("barrier_eV")
    if barrier is None:
        store.record_check("magnitude", False, "no barrier computed")
        return "magnitude: FAIL - no barrier computed"

    passed = 0.05 < barrier < 3.0
    if barrier <= 0.05:
        detail = f"{barrier:.3f} eV is implausibly small - check endpoints differ"
    elif barrier >= 3.0:
        detail = f"{barrier:.3f} eV is implausibly large - check final state"
    else:
        detail = f"{barrier:.3f} eV is plausible"

    store.record_check("magnitude", passed, detail)
    return f"magnitude: {'PASS' if passed else 'FAIL'} - {detail}"


@tool
def validation_summary() -> str:
    """Report which checks have run and which passed.

    Call this after running the individual checks. The run cannot finish
    until every check has been run and every one has passed.
    """
    checks = store.validation()
    if not checks:
        return "No checks have been run yet."

    detail = store.get("validation_detail", {})
    lines = [f"{'PASS' if ok else 'FAIL'}  {name}: {detail.get(name, '')}"
             for name, ok in checks.items()]
    lines.append("")
    lines.append("ALL PASSED" if store.all_checks_passed()
                 else "NOT ALL CHECKS PASSED - the run is not finished")
    return "\n".join(lines)


# ======================================================================
# Helpers
# ======================================================================

def _closest_contact(atoms, n_metal: int) -> float:
    """Shortest distance from any adsorbate atom to any metal atom."""
    if n_metal >= len(atoms):
        return 99.0
    metal = list(range(n_metal))
    best = 99.0
    for i in range(n_metal, len(atoms)):
        d = atoms.get_distances(i, metal, mic=True).min()
        best = min(best, float(d))
    return best


STRUCTURE_TOOLS = [build_slab, place_adsorbate, build_dissociated_endpoint]
SIMULATION_TOOLS = [relax_structure, run_neb, read_results]
VALIDATION_TOOLS = [
    check_convergence,
    check_noise_floor,
    check_dispersion_relevance,
    check_geometry,
    check_barrier_magnitude,
    validation_summary,
]
