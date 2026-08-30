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
from ase.data import atomic_numbers, covalent_radii
from ase.constraints import FixAtoms
from ase.neighborlist import natural_cutoffs, NeighborList
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
    tags = [0] * len(atoms)
    for i in range(n_metal):
        if layers[i] == 1:
            tags[i] = 1
    for i in range(n_metal, len(atoms)):
        tags[i] = 2
    atoms.set_tags(tags)
    return atoms



# Structure tools

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
def build_stepped_slab(metal: str, facet: str = "111", nx: int = 6, ny: int = 3,
                       layers: int = 5, vacuum: float = 10.0,
                       terrace_fraction: float = 0.5,
                       n_fixed_layers: int = 2) -> str:
    """Build a slab with a step edge, for the step-site SBH10 reactions.

    Two of the ten reactions (N2 on Ru(0001) and CH4 on Ni(111)) are
    measured at step sites, where under-coordinated edge atoms lower the
    barrier substantially - N2 on Ru drops from 1.84 eV on the terrace to
    0.40 eV at a step. Running those on a flat slab gives a plausible
    number for the wrong surface, and no validation check catches it.

    The step is made by removing part of the top layer, exposing a lower
    terrace and leaving an under-coordinated edge row. This is a step, but
    it is not a canonical Miller-index stepped surface such as fcc(211);
    terrace width and edge geometry differ, so treat it as an approximate
    step model rather than an exact match to a specific stepped facet.

    metal: chemical symbol, e.g. "Ni", "Ru".
    facet: the terrace facet - "111" for fcc, "0001" for hcp.
    nx: rows along the step direction. Must be 6 or more so each terrace
        is at least three rows wide; narrower and the adsorbate interacts
        with both edges through the periodic image.
    layers: 4 or more, since the top two become surface.
    terrace_fraction: fraction of the cell keeping its top layer. 0.5
        gives two equal terraces.
    n_fixed_layers: bottom layers held fixed to mimic bulk.

    Saves work/slab.traj and records the step-edge atom indices, which
    place_adsorbate(site="step") reads back. If no under-coordinated
    atoms are found, no step was built and the tool fails rather than
    returning a slab that would quietly give terrace barriers.
    """
    if facet not in BUILDERS:
        return f"FAILED: unknown facet {facet}. Use one of {list(BUILDERS)}."
    if nx < 6:
        return (f"FAILED: nx={nx} is too small for a stepped surface. Each "
                "terrace needs at least three atomic rows, so nx must be 6+.")
    if layers < 4:
        return f"FAILED: layers={layers} is too few; use 4 or more."

    try:
        slab = BUILDERS[facet](metal, size=(nx, ny, layers), vacuum=vacuum)
    except Exception as exc:
        return f"FAILED: could not build {metal}({facet}): {exc}"

    layer_of = slab.get_tags()

    # Measure what a flat terrace atom's coordination is for THIS metal,
    # before carving anything. Hardcoding a number does not work: the
    # neighbour cutoff scales with covalent radius, so it reaches the
    # second shell for Ru (12 neighbours on a flat surface) but not for
    # Ni (9). A fixed threshold silently reports "no step" on a slab
    # that has one.
    flat_coordination = _terrace_coordination(slab)

    x_cut = terrace_fraction * slab.cell[0, 0]
    remove = [i for i in range(len(slab))
              if layer_of[i] == 1 and slab.positions[i, 0] >= x_cut - 1e-6]
    if not remove:
        return "FAILED: no top-layer atoms fell in the removal region."

    upper_terrace = sum(1 for i in range(len(slab)) if layer_of[i] == 1) - len(remove)
    del slab[remove]

    layer_of = slab.get_tags()
    bottom = [i for i in range(len(slab))
              if layer_of[i] > layers - n_fixed_layers]

    # Both terraces are surface. place_adsorbate's _tag maps layer tag 1
    # to OC20 surface tag 1, so the exposed second layer must be marked
    # as surface too or the model sees the lower terrace as bulk.
    slab.set_tags([1 if layer_of[i] <= 2 else 2 for i in range(len(slab))])

    slab.set_constraint(FixAtoms(indices=bottom))
    slab.pbc = True
    write(_path("slab.traj"), slab)

    edge = _step_edge_atoms(slab, flat_coordination)

    store.put("slab", {"metal": metal, "facet": facet,
                       "size": [nx, ny, layers], "dopant": "none",
                       "n_atoms": len(slab), "stepped": True,
                       "step_edge_atoms": edge,
                       "terrace_coordination": flat_coordination})

    if not edge:
        return ("FAILED: the carve removed atoms but produced no "
                "under-coordinated surface atoms, so there is no step edge. "
                "Do not use this slab.")

    return (f"Built stepped {metal}({facet}), {len(slab)} atoms: upper terrace "
            f"{upper_terrace} atoms, {len(remove)} removed to expose the lower "
            f"terrace, {len(edge)} under-coordinated step-edge atoms "
            f"(flat terrace coordination is {flat_coordination}), "
            f"{len(bottom)} fixed. Saved to slab.traj.")


def _top_layer(atoms, metal_indices, tol=0.5):
    """Indices of the metal atoms in the topmost surface layer."""
    zmax = max(atoms.positions[i, 2] for i in metal_indices)
    return [i for i in metal_indices if atoms.positions[i, 2] > zmax - tol]


def _hollow_sites(atoms, metal_indices, n_grid=64):
    """Lateral positions locally furthest from any top-layer metal atom.

    A hollow is the point on the surface with the greatest clearance from
    the surrounding metal atoms, so locating hollows as local maxima of the
    distance field works for fcc(111), fcc(100) and hcp(0001) without the
    code knowing which facet it is looking at. Checked against 3x3 cells:
    18 sites on fcc(111) and hcp(0001) (the fcc and hcp hollows), 9 on
    fcc(100) (the four-fold hollows).

    Returns a list of xy positions. Note this cannot distinguish an fcc
    hollow from an hcp one; they are both returned and the caller takes
    whichever is nearest.
    """
    top = _top_layer(atoms, metal_indices)
    cell = np.array(atoms.cell[:2, :2], dtype=float)
    top_xy = atoms.positions[top][:, :2]
    inv = np.linalg.inv(cell)

    us = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    frac = np.array([[u, v] for u in us for v in us])
    grid_xy = frac @ cell
    shifts = np.array([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1)],
                      dtype=float) @ cell
    images = (top_xy[:, None, :] + shifts[None, :, :]).reshape(-1, 2)
    d = np.linalg.norm(grid_xy[:, None, :] - images[None, :, :], axis=2)
    clearance = d.min(axis=1).reshape(n_grid, n_grid)

    dm = np.linalg.norm(top_xy[:, None, :] - images[None, :, :], axis=2)
    dm[dm < 1e-6] = np.inf
    nn = dm.min()

    cmax = clearance.max()
    peaks = []
    for i in range(n_grid):
        for j in range(n_grid):
            c = clearance[i, j]
            if c < 0.9 * cmax:
                continue
            neigh = [clearance[(i + di) % n_grid, (j + dj) % n_grid]
                     for di in (-1, 0, 1) for dj in (-1, 0, 1)
                     if (di, dj) != (0, 0)]
            if c >= max(neigh) - 1e-9:
                peaks.append(grid_xy[i * n_grid + j])

    merged = []
    for xy in peaks:
        for k, (mxy, n) in enumerate(merged):
            df = (xy - mxy) @ inv
            df -= np.round(df)
            if np.linalg.norm(df @ cell) < 0.3 * nn:
                merged[k] = ((mxy * n + xy) / (n + 1), n + 1)
                break
        else:
            merged.append((xy, 1))
    return [xy for xy, _ in merged]


def _coordination(slab):
    """Neighbour count for every atom, using covalent-radius cutoffs."""
    cutoffs = natural_cutoffs(slab, mult=1.15)
    neighbours = NeighborList(cutoffs, self_interaction=False, bothways=True)
    neighbours.update(slab)
    return [len(neighbours.get_neighbors(i)[0]) for i in range(len(slab))]


def _terrace_coordination(flat_slab) -> int:
    """The neighbour count of a top-layer atom on the uncarved slab."""
    counts = _coordination(flat_slab)
    layer_of = flat_slab.get_tags()
    top = [c for i, c in enumerate(counts) if layer_of[i] == 1]
    return int(max(set(top), key=top.count))      # the most common value


def _step_edge_atoms(slab, reference_coordination: int) -> list:
    """Indices of surface atoms less coordinated than a flat terrace atom.

    reference_coordination is measured once, before carving, and reused
    by every caller - this function when build_stepped_slab reports the
    step, and place_adsorbate when it sites the molecule. One
    measurement, one source of truth: the two tools can never disagree
    about which atoms count as the edge.

    If this returns empty, no step was created and a reaction placed
    with site="step" would silently run on a terrace instead. Nothing
    downstream catches that: a terrace barrier is a plausible number.
    """
    counts = _coordination(slab)
    tags = slab.get_tags()
    return [i for i in range(len(slab))
            if tags[i] == 1 and counts[i] < reference_coordination]

@tool
def place_adsorbate(species: str, height: float = 2.5,
                    site: str = "ontop", overhang: float = 0.5) -> str:
    """Place a molecule above the slab and save the combined system.

    species: ASE g2-database name, e.g. "H2", "N2", "CH4", "O2", "CO".
    height: clearance in Angstrom between the molecule's lowest atom and
            the metal atom it sits above. 2.5 Å is a sensible physisorbed
            start. This is a true clearance - ASE's add_adsorbate measures
            to atom 0, which this tool corrects for.
    site: "ontop", "bridge", or "hollow" for a terrace site, or "step" to
          place at an under-coordinated step edge. Use "step" for the two
          SBH10 step reactions (N2 on Ru(0001), CH4 on Ni(111)); placing
          those on a terrace computes the terrace barrier while the record
          claims a step, and no validation check catches it.
    overhang: for site="step" only, how far toward the lower terrace the
              molecule sits from directly above the edge atom, in Angstrom.

    site="step" requires a slab from build_stepped_slab, which records
    which atoms form the edge. Reads work/slab.traj, saves
    work/initial.traj.
    """
    if site not in ("ontop", "bridge", "hollow", "step"):
        return (f"FAILED: unknown site '{site}'. Use ontop, bridge, hollow "
                "or step. An unrecognised name must not fall back to a "
                "default - that places the molecule somewhere other than "
                "where it was asked for and reports success.")

    slab_file = Path(_path("slab.traj"))
    if not slab_file.exists():
        return "FAILED: no slab.traj. Call build_slab or build_stepped_slab first."

    slab = read(str(slab_file))
    n_metal = len(slab)

    try:
        ads = molecule(species)
    except Exception:
        return (f"FAILED: '{species}' is not in ASE's g2 database. "
                "Use a small molecule name like H2, N2, CH4, CO, O2.")

    note = ""

    if site == "step":
        slab_record = store.get("slab") or {}
        edge = slab_record.get("step_edge_atoms")
        flat_cn = slab_record.get("terrace_coordination")

        if not edge:
            return ("FAILED: no step edge recorded for this slab. Use "
                    "build_stepped_slab, not build_slab, before calling "
                    "place_adsorbate with site='step'.")

        counts = _coordination(slab)

        # Pick the edge atom nearest the middle in y, so the molecule sits
        # as far as possible from its own periodic images along the step.
        ys = slab.positions[edge, 1]
        chosen = edge[int(np.argmin(np.abs(ys - ys.mean())))]
        anchor = slab.positions[chosen].copy()
        edge_cn = int(counts[chosen])

        # Which side is the lower terrace? Compare the highest surface atom
        # either side of the chosen atom in x.
        right = slab.positions[:n_metal, 0] > anchor[0]
        z_right = slab.positions[:n_metal][right, 2].max() if right.any() else -1e9
        z_left = slab.positions[:n_metal][~right, 2].max() if (~right).any() else -1e9
        step_dir = 1.0 if z_right < z_left else -1.0

        x, y = anchor[0] + step_dir * overhang, anchor[1]

        # The reference for `height` is the EDGE atom, not the highest atom
        # in the cell - the upper terrace is above the edge, so measuring
        # from the cell maximum would float the molecule too high.
        z_ref = anchor[2]

        note = (f" Edge atom {chosen} has coordination {edge_cn} against a "
                f"{flat_cn}-coordinate flat terrace; {len(edge)} atoms share "
                f"the edge.")
        if len(edge) < 2:
            note += (" Only one such atom - that may be a corner artefact "
                     "rather than an edge row; inspect slab.traj.")
    else:
        top_z = max(slab.positions[:n_metal, 2])
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
        z_ref = top_z

    add_adsorbate(slab, ads, height=height, position=(float(x), float(y)))

    # add_adsorbate measures height to atom 0 of the molecule, which may
    # not be its lowest atom. Shift so `height` is a real clearance above
    # the reference atom.
    z_low = min(slab.positions[n_metal:, 2])
    slab.positions[n_metal:, 2] += (z_ref + height) - z_low

    slab.center(axis=2)
    _tag(slab, n_metal)
    write(_path("initial.traj"), slab)

    contact = _closest_contact(slab, n_metal)
    store.put("initial", {"species": species, "height": height,
                          "site": site, "start_contact": contact,
                          "stepped_site": site == "step"})
    return (f"Placed {species} at {site}, {height:.2f} Å clearance. "
            f"Closest adsorbate-metal contact {contact:.2f} Å.{note} "
            f"Saved to initial.traj.")

@tool
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

    r_metal = covalent_radii[atomic_numbers[atoms[metal[0]].symbol]]
    r_ads = sum(covalent_radii[atoms[i].number] for i in ads) / len(ads)

    if separation is None:
        separation = 2.5 * (r_metal + r_ads)

    # Two hydrogens on opposite sides of a carbon sit further apart than
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

    # Put each fragment over a hollow, not wherever the lateral separation
    # happened to leave it. Previously only z was set, so a fragment could
    # land atop a surface atom and relax into that minimum: two N atoms on
    # Ru(0001) came out 2.07 eV ABOVE the intact molecule, which made the
    # NEB unconvergeable because the endpoint itself was wrong.
    surface_z = max(atoms.positions[i, 2] for i in metal)
    sites = _hollow_sites(atoms, metal)
    used = []
    heights = []

    for group in (left, right):
        # the largest atom in the fragment is the one that binds, so it is
        # the anchor and its own radius sets the height. Averaging radii
        # across the whole adsorbate put a CH3 carbon at a hydrogen height.
        anchor = max(group, key=lambda k: covalent_radii[atoms[k].number])
        r_group = covalent_radii[atoms[anchor].number]
        h = height if height is not None else r_metal + r_group
        heights.append(h)

        if sites:
            free = [s for n, s in enumerate(sites) if n not in used]
            if free:
                anchor_xy = atoms.positions[anchor, :2]
                best_site = min(free,
                                key=lambda s: np.linalg.norm(s - anchor_xy))
                used.append(sites.index(best_site))
                atoms.positions[group, 0] += best_site[0] - anchor_xy[0]
                atoms.positions[group, 1] += best_site[1] - anchor_xy[1]

        lowest = min(atoms.positions[i, 2] for i in group)
        atoms.positions[group, 2] += (surface_z + h) - lowest

    height = heights

    write(_path("final.traj"), atoms)

    # record what the fragments actually are, so a check can tell CH3 + H from CH2 + H2.
    left_symbols = sorted(atoms[i].symbol for i in left)
    right_symbols = sorted(atoms[i].symbol for i in right)

    store.put("final", {"separation": separation,
                        "height": [round(h, 3) for h in height],
                        "fragments": [len(left), len(right)],
                        "broken_bond": f"{atoms[a].symbol}-{atoms[b].symbol}",
                        "fragment_symbols": [left_symbols, right_symbols]})
    return (f"Built dissociated endpoint: broke the "
            f"{atoms[a].symbol}-{atoms[b].symbol} bond into fragments of "
            f"{len(left)} and {len(right)} atoms, {separation:.2f} Å apart, "
            f"at {height[0]:.2f} and {height[1]:.2f} Å above the surface, "
            f"each over a hollow site. Saved to final.traj.")


# Simulation tools

@tool
def relax_structure(structure: str, model_key: str = None, with_d3: bool = True,
                    fmax: float = 0.02, max_steps: int = 300) -> str:
    """Relax a saved structure to its nearest local minimum.

    structure: "initial", "final", or "gasref".
    with_d3: include Grimme D3 dispersion in the relaxation loop. OC20
             is RPBE, which has no dispersion term, so leaving this off
             gives physically wrong geometries for weakly bound species
             that still converge cleanly.
    fmax: force convergence threshold in eV/Å.

    Overwrites the structure file with the relaxed geometry. Reports the
    energy AND whether it converged, a non-converged energy is not a
    minimum and must not be used.
    """
    src = Path(_path(f"{structure}.traj"))
    if not src.exists():
        return f"FAILED: no {structure}.traj. Build it first."

    model_key = model_key or config.DEFAULT_MODEL
    store.put("model_key", model_key)

    atoms = read(str(src))
    atoms.calc = new_calculator(model_key, with_d3=with_d3)

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
def run_neb(n_images: int = 10, model_key: str = None, with_d3: bool = True,
            max_steps: int = 400) -> str:
    """Find the transition state between the relaxed endpoints.

    Runs a two-pass climbing-image NEB with the FIRE optimiser. NEB
    forces are not the gradient of any single scalar function, so
    quasi-Newton methods build a Hessian from a false premise and go
    unstable near convergence, this is why FIRE, not BFGS.

    n_images: intermediate images. The band has n_images + 2 in total.
              Capped at 24 - GPU memory scales linearly with this and a
              24 GB card runs out beyond that.
              Raise this if the energy profile shows a sharp spike
              between two neighbouring images.
    with_d3: match whatever the endpoint relaxations used.

    Reads initial.traj and final.traj, saves neb.traj. Reports the
    barrier AND whether the band converged.
    """
    for name in ("initial", "final"):
        if not Path(_path(f"{name}.traj")).exists():
            return f"FAILED: no {name}.traj. Relax both endpoints first."

    model_key = model_key or config.DEFAULT_MODEL

    # Each image carries its own calculator, so GPU memory scales linearly
    # with n_images. At 32 images a 24 GB card is already exhausted. Cap it
    # rather than let the agent escalate into an out-of-memory crash.
    MAX_IMAGES = 24
    requested_images = n_images
    if n_images > MAX_IMAGES:
        n_images = MAX_IMAGES

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    store.put("model_key", model_key)

    start = read(_path("initial.traj"))
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

    # Pinned, deliberately not a tool argument. Every reaction in the
    # benchmark must converge to the same tolerance or the barriers are not
    # comparable across the grid. This was previously agent-settable with a
    # default of 0.10, so a run could use any tolerance without that being
    # obvious in the output. Recorded in the store below so every result
    # carries the tolerance it was computed at.
    NEB_FMAX = 0.05
    converged = opt2.run(fmax=NEB_FMAX, steps=max_steps)

    # get_barrier() defaults to fit=True, which returns the peak of a spline
    # fitted through the images rather than any computed image. On an
    # under-resolved band that spline overshoots: for N2_Ru0001_step it
    # reported 1.618 eV when the highest actual image was 1.323 eV. Use
    # fit=False so the barrier is always a real computed energy. The band
    # being coarse is a separate problem, flagged by the path_resolved
    # check; this stops it being papered over with an interpolated number.
    _, reaction_energy = NEBTools(images).get_barrier(fit=False)
    energies = [img.get_potential_energy() for img in images]
    uphill = [e - energies[0] for e in energies]
    peak = int(np.argmax(uphill))
    barrier = float(uphill[peak])

    store.put("neb", {
        "barrier_eV": float(barrier),
        "reaction_energy_eV": float(reaction_energy),
        "converged": bool(converged),
        "with_d3": bool(with_d3),          # CHANGED: needed for the
                                           # dispersion consistency check
        "peak_image": peak,
        "n_images": len(images),
        "fmax_target": NEB_FMAX,
        "profile_eV": [float(u) for u in uphill],
    })
    store.put("barrier_eV", float(barrier))

    status = "converged" if converged else "DID NOT CONVERGE"
    capped = (f" Requested {requested_images} images but capped at "
              f"{MAX_IMAGES} - asking for more will not change this."
              if requested_images > MAX_IMAGES else "")
    return (f"NEB {status}. Barrier {barrier:.3f} eV, reaction energy "
            f"{reaction_energy:.3f} eV, peak at image {peak} of "
            f"{len(images) - 1}.{capped}")


@tool
def build_gas_reference(height: float = 8.0) -> str:
    """Build the gas-phase reference: same system, molecule far away.

    SBH10 barriers are referenced to a free molecule, not to a
    physisorbed one. The OC20 task is not trained on isolated
    molecules, so instead of removing the slab we lift the molecule
    clear of it. The slab contribution then cancels in the difference,
    and the model only ever sees a slab plus adsorbate.

    Reads work/initial.traj, saves work/gasref.traj.
    """
    src = Path(_path("initial.traj"))
    if not src.exists():
        return "FAILED: no initial.traj. Call place_adsorbate first."

    atoms = read(str(src))
    tags = atoms.get_tags()
    n_metal = sum(1 for t in tags if t != 2)

    z_metal = max(atoms.positions[:n_metal, 2])
    z_low = min(atoms.positions[n_metal:, 2])
    atoms.positions[n_metal:, 2] += (z_metal + height) - z_low

    # The cell must be tall enough that the lifted molecule does not
    # meet the slab's periodic image from above.
    cell = atoms.get_cell()
    if cell[2, 2] < z_metal + height + 10.0:
        cell[2, 2] = z_metal + height + 10.0
        atoms.set_cell(cell)

    write(_path("gasref.traj"), atoms)
    store.put("gasref", {"height": height})
    return (f"Built gas-phase reference with the molecule {height:.1f} Å "
            f"above the surface. Saved to gasref.traj. Relax it, then "
            f"call compute_gas_referenced_barrier.")


@tool
def compute_gas_referenced_barrier() -> str:
    """Convert the NEB barrier to a gas-phase reference.

    The NEB measures from the physisorbed state. Experiment measures
    from a free molecule. The difference is the physisorption well
    depth, which this reports as a diagnostic - if it is a few meV the
    two conventions agree and the distinction does not matter for this
    system.
    """
    neb = store.get("neb")
    initial = store.get("initial_relaxed")
    gasref = store.get("gasref_relaxed")

    if not all((neb, initial, gasref)):
        missing = [n for n, v in (("neb", neb), ("initial", initial),
                                  ("gasref", gasref)) if not v]
        return f"FAILED: missing {missing}. Relax the gas reference first."

    well_depth = gasref["energy_eV"] - initial["energy_eV"]
    # Subtract, not add. well_depth > 0 means the physisorbed minimum sits
    # BELOW the free molecule, so a trajectory starting from gas is already
    # part-way up the hill: the gas-referenced barrier must be SMALLER than
    # the barrier measured from the physisorbed state, never larger.
    barrier_gas = neb["barrier_eV"] - well_depth

    store.put("well_depth_eV", float(well_depth))
    store.put("barrier_gas_eV", float(barrier_gas))
    store.put("barrier_eV", float(barrier_gas))   # this is what gets scored

    return (f"Physisorption well depth {well_depth:.3f} eV. "
            f"Barrier from physisorbed state {neb['barrier_eV']:.3f} eV, "
            f"from gas phase {barrier_gas:.3f} eV.")


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

# Validation tools

@tool
def check_convergence() -> str:
    """Check that every calculation in this run actually converged.

    An energy from a non-converged optimisation is not a minimum and a
    barrier from a non-converged band is not a saddle point. Neither is
    usable, however plausible the number looks.
    """
    failures = []
    for key in ("initial_relaxed", "final_relaxed", "gasref_relaxed", "neb"):
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
    """Report whether the barrier is above the model's own error bar.

    UMA's benchmarked MAE against reference DFT is roughly 0.1-0.3 eV
    for adsorption energies. A barrier is a difference of two energies,
    so errors partly cancel and this is an upper-bound estimate rather
    than a strict error bar on the barrier itself.

    This check is informational, not a gate. Two of the SBH10 reactions
    are genuinely non-activated, so a barrier inside the error bar can
    be the correct answer. Whether it is a real non-activated reaction
    or a failed calculation is what check_endpoints_distinct decides.
    """
    barrier = store.get("barrier_eV")
    if barrier is None:
        store.record_check("noise_floor", False, "no barrier computed")
        return "noise_floor: FAIL (no barrier computed)"

    model_key = store.get("model_key", config.DEFAULT_MODEL)
    noise_floor = config.MODELS[model_key]["noise_floor_eV"]

    resolvable = abs(barrier) > noise_floor
    if resolvable:
        detail = (f"[{model_key}] barrier {barrier:.3f} eV is above the "
                  f"{noise_floor} eV noise floor")
    else:
        detail = (f"[{model_key}] barrier {barrier:.3f} eV is within the "
                  f"model's own error bar ({noise_floor} eV) - consistent "
                  "with a non-activated reaction, but not resolvable from zero")

    # Passes either way. The number being small is a finding, not a failure.
    store.record_check("noise_floor", True, detail)
    return f"noise_floor: PASS - {detail}"


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
        detail = (f"contact {contact:.2f} Å is large but D3 was on: "
                  "may genuinely be unbound")
    elif passed:
        detail = f"contact {contact:.2f} Å, D3 {'on' if with_d3 else 'off'}"
    else:
        detail = (f"contact {contact:.2f} Å with D3 OFF: rerun the "
                  "relaxation with with_d3=true")

    store.record_check("dispersion", passed, detail)
    return f"dispersion: {'PASS' if passed else 'FAIL'} - {detail}"


@tool
def check_dispersion_consistent() -> str:
    """Check endpoints, gas reference and NEB all used the same D3 setting.

    A barrier assembled from a mix of D3-on and D3-off energies is not a
    barrier on any single potential energy surface. This is a separate
    question from check_dispersion_relevance, which asks whether a
    missing dispersion term explains a drifted geometry.
    """
    settings = {}
    for key in ("initial_relaxed", "final_relaxed", "gasref_relaxed", "neb"):
        record = store.get(key)
        if record is None:
            settings[key] = "missing"
        else:
            settings[key] = record.get("with_d3", "not recorded")

    passed = (set(settings.values()) == {True})

    detail = ", ".join(f"{k}={v}" for k, v in settings.items())
    if not passed:
        detail += (" - every stage must use the same setting, and D3 should "
                   "be on for an RPBE-trained model")

    store.record_check("dispersion_consistent", passed, detail)
    return f"dispersion_consistent: {'PASS' if passed else 'FAIL'} - {detail}"


@tool
def check_gas_reference_applied() -> str:
    """Check the scored barrier is referenced to the free molecule.

    SBH10 measures from an isolated gas-phase molecule. run_neb reports
    from the physisorbed state. If compute_gas_referenced_barrier was
    never called, the scored number is the wrong quantity, and it will
    look perfectly reasonable, because it is a real barrier, just
    measured from the wrong zero.
    """
    barrier_gas = store.get("barrier_gas_eV")
    well_depth = store.get("well_depth_eV")

    if barrier_gas is None or well_depth is None:
        passed = False
        detail = ("compute_gas_referenced_barrier was never called, so the "
                  "scored barrier is still measured from the physisorbed "
                  "state")
    elif well_depth < 0:
        passed = False
        detail = (f"well depth {well_depth:.3f} eV is negative, meaning the "
                  "lifted molecule relaxed below the physisorbed state, one "
                  "of the two is not a real minimum")
    else:
        passed = True
        detail = (f"gas-referenced barrier {barrier_gas:.3f} eV, "
                  f"well depth {well_depth:.3f} eV")

    store.record_check("gas_reference", passed, detail)
    return f"gas_reference: {'PASS' if passed else 'FAIL'} - {detail}"


# NEW
@tool
def check_fragments_sensible() -> str:
    """Check the endpoint split the molecule into plausible products.

    Guards the bond-selection logic in build_dissociated_endpoint. The
    classic failure is CH4 splitting into CH2 + H2 instead of CH3 + H,
    because two hydrogens on the same carbon sit further apart than any
    C-H bond.
    """
    record = store.get("final")
    if record is None:
        store.record_check("fragments", False, "no dissociated endpoint built")
        return "fragments: FAIL (no dissociated endpoint built)"

    groups = record.get("fragment_symbols")
    if groups is None:
        store.record_check("fragments", False,
                           "fragment composition was not recorded")
        return "fragments: FAIL (fragment composition was not recorded)"

    left, right = groups
    problems = []

    if not left or not right:
        problems.append("one fragment is empty")

    for group in (left, right):
        if sorted(group) == ["H", "H"]:
            problems.append("one fragment is H2, which means a geminal pair "
                            "was pulled apart rather than a bond broken")

    passed = not problems
    detail = f"broke {record.get('broken_bond', '?')}, giving {left} and {right}"
    if problems:
        detail += " - " + "; ".join(problems)

    store.record_check("fragments", passed, detail)
    return f"fragments: {'PASS' if passed else 'FAIL'} - {detail}"


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


def _longest_bond(atoms) -> float:
    """Longest distance between any two adsorbate atoms."""
    tags = atoms.get_tags()
    ads = [i for i in range(len(atoms)) if tags[i] == 2]
    return max(atoms.get_distance(i, j, mic=True)
               for i in ads for j in ads if i < j)


@tool
def check_reaction_consistency() -> str:
    """Check the barrier is consistent with the reaction energetics.

    An endothermic reaction cannot have a barrier below its reaction
    energy, the peak must sit at least as high as the final state. And
    a barrier above ~2.5 eV on a metal surface means a badly built
    endpoint, not difficult chemistry (based on typical barriers on transition metal surfaces).
    """
    neb = store.get("neb")
    if neb is None:
        store.record_check("reaction_consistency", False, "no NEB run")
        return "reaction_consistency: FAIL - no NEB run"

    ea, dE = neb["barrier_eV"], neb["reaction_energy_eV"]

    if dE > 0 and ea < dE - 0.01:
        passed, detail = False, f"barrier {ea:.3f} eV is below reaction energy {dE:.3f} eV"
    elif ea > 2.5:
        passed, detail = False, f"barrier {ea:.3f} eV is chemically unreasonable"
    else:
        passed, detail = True, f"barrier {ea:.3f} eV, reaction energy {dE:.3f} eV"

    store.record_check("reaction_consistency", passed, detail)
    return f"reaction_consistency: {'PASS' if passed else 'FAIL'} - {detail}"


@tool
def check_endpoints_distinct() -> str:
    """Check the two endpoints are genuinely different states.

    This is what makes a zero barrier interpretable. Zero between two
    distinct minima is a real non-activated reaction. Zero because both
    endpoints relaxed into the same structure is not a result at all.
    """
    try:
        d0 = _longest_bond(read(_path("initial.traj")))
        d1 = _longest_bond(read(_path("final.traj")))
    except Exception as exc:
        store.record_check("endpoints_distinct", False, str(exc))
        return f"endpoints_distinct: FAIL - {exc}"

    passed = (d1 - d0) > 0.8 # Typical H-H bond is 0.74 Å.
    detail = f"adsorbate bond {d0:.2f} -> {d1:.2f} Å"
    if not passed:
        detail += " - endpoints look like the same state"

    store.record_check("endpoints_distinct", passed, detail)
    return f"endpoints_distinct: {'PASS' if passed else 'FAIL'} - {detail}"


@tool
def check_path_resolved() -> str:
    """Check the band actually samples the barrier.

    If one image-to-image step accounts for most of the climb, the
    transition state sits inside that gap and the reported barrier is a
    lower bound on a shape you have not sampled. The fix is more images.
    """
    neb = store.get("neb")
    if neb is None:
        store.record_check("path_resolved", False, "no NEB run")
        return "path_resolved: FAIL - no NEB run"

    profile, ea = neb["profile_eV"], neb["barrier_eV"]

    if ea < 0.1:
        store.record_check("path_resolved", True, "path nearly flat")
        return "path_resolved: PASS - path nearly flat, nothing to resolve"

    jumps = [abs(b - a) for a, b in zip(profile, profile[1:])]
    fraction = max(jumps) / ea
    worst = int(np.argmax(jumps))

    passed = fraction < 0.6
    detail = (f"largest single step is {fraction:.0%} of the barrier, "
              f"between images {worst} and {worst + 1}")
    if not passed:
        detail += " - refine the path around those images"

    store.record_check("path_resolved", passed, detail)
    return f"path_resolved: {'PASS' if passed else 'FAIL'} - {detail}"


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

    # name the checks that were never run, rather than
    # summarising a partial set as though it were complete.
    expected = {"convergence", "noise_floor", "dispersion",
                "dispersion_consistent", "gas_reference", "fragments",
                "geometry", "reaction_consistency", "endpoints_distinct",
                "path_resolved"}
    missing = sorted(expected - set(checks))

    lines.append("")
    if missing:
        lines.append(f"NOT ALL CHECKS WERE RUN - missing: {', '.join(missing)}")
    elif store.all_checks_passed():
        lines.append("ALL PASSED")
    else:
        lines.append("NOT ALL CHECKS PASSED - the run is not finished")
    return "\n".join(lines)



# Helpers

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


STRUCTURE_TOOLS = [build_slab, build_stepped_slab, place_adsorbate, build_dissociated_endpoint]
SIMULATION_TOOLS = [
    relax_structure,
    run_neb,
    build_gas_reference,
    compute_gas_referenced_barrier,
    read_results,
]
VALIDATION_TOOLS = [
    check_convergence,
    check_noise_floor,
    check_dispersion_relevance,
    check_dispersion_consistent,
    check_gas_reference_applied,
    check_fragments_sensible,
    check_geometry,
    check_reaction_consistency,
    check_endpoints_distinct,
    check_path_resolved,
    validation_summary,
]