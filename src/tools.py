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
from ase.vibrations import Vibrations


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


def _z_layers(atoms, metal_indices, tol=0.6):
    """Metal atoms grouped into z-layers, highest first."""
    ordered = sorted((atoms.positions[i, 2], i) for i in metal_indices)
    layers, current, cz = [], [], None
    for z, i in ordered:
        if cz is None or abs(z - cz) <= tol:
            current.append(i)
            cz = z if cz is None else cz
        else:
            layers.append(current)
            current, cz = [i], z
    if current:
        layers.append(current)
    return list(reversed(layers))


def _nn_distance(atoms, indices):
    """Shortest distance between two atoms of one layer, under pbc."""
    return min(atoms.get_distance(i, j, mic=True)
               for i in indices for j in indices if i != j)


def _hollows_one_layer(atoms, indices, n_grid=64):
    """Hollow sites on a single layer of metal atoms.

    Returns (sites, nn): the sites as (xy, z, clearance) triples, and the
    layer's nearest-neighbour distance, which the caller needs to judge
    distances on this layer against.

    A hollow is a local maximum of the in-plane distance to that layer's
    atoms, within 10% of the widest such point. Locating hollows as maxima
    of the distance field works for fcc(111), fcc(100) and hcp(0001)
    without the code knowing which facet it is looking at. Checked against
    3x3 cells: 18 sites on fcc(111) and hcp(0001) (the fcc and hcp
    hollows), 9 on fcc(100) (the four-fold hollows). Note this cannot tell
    an fcc hollow from an hcp one; both are returned and the caller takes
    whichever it wants.

    Clearance is returned with each site because the caller needs it to
    work out how high to sit: at a hollow the neighbours are laterally
    displaced, so the vertical drop is not the bond length.
    """
    cell = np.array(atoms.cell[:2, :2], dtype=float)
    inv = np.linalg.inv(cell)
    xy = atoms.positions[indices][:, :2]
    z = float(np.mean(atoms.positions[indices][:, 2]))
    nn = _nn_distance(atoms, indices)

    us = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    grid = np.array([[u, v] for u in us for v in us]) @ cell
    shifts = np.array([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1)],
                      dtype=float) @ cell
    images = (xy[:, None, :] + shifts[None, :, :]).reshape(-1, 2)
    d = np.linalg.norm(grid[:, None, :] - images[None, :, :], axis=2)
    clearance = d.min(axis=1).reshape(n_grid, n_grid)

    tops, widest = [], 0.0
    for i in range(n_grid):
        for j in range(n_grid):
            c = clearance[i, j]
            if c > 0.85 * nn:
                continue                # the carved void, not a binding site
            neigh = [clearance[(i + a) % n_grid, (j + b) % n_grid]
                     for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
            if c < max(neigh) - 1e-9:
                continue
            tops.append((grid[i * n_grid + j], c))
            # Three or more atoms equidistant is what makes a point a hollow
            # rather than a bridge, and the widest such point sets the scale
            # everything else is judged against. Taking that scale from the
            # distance field instead, as a flat surface can, fails on the
            # upper terrace of a stepped slab: there the widest point of the
            # field is the void the carve left, three times any real hollow,
            # and measuring against it rejects every site on that terrace.
            if c > widest and np.count_nonzero(d[i * n_grid + j] < 1.15 * c) >= 3:
                widest = c

    if widest == 0.0:
        return [], nn          # nothing on this layer looks like a hollow

    peaks = [(pt, c) for pt, c in tops if c >= 0.9 * widest]

    merged = []
    for pt, c in peaks:
        for k, (m, cc, n) in enumerate(merged):
            df = (pt - m) @ inv
            df -= np.round(df)
            if np.linalg.norm(df @ cell) < 0.3 * nn:
                merged[k] = ((m * n + pt) / (n + 1), max(cc, c), n + 1)
                break
        else:
            merged.append((pt, c, 1))

    return [(pt, z, c) for pt, c, _ in merged], nn


def _bridges_one_layer(atoms, indices, nn):
    """Bridge sites: midpoints between nearest-neighbour pairs on one layer.

    Used only when every hollow on a layer is too wide for a fragment's
    bond length - see _surface_sites. Verified on Ni(100): 18 bridge
    sites, clearance 1.245 A, against 9 hollow sites at 1.760 A, for a
    H-Ni bond target of 1.550 A.
    """
    cell2 = np.array(atoms.cell[:2, :2], dtype=float)
    inv2 = np.linalg.inv(cell2)
    pos = atoms.positions[indices]
    z = float(np.mean(pos[:, 2]))

    bridges = []
    for i in range(len(indices)):
        for j in range(len(indices)):
            if i >= j:
                continue
            dist = _mic_xy(pos[i, :2], pos[j, :2], cell2, inv2)
            if 0.8 * nn < dist < 1.2 * nn:
                diff = (pos[j, :2] - pos[i, :2]) @ inv2
                diff -= np.round(diff)
                mid_xy = pos[i, :2] + (diff / 2.0) @ cell2
                bridges.append((mid_xy, z, dist / 2.0))

    merged = []
    for pt, b_z, c in bridges:
        for m_pt, _, _ in merged:
            if _mic_xy(pt, m_pt, cell2, inv2) < 0.2 * nn:
                break
        else:
            merged.append((pt, b_z, c))
    return merged


def _atops_one_layer(atoms, indices):
    """Atop sites: directly over a surface atom, clearance zero.

    The last resort when even bridge sites are too wide. Clearance zero
    always yields a valid height, h = bond, so this can never itself
    produce the invalid-sqrt bug that motivated this fallback chain.
    """
    return [(atoms.positions[i, :2], atoms.positions[i, 2], 0.0) for i in indices]


def _surface_sites(atoms, metal_indices, max_layers=2, target_bond=None):
    """Hollow sites on every exposed terrace, each carrying its own height.

    A stepped slab has two terraces at different heights whose in-plane
    projections overlap, so one flat sheet of surface atoms is the wrong
    model. Searched that way, the point furthest from the eight atoms of
    the upper terrace is the void the carve left behind, and both nitrogen
    atoms of N2/Ru(0001) were placed in it: 3.2 A from the nearest metal
    atom against a 2.17 A covalent bond, and 4.5 A from the step edge the
    reaction is supposed to happen at.

    Each layer is therefore searched on its own and every site remembers
    the height of the layer it belongs to. A hollow in a lower layer counts
    only if it is exposed: nothing in a higher layer within 0.85 nearest
    neighbour distances of it in-plane. Without that test the second layer
    of a stepped slab offers hollows right across the cell, the ones roofed
    over by the upper terrace included, and a nitrogen dropped into one of
    those came out under the surface, 0.56 A from the atom above it.
    """
    layers = _z_layers(atoms, metal_indices)
    if not layers:
        return []

    cell2 = np.array(atoms.cell[:2, :2], dtype=float)
    inv2 = np.linalg.inv(cell2)

    sites, above, previous = [], [], None
    for indices in layers[:max_layers]:
        if len(indices) < 3:
            break              # too few atoms to define a hollow
        # a layer is surface only where the one above it was carved away
        if previous is not None and len(previous) >= 0.8 * len(indices):
            break
        found, nn = _hollows_one_layer(atoms, indices)

        # A hollow's clearance can exceed a fragment's own bond length: on
        # Ni(100) every 4-fold hollow is 1.73-1.76 A from its nearest atom,
        # wider than a H-Ni bond (1.55 A). sqrt(bond^2 - clearance^2) then
        # has no real solution, and the placement code upstream silently
        # floors it to 0.25 rather than raising, producing an actual
        # metal-fragment distance of 2.4-2.5 A against a 1.55 A target. If
        # every hollow on this layer is too wide, fall back to bridge
        # sites, then atop, rather than returning a site with no valid
        # height at all.
        if (target_bond is not None and found
                and all(c >= target_bond - 1e-4 for _, _, c in found)):
            bridges = _bridges_one_layer(atoms, indices, nn)
            if bridges and any(c < target_bond - 1e-4 for _, _, c in bridges):
                found = bridges
            else:
                found = _atops_one_layer(atoms, indices)

        for xy, z, c in found:
            roofed = any(_mic_xy(xy, atoms.positions[i, :2], cell2, inv2)
                         < 0.85 * nn for i in above)
            if not roofed:
                sites.append((xy, z, c))
        above.extend(indices)
        previous = indices
    return sites


def _mic_xy(p, q, cell2, inv2):
    """In-plane distance between two points under periodic boundaries."""
    df = (np.asarray(p, dtype=float) - np.asarray(q, dtype=float)) @ inv2
    df -= np.round(df)
    return float(np.linalg.norm(df @ cell2))


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

# A diatomic dissociating on a surface approaches with its bond roughly
# parallel to the surface, so both atoms can reach binding sites. ASE's g2
# database returns diatomics aligned along z, standing perpendicular.
#
# Left unrotated, the reaction path has to rotate the molecule 90 degrees,
# stretch the bond and move both atoms to their sites, all inside the same
# interpolation. On H2/Cu(111) that produced bands whose peak sat off the
# dissociation path entirely: refinement from those peaks converged to a
# stationary point with one hydrogen 1.69 A inside the slab.
#
# Only diatomics are rotated. Polyatomics like CH4 have no single bond axis
# to align and are left as the database supplies them.


def _orient_for_dissociation(ads):
    """Lay a diatomic's bond parallel to the surface. Others are unchanged."""
    if len(ads) != 2:
        return ads
    axis = ads.positions[1] - ads.positions[0]
    if np.linalg.norm(axis) < 1e-6:
        return ads
    ads.rotate(axis, (1.0, 0.0, 0.0), center="COM")
    return ads


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

    ads = _orient_for_dissociation(ads)

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

def _wrap_to_nearest_image(atoms, group, reference_xy):
    """Move a fragment to its minimum-image position near reference_xy.

    Fragment placement pushes atoms apart and then snaps them to hollow
    sites, and the site chosen can lie outside the cell in stored
    coordinates. That is physically the same position, but the NEB
    interpolates stored coordinates, not minimum images, so an atom
    recorded 11.8 A away is dragged right across the slab and back when
    its true displacement is 1.5 A. The band arcs over the surface, the
    path is not the reaction coordinate, and every saddle refined from it
    belongs to some other process.

    Measured on H2/Cu(111): raw lateral moves of 11.81 and 7.60 A against
    true minimum-image moves of 1.45 and 2.19 A, in a cell 7.66 A wide.
    """
    cell2 = np.array(atoms.cell[:2, :2], dtype=float)
    inv2 = np.linalg.inv(cell2)
    anchor = max(group, key=lambda k: covalent_radii[atoms[k].number])
    delta = atoms.positions[anchor, :2] - np.asarray(reference_xy, dtype=float)
    frac = delta @ inv2
    correction = np.round(frac) @ cell2
    atoms.positions[group, 0] -= correction[0]
    atoms.positions[group, 1] -= correction[1]
    return atoms


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
        # The old default, 2.5x the summed covalent radii, targets a real
        # site but overshoots past the genuine adjacent-site product into a
        # second, further-diffused minimum. On H2/Cu(111) it aimed for
        # 4.08 A and landed the endpoint at 3.89 A, while the actual
        # transition state connects to an adjacent-site product only 2.03 A
        # apart (confirmed by relaxing backward from the real saddle along
        # its imaginary mode). The saddle was correct; the endpoint was
        # aimed at the wrong basin.
        #
        # This targets the nearest real adjacent site instead of an
        # arbitrary continuous distance, with a safety margin against
        # recombination. 1.4x the intact bond let N2/Ru0001 relax back
        # into N2 at 1.9 A in an earlier bug, so the margin here is 2.0x,
        # and if the nearest site is still inside that margin the search
        # moves out to the next real site rather than an interpolated
        # point that corresponds to no actual binding site.
        #
        # This initial value only sets the first lateral push, before the
        # site search below moves each fragment onto a real hollow and
        # overwrites it with the actual site-to-site distance. 2.5x the
        # summed covalent radii is a reasonable rough push - anything in
        # the right neighbourhood works, since the site snap corrects it.
        use_site_search = True
        separation = 2.5 * (r_metal + r_ads)
    else:
        use_site_search = False

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

    # `best` is the intact bond length, only known once the bonded pair has
    # been found, so the recombination-safe margin is computed here rather
    # than up with the separation default.
    min_safe_separation = 2.0 * best

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
    # _surface_sites is called once and shared between both fragments, but
    # each fragment's own bond length is only known per-anchor, inside the
    # loop below. On CH4/Ni(100) the CH3 fragment's anchor is carbon
    # (bond ~2.00 A, satisfied by any hollow) while the lone H's anchor is
    # hydrogen (bond ~1.55 A), and every hollow on Ni(100) has a clearance
    # of ~1.73-1.76 A - wider than the H bond but narrower than the C bond.
    # sqrt(bond^2 - clearance^2) is then a negative number under the root
    # for the H fragment specifically, silently floored to 0.25 by the
    # max() below rather than raising, which produced an actual
    # metal-fragment distance of 2.4-2.5 A against a 1.55 A target.
    #
    # The tighter of the two fragments' bonds is what determines whether a
    # hollow is usable at all, so that is what is passed to the site
    # search, not either fragment's bond alone.
    def _anchor_bond(group):
        anc = max(group, key=lambda k: covalent_radii[atoms[k].number])
        return height if height is not None else r_metal + covalent_radii[atoms[anc].number]

    min_bond = min(_anchor_bond(left), _anchor_bond(right))
    sites = _surface_sites(atoms, metal, target_bond=min_bond)
    used = []
    heights = []

    cell2 = np.array(atoms.cell[:2, :2], dtype=float)
    inv2 = np.linalg.inv(cell2)

    for group in (left, right):
        # the largest atom in the fragment is the one that binds, so it is
        # the anchor and its own radius sets the bond length. Averaging radii
        # across the whole adsorbate put a CH3 carbon at a hydrogen height.
        anchor = max(group, key=lambda k: covalent_radii[atoms[k].number])
        bond = (height if height is not None
                else r_metal + covalent_radii[atoms[anchor].number])

        site_xy = atoms.positions[anchor, :2].copy()
        site_z = max(atoms.positions[i, 2] for i in metal)
        lateral = 0.0
        free = [n for n in range(len(sites)) if n not in used]
        if free:
            axy = atoms.positions[anchor, :2].copy()
            if not used:
                # first fragment: the hollow nearest where it already sits
                best_n = min(free, key=lambda n: _mic_xy(sites[n][0], axy, cell2, inv2))
            else:
                first_xy = sites[used[0]][0]
                if use_site_search:
                    # Nearest real site to the first that clears the
                    # recombination-safe margin. A continuous target
                    # distance corresponds to no actual binding site and
                    # either overshoots into a further basin (the
                    # H2/Cu(111) bug this replaces) or, matched too
                    # eagerly, collapses two fragments onto neighbouring
                    # hollows - on Ru(0001) two N atoms separated to
                    # 3.97 A ended up 1.91 A apart after snapping and
                    # recombined into N2 during relaxation.
                    by_distance = sorted(
                        free, key=lambda n: _mic_xy(sites[n][0], first_xy, cell2, inv2))
                    best_n = next(
                        (n for n in by_distance if _mic_xy(
                            sites[n][0], first_xy, cell2, inv2) >= min_safe_separation),
                        by_distance[-1])
                else:
                    # An explicit separation was requested: honour it, same
                    # as before.
                    best_n = min(free, key=lambda n: abs(
                        _mic_xy(sites[n][0], first_xy, cell2, inv2) - separation))
            used.append(best_n)
            site_xy, site_z, lateral = sites[best_n]
            atoms.positions[group, 0] += site_xy[0] - axy[0]
            atoms.positions[group, 1] += site_xy[1] - axy[1]

        # Vertical offset such that the ACTUAL distance to the nearest surface
        # atom is the covalent bond length. At a hollow the neighbours are
        # laterally displaced, so setting the vertical drop equal to the bond
        # length leaves the fragment sqrt(bond^2 + lateral^2) away: CH3 came
        # out 3.13 A from Ni(100) instead of about 2.0.
        # lateral now comes from the site itself, measured against the
        # terrace that site belongs to rather than the whole slab
        h = float(np.sqrt(max(bond ** 2 - lateral ** 2, 0.25)))
        heights.append(h)

        # Anchor on the binding atom, not the lowest atom in the fragment. For
        # CH3 a hydrogen often sits below the carbon, so anchoring on the
        # lowest atom gives the hydrogen the carbon's intended height and
        # floats the carbon above the surface.
        atoms.positions[group, 2] += (site_z + h) - atoms.positions[anchor, 2]
        lowest = min(atoms.positions[i, 2] for i in group)
        floor = site_z + 1.0
        if lowest < floor:
            atoms.positions[group, 2] += floor - lowest

    height = heights

    # Bring each fragment back to its minimum image near where the molecule
    # started, so the stored coordinates describe the short path rather than
    # one that crosses the cell.
    start = read(str(init_file))
    start_tags = start.get_tags()
    start_ads = [i for i in range(len(start)) if start_tags[i] == 2]
    reference_xy = start.positions[start_ads][:, :2].mean(axis=0)
    for group in (left, right):
        atoms = _wrap_to_nearest_image(atoms, group, reference_xy)

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



# Saddle refinement

SADDLE_FMAX = 0.05          # matched to NEB_FMAX so barriers stay comparable
SADDLE_IMAG_MIN_meV = 5.0   # below this a mode is numerical noise, not motion


def _mobile_indices(atoms):
    """Atoms not held fixed by a constraint."""
    fixed = set()
    for c in atoms.constraints:
        if hasattr(c, "get_indices"):
            fixed.update(int(i) for i in c.get_indices())
    return [i for i in range(len(atoms)) if i not in fixed]


def _count_imaginary(atoms, indices, name):
    """Imaginary vibrational modes, in meV, for the given atom indices.

    A first-order saddle has exactly one. Zero means the optimiser landed in a
    minimum; more than one means it is a higher-order stationary point and the
    barrier taken from it is not a reaction barrier.
    """
    import shutil
    shutil.rmtree(name, ignore_errors=True)
    try:
        vib = Vibrations(atoms, indices=indices, name=name)
        vib.run()
        energies = vib.get_energies()
    finally:
        shutil.rmtree(name, ignore_errors=True)
    imag = [abs(e.imag) * 1000.0 for e in energies
            if np.iscomplex(e) and abs(e.imag) * 1000.0 > SADDLE_IMAG_MIN_meV]
    return sorted(imag, reverse=True), len(energies)


@tool
def refine_saddle(model_key: str = None, with_d3: bool = True,
                  max_steps: int = 200, scope: str = "adsorbate") -> str:
    """Refine the NEB's highest image onto the true transition state.

    A band gives you the highest image it happened to sample, which is only the
    saddle if an image landed on it. When path_resolved fails - one
    image-to-image step carrying most of the climb - the real peak sits inside
    that gap and was never computed, so the reported barrier is a lower bound.

    Adding images is the obvious response and an expensive one: every rerun is
    a whole new band from scratch. This converges *to* the saddle instead,
    using partitioned rational function optimisation with iterative Hessian
    diagonalisation (Sella), starting from the highest image. On a test case
    where a 3-image band failed path_resolved at 52 percent, refinement from
    that band reproduced the 11-image answer to within 0.3 meV, in one step.

    It then confirms the result really is a first-order saddle by counting
    imaginary vibrational modes. Exactly one is required. Zero means the
    optimiser fell into a minimum; two or more means a higher-order stationary
    point, which is not a transition state and whose energy is not a barrier.

    scope: which atoms enter the vibrational check. "adsorbate" (default) uses
           the adsorbate atoms only, which is the usual convention for surface
           transition states and costs 6N force evaluations. "mobile" includes
           every unconstrained atom - stricter, and several times slower.

    Run after run_neb. Reads work/peak.traj, writes work/saddle.traj.
    """
    peak_file = Path(_path("peak.traj"))
    if not peak_file.exists():
        return ("FAILED: no peak.traj. Run run_neb first - refinement starts "
                "from the highest image of a converged band, not from scratch.")

    neb = store.get("neb")
    if not neb:
        return "FAILED: no NEB record in the store. Run run_neb first."

    if scope not in ("adsorbate", "mobile"):
        return (f"FAILED: unknown scope '{scope}'. Use 'adsorbate' or 'mobile'. "
                "An unrecognised value must not silently fall back to a default.")

    model_key = model_key or config.DEFAULT_MODEL

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        from sella import Sella
    except ImportError:
        return ("FAILED: sella is not installed. pip install sella. It is in "
                "requirements.txt; this environment predates that entry.")

    atoms = read(str(peak_file))
    atoms.calc = new_calculator(model_key, with_d3=with_d3)

    initial = store.get("initial_relaxed") or {}
    e_initial = initial.get("energy_eV")
    if e_initial is None:
        return ("FAILED: no relaxed initial state. The refined barrier is "
                "measured from it, so relax the initial endpoint first.")

    e_before = float(atoms.get_potential_energy())

    dyn = Sella(atoms, order=1, internal=False,
                trajectory=_path("saddle.traj"), logfile="-")
    converged = bool(dyn.run(fmax=SADDLE_FMAX, steps=max_steps))
    n_steps = int(dyn.get_number_of_steps())
    e_after = float(atoms.get_potential_energy())

    tags = atoms.get_tags()
    ads = [i for i in range(len(atoms)) if tags[i] == 2]
    indices = ads if scope == "adsorbate" else _mobile_indices(atoms)
    if not indices:
        return ("FAILED: no atoms to analyse - the adsorbate is untagged or "
                "every atom is constrained.")

    try:
        imag, n_modes = _count_imaginary(atoms, indices, _path("vib_saddle"))
    except Exception as exc:
        imag, n_modes = None, 0
        vib_note = f" Mode analysis failed ({type(exc).__name__}: {exc})."
    else:
        vib_note = ""

    barrier = e_after - e_initial
    shift = e_after - e_before
    first_order = (imag is not None and len(imag) == 1)

    record = {
        "barrier_eV": barrier,
        "energy_eV": e_after,
        "converged": converged,
        "n_steps": n_steps,
        "shift_from_neb_peak_eV": shift,
        "with_d3": bool(with_d3),
        "fmax_target": SADDLE_FMAX,
        "scope": scope,
        "n_modes": n_modes,
        "imaginary_modes_meV": imag,
        "first_order_saddle": first_order,
        "neb_barrier_eV": neb.get("barrier_eV"),
    }
    store.put("saddle", record)
    store.put("model_key", model_key)

    if not converged:
        return (f"SADDLE REFINEMENT DID NOT CONVERGE after {n_steps} steps at "
                f"fmax {SADDLE_FMAX} eV/A. Energy moved {shift:+.3f} eV from the "
                f"NEB peak. Do not use this barrier. Raise max_steps, or the "
                f"starting image may be too far from any saddle." + vib_note)

    if imag is None:
        return (f"Refined in {n_steps} steps to a barrier of {barrier:.3f} eV "
                f"({shift:+.3f} eV from the NEB peak), but the vibrational "
                f"check did not run, so this is NOT confirmed as a transition "
                f"state." + vib_note)

    if len(imag) == 0:
        return (f"NOT A SADDLE: refinement converged but found zero imaginary "
                f"modes out of {n_modes}, so this is a minimum, not a "
                f"transition state. {barrier:.3f} eV is not a barrier.")

    if len(imag) > 1:
        modes = ", ".join(f"{m:.0f}" for m in imag)
        return (f"NOT A FIRST-ORDER SADDLE: {len(imag)} imaginary modes "
                f"({modes} meV) out of {n_modes}. A transition state has "
                f"exactly one. {barrier:.3f} eV is not a reaction barrier.")

    return (f"Refined to a first-order saddle in {n_steps} steps. Barrier "
            f"{barrier:.3f} eV from the relaxed initial state, {shift:+.3f} eV "
            f"from the NEB peak of {neb.get('barrier_eV'):.3f} eV. One "
            f"imaginary mode at {imag[0]:.0f} meV out of {n_modes}, as a "
            f"transition state requires. Saved to saddle.traj.")

@tool
def check_endpoints_are_minima(model_key: str = None, with_d3: bool = True,
                               scope: str = "adsorbate") -> str:
    """Verify the relaxed endpoints are genuine minima before running a band.

    An optimiser stops when forces fall below tolerance. That can happen at a
    minimum, but also on a shoulder or at a saddle. A band built between
    endpoints that are not minima has no well defined reaction coordinate,
    and the profile comes out incoherent.

    On N2/Ru(0001) at a step, the band swung more than 2 eV between adjacent
    images and two saddle refinements seeded from it converged to different
    stationary points 1.4 eV apart. Neither was the transition state for the
    reaction. That is the failure this check exists to catch, before the band
    is ever run rather than after.

    A minimum has zero imaginary vibrational modes. One or more means the
    structure sits on a slope, and the endpoint needs rebuilding or
    relaxing from a different starting geometry.

    Run after relaxing both endpoints and before run_neb. Costs 6N force
    evaluations per endpoint.
    """
    results, failed = [], False
    for name in ("initial", "final"):
        f = Path(_path(f"{name}.traj"))
        if not f.exists():
            return f"FAILED: no {name}.traj. Relax both endpoints first."

        atoms = read(str(f))
        atoms.calc = new_calculator(model_key or config.DEFAULT_MODEL,
                                    with_d3=with_d3)
        tags = atoms.get_tags()
        ads = [i for i in range(len(atoms)) if tags[i] == 2]
        indices = ads if scope == "adsorbate" else _mobile_indices(atoms)
        if not indices:
            return f"FAILED: no atoms to analyse in {name}.traj."

        try:
            imag, n_modes = _count_imaginary(
                atoms, indices, _path(f"vib_{name}"))
        except Exception as exc:
            return (f"FAILED: mode analysis on {name} did not run "
                    f"({type(exc).__name__}: {exc}).")

        results.append((name, imag, n_modes))
        if imag:
            failed = True

    store.put("endpoint_modes", {
        name: {"imaginary_meV": imag, "n_modes": n, "is_minimum": not imag}
        for name, imag, n in results
    })

    if not failed:
        detail = ", ".join(f"{name} 0 of {n}" for name, _, n in results)
        return (f"Both endpoints are genuine minima, zero imaginary modes "
                f"({detail}). Safe to run the band.")

    lines = []
    for name, imag, n in results:
        if imag:
            modes = ", ".join(f"{m:.0f}" for m in imag)
            lines.append(f"{name} has {len(imag)} imaginary mode(s) at "
                         f"{modes} meV out of {n}")
        else:
            lines.append(f"{name} is a minimum")
    return ("ENDPOINT IS NOT A MINIMUM: " + "; ".join(lines) + ". A band "
            "between these will not trace a reaction coordinate, and any "
            "saddle refined from it may belong to a different process. "
            "Rebuild the endpoint at a different site, or relax it again "
            "from a perturbed geometry, before running run_neb.")



def _imaginary_mode_vector(atoms, indices, name):
    """The single imaginary mode's frequency in meV and its eigenvector.

    Returns (count, None) when there is not exactly one imaginary mode, so
    the caller can report what it found rather than following a mode that
    is not there.
    """
    import shutil
    shutil.rmtree(name, ignore_errors=True)
    try:
        vib = Vibrations(atoms, indices=indices, name=name)
        vib.run()
        energies = vib.get_energies()
        imag = [i for i, e in enumerate(energies)
                if np.iscomplex(e) and abs(e.imag) * 1000.0 > SADDLE_IMAG_MIN_meV]
        if len(imag) != 1:
            return len(imag), None
        return abs(energies[imag[0]].imag) * 1000.0, vib.get_mode(imag[0])
    finally:
        shutil.rmtree(name, ignore_errors=True)


@tool
def check_saddle_connects(model_key: str = None, with_d3: bool = True,
                          displacement: float = 0.35) -> str:
    """Check the refined saddle belongs to THIS reaction.

    One imaginary mode proves a structure is a first-order saddle. It does
    not prove it is the saddle for the reaction being computed. On
    N2/Ru(0001) at a step, two refine_saddle calls seeded from two different
    bands each converged cleanly, each reported exactly one imaginary mode,
    and their gas-referenced barriers differed by 1.4 eV. Both were genuine
    saddles. At most one of them was the transition state for dissociation.

    This displaces the saddle along its imaginary mode in both directions,
    relaxes each, and checks one side falls to the initial state and the
    other to the final state. Falling to the same state twice, or to
    neither, means the saddle sits on a different process and its energy is
    not the barrier for this reaction.

    Costs two short relaxations. Run after refine_saddle.
    """
    for name in ("saddle", "initial", "final"):
        if not Path(_path(f"{name}.traj")).exists():
            return f"FAILED: no {name}.traj. Run refine_saddle first."

    saddle = read(_path("saddle.traj"))
    initial = read(_path("initial.traj"))
    final = read(_path("final.traj"))

    tags = saddle.get_tags()
    ads = [i for i in range(len(saddle)) if tags[i] == 2]
    if len(ads) < 2:
        return "FAILED: fewer than two adsorbate atoms; nothing to compare."

    model_key = model_key or config.DEFAULT_MODEL

    def pair(atoms):
        return float(atoms.get_distance(ads[0], ads[1], mic=True))

    d_initial, d_final = pair(initial), pair(final)
    if abs(d_final - d_initial) < 1.0:
        return (f"FAILED: the endpoints differ by only "
                f"{abs(d_final - d_initial):.2f} A in adsorbate separation, "
                f"too little to tell which one a relaxation fell to. Fix "
                f"endpoints_distinct first.")

    saddle.calc = new_calculator(model_key, with_d3=with_d3)
    try:
        imag, mode = _imaginary_mode_vector(saddle, ads, _path("vib_connect"))
    except Exception as exc:
        return f"FAILED: mode analysis did not run ({type(exc).__name__}: {exc})."
    if mode is None:
        return (f"FAILED: expected exactly one imaginary mode to follow, "
                f"found {imag}. This is not a first-order saddle.")

    landed = {}
    for direction, label in ((1.0, "forward"), (-1.0, "backward")):
        moved = saddle.copy()
        moved.positions += direction * displacement * mode
        moved.calc = new_calculator(model_key, with_d3=with_d3)
        moved.set_constraint(saddle.constraints)
        FIRE(moved, logfile="-").run(fmax=0.05, steps=200)
        d = pair(moved)
        # Classify by which side of the midpoint between the two endpoint
        # separations the relaxation landed, not by nearest endpoint.
        #
        # Dissociation on a surface is often two steps: the molecule splits
        # into adjacent sites, then the fragments diffuse apart. On
        # H2/Cu(111) the IRC from the true dissociation saddle relaxes to a
        # pair separation of 2.03 A, two chemisorbed H atoms 1.62 A from
        # their nearest Cu. That is genuinely dissociated, 2.7 times the
        # 0.74 A bond, but it is not the 3.89 A final state, which lies
        # 0.21 eV lower and is reached by a later diffusion step.
        #
        # Nearest-endpoint matching called that "initial" and rejected a
        # correct saddle. What matters is which basin it fell into.
        midpoint = 0.5 * (d_initial + d_final)
        landed[label] = (d, "initial" if d < midpoint else "final")

    fwd_d, fwd_where = landed["forward"]
    bwd_d, bwd_where = landed["backward"]

    store.put("saddle_connectivity", {
        "connects": fwd_where != bwd_where,
        "forward_pair_A": fwd_d, "forward_lands_on": fwd_where,
        "backward_pair_A": bwd_d, "backward_lands_on": bwd_where,
        "initial_pair_A": d_initial, "final_pair_A": d_final,
        "imaginary_mode_meV": imag,
    })

    if fwd_where == bwd_where:
        return (f"SADDLE DOES NOT CONNECT THE ENDPOINTS: following the "
                f"imaginary mode both ways falls to the {fwd_where} state "
                f"(adsorbate pair {fwd_d:.2f} and {bwd_d:.2f} A, against "
                f"{d_initial:.2f} initial and {d_final:.2f} final). This "
                f"saddle belongs to some other process. Its energy is not "
                f"the barrier for this reaction.")

    return (f"Saddle connects the endpoints. Following the imaginary mode "
            f"({imag:.0f} meV) forward falls to the {fwd_where} state "
            f"(pair {fwd_d:.2f} A) and backward to the {bwd_where} state "
            f"(pair {bwd_d:.2f} A), against {d_initial:.2f} A initial and "
            f"{d_final:.2f} A final. The barrier belongs to this reaction.")



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



MAX_NEB_ATTEMPTS = 2


def _neb_attempt_guard(attempts_so_far, peak_exists):
    """Refuse a further band once the cap is reached.

    Returns a refusal string, or None to allow the call.

    The simulation agent decides to rerun run_neb the instant a band fails,
    within the same turn, without consulting the validation agent. Routing
    guidance placed in validation_agent_prompt is therefore never read at
    the moment the decision is made. This guard sits in the tool, where the
    agent cannot route around it.

    Two is the cap because a third band has never resolved a peak the first
    two could not: on N2/Ru(0001) successive bands returned 0.715 eV and
    9.715 eV, both unconverged, while refine_saddle starts from the highest
    image already computed.
    """
    if attempts_so_far < MAX_NEB_ATTEMPTS:
        return None
    if peak_exists:
        return (
            f"FAILED: run_neb has already run {attempts_so_far} times for this "
            f"reaction, which is the cap. A third band does not resolve a peak "
            f"the first two could not. Call refine_saddle: it starts from the "
            f"highest image already computed, in work/peak.traj, and converges "
            f"onto the saddle directly."
        )
    return (
        f"FAILED: run_neb has already run {attempts_so_far} times for this "
        f"reaction, which is the cap, and no peak image was saved. Report the "
        f"reaction as unresolved rather than running another band."
    )


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

    # Write the highest image on its own. refine_saddle starts from this, and
    # reading it back out of neb.traj is fragile: the optimiser appends the
    # whole band on every step, so the file holds n_images x n_steps frames.
    write(_path("peak.traj"), images[peak])

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

    # Prefer a confirmed saddle over the NEB's highest image. The band only
    # samples the path; the refined structure sits on the transition state and
    # has been shown to have exactly one imaginary mode.
    saddle = store.get("saddle")
    source = "NEB peak"
    barrier_ads = neb["barrier_eV"]
    if saddle and saddle.get("converged") and saddle.get("first_order_saddle"):
        barrier_ads = saddle["barrier_eV"]
        source = "refined saddle"

    well_depth = gasref["energy_eV"] - initial["energy_eV"]
    # Subtract, not add. well_depth > 0 means the physisorbed minimum sits
    # BELOW the free molecule, so a trajectory starting from gas is already
    # part-way up the hill: the gas-referenced barrier must be SMALLER than
    # the barrier measured from the physisorbed state, never larger.
    barrier_gas = barrier_ads - well_depth

    store.put("well_depth_eV", float(well_depth))
    store.put("barrier_gas_eV", float(barrier_gas))
    store.put("barrier_eV", float(barrier_gas))   # this is what gets scored

    store.put("barrier_source", source)

    return (f"Physisorption well depth {well_depth:.3f} eV. "
            f"Barrier from physisorbed state {barrier_ads:.3f} eV ({source}), "
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
    # A confirmed, connected first-order saddle supersedes the band's own
    # convergence, and only the band's. The barrier is then measured from
    # the refined saddle, not from the NEB peak, so whether the band itself
    # reached its force tolerance no longer decides whether the number is
    # usable. This is the same reasoning as check_path_resolved.
    #
    # It never excuses an unconverged endpoint or gas reference: the
    # barrier is measured FROM those energies, so a bad one is fatal
    # regardless of how good the saddle is. Verified against ten scenarios
    # including unconverged endpoints, a disconnected saddle, a
    # non-first-order saddle, and an unconverged saddle - none of which
    # may excuse anything.
    saddle = store.get("saddle")
    connectivity = store.get("saddle_connectivity")
    saddle_resolved = bool(
        saddle and saddle.get("converged") and saddle.get("first_order_saddle")
        and connectivity and connectivity.get("connects"))

    failures = []
    for key in ("initial_relaxed", "final_relaxed", "gasref_relaxed", "neb"):
        record = store.get(key)
        if record is None:
            failures.append(f"{key} was never run")
        elif not record.get("converged"):
            if key == "neb" and saddle_resolved:
                continue
            failures.append(f"{key} did not converge")

    passed = not failures
    if passed and saddle_resolved:
        neb = store.get("neb")
        if neb is not None and not neb.get("converged"):
            detail = ("all converged except the band itself, which was "
                      "superseded by refine_saddle: a confirmed, connected "
                      "first-order saddle")
        else:
            detail = "all converged"
    else:
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


# Minimum height of an adsorbate atom above the top metal layer, in Angstrom.
# Anything below this has penetrated the slab rather than bonded to it.
MIN_ADSORBATE_HEIGHT = 0.3


def _subsurface_adsorbates(atoms):
    """Adsorbate atoms sitting at or below the top metal layer.

    A refined saddle on H2/Cu(111) came back with one hydrogen at -1.69 A,
    embedded in the slab. It was a genuine first-order saddle with exactly
    one imaginary mode, but for subsurface hydrogen penetration rather than
    dissociation, so refine_saddle and the mode count both accepted it.

    closest_contact cannot catch this: the atom was 1.68 A from its nearest
    metal neighbour, an ordinary bond length whether the atom is above the
    surface or inside it. Height is the discriminator, not distance.

    Returns a list of (index, symbol, height) for offending atoms.
    """
    tags = atoms.get_tags()
    ads = [i for i in range(len(atoms)) if tags[i] == 2]
    metal = [i for i in range(len(atoms)) if tags[i] != 2]
    if not ads or not metal:
        return []
    top_z = max(atoms.positions[m, 2] for m in metal)
    return [(i, atoms[i].symbol, float(atoms.positions[i, 2] - top_z))
            for i in ads
            if atoms.positions[i, 2] - top_z < MIN_ADSORBATE_HEIGHT]


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

    # Adsorbate atoms must sit above the surface in every structure that
    # exists. A saddle with a hydrogen inside the slab is a stationary point
    # for subsurface penetration, not for the reaction being computed.
    for name in ("initial", "final", "saddle"):
        f = Path(_path(f"{name}.traj"))
        if not f.exists():
            continue
        try:
            buried = _subsurface_adsorbates(read(str(f)))
        except Exception:
            continue
        for _, symbol, height in buried:
            problems.append(
                f"{name}: {symbol} sits {height:+.2f} A relative to the top "
                f"metal layer, so it is inside the slab rather than bonded "
                f"to the surface")

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

    # A connected, first-order saddle has already resolved the peak this
    # check exists to find. On H2/Cu(111) the NEB peak was 97 percent of
    # the barrier in one image-to-image gap - genuinely under-resolved -
    # but refine_saddle converged onto the true saddle anyway (its energy
    # matched the NEB peak to 6e-9 eV) and check_saddle_connects confirmed
    # it bridges the real reactant and product. Re-litigating the raw
    # band's resolution after that is asking a question refinement has
    # already answered.
    #
    # All three conditions - converged, first-order, connects - are
    # required together. Verified against nine scenarios including a
    # disconnected saddle, an unconverged one, and a non-first-order one:
    # none of those may pass on saddle presence alone.
    saddle = store.get("saddle")
    connectivity = store.get("saddle_connectivity")
    if (saddle and saddle.get("converged") and saddle.get("first_order_saddle")
            and connectivity and connectivity.get("connects")):
        store.record_check(
            "path_resolved", True,
            "NEB peak under-resolved, but refine_saddle converged onto a "
            "confirmed, connected saddle - the peak has been resolved by "
            "direct optimisation rather than by adding images")
        return ("path_resolved: PASS - resolved via refine_saddle, "
                "confirmed by check_saddle_connects")

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
    refine_saddle,
    check_endpoints_are_minima,
    check_saddle_connects,
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