from pathlib import Path
import sys

p = Path("src/tools.py")
text = p.read_text()

if "_z_layers" in text:
    sys.exit("ABORT: already patched.")

NEW = '''def _z_layers(atoms, metal_indices, tol=0.6):
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


def _hollows_one_layer(atoms, indices, n_grid=64):
    """Hollow sites on one flat terrace, as (xy, z, clearance)."""
    cell = np.array(atoms.cell[:2, :2], dtype=float)
    inv = np.linalg.inv(cell)
    xy = atoms.positions[indices][:, :2]
    z = float(np.mean(atoms.positions[indices][:, 2]))

    us = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    grid = np.array([[u, v] for u in us for v in us]) @ cell
    shifts = np.array([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1)],
                      dtype=float) @ cell
    images = (xy[:, None, :] + shifts[None, :, :]).reshape(-1, 2)
    d = np.linalg.norm(grid[:, None, :] - images[None, :, :], axis=2)
    clearance = d.min(axis=1).reshape(n_grid, n_grid)

    dm = np.linalg.norm(xy[:, None, :] - images[None, :, :], axis=2)
    dm[dm < 1e-6] = np.inf
    nn = float(dm.min())

    cmax = clearance.max()
    peaks = []
    for i in range(n_grid):
        for j in range(n_grid):
            c = clearance[i, j]
            if c < 0.9 * cmax:
                continue
            neigh = [clearance[(i + a) % n_grid, (j + b) % n_grid]
                     for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
            if c >= max(neigh) - 1e-9:
                peaks.append((grid[i * n_grid + j], c))

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

    # A genuine hollow sits 0.58*nn (three-fold) to 0.71*nn (four-fold) from
    # its neighbours. Anything much further is a hole where atoms were removed,
    # not a binding site.
    return [(pt, z, c) for pt, c, _ in merged if c <= 0.85 * nn]


def _surface_sites(atoms, metal_indices, max_layers=2):
    """Hollow sites on every exposed terrace, each carrying its own height.

    A stepped slab has two surfaces at different heights, and their in-plane
    projections overlap, so treating the surface as one flat sheet is wrong.
    Doing it that way put the two nitrogen atoms of N2/Ru(0001) 3.26 A from
    the nearest metal, hovering over the hole the carve left behind, because
    the point furthest from the eight remaining upper-terrace atoms is the
    void itself.

    Each terrace is therefore searched separately and every site remembers the
    height of the terrace it belongs to.
    """
    layers = _z_layers(atoms, metal_indices)
    if not layers:
        return []
    sites, top_n = [], len(layers[0])
    for depth, indices in enumerate(layers[:max_layers]):
        # the layer below only counts as surface if the one above is carved
        if depth > 0 and top_n >= 0.8 * len(indices):
            break
        sites.extend(_hollows_one_layer(atoms, indices))
    return sites


def _mic_xy'''

text = text.replace("def _mic_xy", NEW, 1)

OLD_BLOCK = '''    surface_z = max(atoms.positions[i, 2] for i in metal)
    sites = _hollow_sites(atoms, metal)
    used = []
    heights = []'''
NEW_BLOCK = '''    sites = _surface_sites(atoms, metal)
    used = []
    heights = []'''
if text.count(OLD_BLOCK) != 1:
    sys.exit(f"ABORT: found {text.count(OLD_BLOCK)} matches for the sites block.")
text = text.replace(OLD_BLOCK, NEW_BLOCK)

OLD_LOOP_START = '''    top = _top_layer(atoms, metal)
    top_xy = atoms.positions[top][:, :2]
    cell2 = np.array(atoms.cell[:2, :2], dtype=float)
    inv2 = np.linalg.inv(cell2)
'''
NEW_LOOP_START = '''    cell2 = np.array(atoms.cell[:2, :2], dtype=float)
    inv2 = np.linalg.inv(cell2)
'''
if text.count(OLD_LOOP_START) != 1:
    sys.exit("ABORT: loop preamble not found as expected.")
text = text.replace(OLD_LOOP_START, NEW_LOOP_START)

OLD_PLACE = '''        site_xy = atoms.positions[anchor, :2].copy()
        free = [n for n in range(len(sites)) if n not in used]
        if free:
            axy = atoms.positions[anchor, :2].copy()
            if not used:
                # first fragment: the hollow nearest where it already sits
                best_n = min(free, key=lambda n: _mic_xy(sites[n], axy, cell2, inv2))
            else:'''
NEW_PLACE = '''        site_xy = atoms.positions[anchor, :2].copy()
        site_z = max(atoms.positions[i, 2] for i in metal)
        lateral = 0.0
        free = [n for n in range(len(sites)) if n not in used]
        if free:
            axy = atoms.positions[anchor, :2].copy()
            if not used:
                # first fragment: the hollow nearest where it already sits
                best_n = min(free, key=lambda n: _mic_xy(sites[n][0], axy, cell2, inv2))
            else:'''
if text.count(OLD_PLACE) != 1:
    sys.exit("ABORT: placement block not found as expected.")
text = text.replace(OLD_PLACE, NEW_PLACE)

OLD_SECOND = '''                first_xy = sites[used[0]]
                best_n = min(free, key=lambda n: abs(
                    _mic_xy(sites[n], first_xy, cell2, inv2) - separation))
            used.append(best_n)
            site_xy = sites[best_n]
            atoms.positions[group, 0] += site_xy[0] - axy[0]
            atoms.positions[group, 1] += site_xy[1] - axy[1]'''
NEW_SECOND = '''                first_xy = sites[used[0]][0]
                best_n = min(free, key=lambda n: abs(
                    _mic_xy(sites[n][0], first_xy, cell2, inv2) - separation))
            used.append(best_n)
            site_xy, site_z, lateral = sites[best_n]
            atoms.positions[group, 0] += site_xy[0] - axy[0]
            atoms.positions[group, 1] += site_xy[1] - axy[1]'''
if text.count(OLD_SECOND) != 1:
    sys.exit("ABORT: second-fragment block not found as expected.")
text = text.replace(OLD_SECOND, NEW_SECOND)

OLD_HEIGHT = '''        df = (top_xy - site_xy) @ inv2
        df -= np.round(df)
        lateral = float(np.linalg.norm(df @ cell2, axis=1).min())
        h = float(np.sqrt(max(bond ** 2 - lateral ** 2, 0.25)))
        heights.append(h)'''
NEW_HEIGHT = '''        # lateral now comes from the site itself, measured against the
        # terrace that site belongs to rather than the whole slab
        h = float(np.sqrt(max(bond ** 2 - lateral ** 2, 0.25)))
        heights.append(h)'''
if text.count(OLD_HEIGHT) != 1:
    sys.exit("ABORT: height block not found as expected.")
text = text.replace(OLD_HEIGHT, NEW_HEIGHT)

OLD_Z = '''        atoms.positions[group, 2] += (surface_z + h) - atoms.positions[anchor, 2]
        lowest = min(atoms.positions[i, 2] for i in group)
        floor = surface_z + 1.0'''
NEW_Z = '''        atoms.positions[group, 2] += (site_z + h) - atoms.positions[anchor, 2]
        lowest = min(atoms.positions[i, 2] for i in group)
        floor = site_z + 1.0'''
if text.count(OLD_Z) != 1:
    sys.exit("ABORT: z-placement block not found as expected.")
text = text.replace(OLD_Z, NEW_Z)

p.write_text(text)
print("Patched src/tools.py: per-terrace surface sites")
