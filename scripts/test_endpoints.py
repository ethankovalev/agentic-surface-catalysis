"""Ten checks on dissociated endpoint construction, across all reactions.

No GPU, no checkpoint, no API key. Pure geometry.

    python scripts/test_endpoints.py
"""
import sys
from pathlib import Path
import numpy as np
from ase.data import atomic_numbers, covalent_radii
from ase.io import read

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src import store
from src.benchmark import SBH10
from src.tools import (build_dissociated_endpoint, build_slab,
                       build_stepped_slab, place_adsorbate)


def _path(n):
    return str(Path(getattr(config, "WORK_DIR", "work")) / n)


def geometry(rid, spec):
    store.reset(rid)
    stepped = spec.get("site_type") == "step"
    (build_stepped_slab if stepped else build_slab).invoke(
        {"metal": spec["metal"], "facet": spec["facet"]})
    place_adsorbate.invoke(
        {"species": spec["molecule"], "site": "step" if stepped else "ontop"})
    build_dissociated_endpoint.invoke({})
    return read(_path("initial.traj")), read(_path("final.traj"))


def run(rid, spec):
    ini, fin = geometry(rid, spec)
    tags = ini.get_tags()
    ads = [i for i in range(len(ini)) if tags[i] == 2]
    metal = [i for i in range(len(ini)) if tags[i] != 2]
    cell = np.array(ini.cell[:2, :2], float)
    inv = np.linalg.inv(cell)
    top_i = max(ini.positions[m, 2] for m in metal)
    top_f = max(fin.positions[m, 2] for m in metal)
    r_metal = covalent_radii[atomic_numbers[ini[metal[0]].symbol]]
    out = []

    def t(name, ok, detail):
        out.append((name, bool(ok), detail))

    # 1. stored coordinates describe the short path, not one across the cell
    worst = 0.0
    for i in ads:
        d = fin.positions[i][:2] - ini.positions[i][:2]
        f = d @ inv
        mic = (f - np.round(f)) @ cell
        worst = max(worst, abs(np.linalg.norm(d) - np.linalg.norm(mic)))
    t("1 no periodic wrap", worst < 0.01, f"raw vs mic differ by {worst:.3f} A")

      f"fractional range {fr.min():.2f} to {fr.max():.2f}")

    # 3. slabs aligned between the two endpoints
    shift = np.abs(fin.positions[metal] - ini.positions[metal]).max()
    t("3 slabs aligned", shift < 0.01, f"max metal shift {shift:.4f} A")

    # 4. no adsorbate inside the slab
    buried = [(fin[i].symbol, fin.positions[i, 2] - top_f)
              for i in ads if fin.positions[i, 2] - top_f < 0.3]
    t("4 nothing buried", not buried, f"{buried}" if buried else "all above surface")

    # 5. the molecule really came apart
    bond_i = ini.get_distance(ads[0], ads[1], mic=True)
    bond_f = fin.get_distance(ads[0], ads[1], mic=True)
    t("5 endpoints distinct", bond_f - bond_i > 0.8,
      f"pair {bond_i:.2f} -> {bond_f:.2f} A")

    # 6. fragments bonded at a sensible distance, not floating or embedded
    bad = []
    for i in ads:
        ideal = r_metal + covalent_radii[fin[i].number]
        near = min(fin.get_distance(i, m, mic=True) for m in metal)
        if abs(near - ideal) > 0.6:
            bad.append(f"{fin[i].symbol} at {near:.2f} vs {ideal:.2f}")
    t("6 fragments bonded", not bad, "; ".join(bad) if bad else "all within 0.6 A")

    # 7. fragments far enough apart not to recombine on relaxation
    t("7 not recombining", bond_f > 1.4 * bond_i,
      f"{bond_f:.2f} A against a {bond_i:.2f} A bond")

    # 8. the lateral journey is short enough for interpolation to be sane
    far = []
    for i in ads:
        d = fin.positions[i][:2] - ini.positions[i][:2]
        f = d @ inv
        mic = np.linalg.norm((f - np.round(f)) @ cell)
        if mic > 5.0:
            far.append(f"{fin[i].symbol} moves {mic:.2f} A")
    t("8 short lateral move", not far, "; ".join(far) if far else "all under 5 A")

    # 9. the intact molecule starts clear of the surface
    clear = min(ini.get_distance(a, m, mic=True) for a in ads for m in metal)
    t("9 initial clear", clear >= 1.8, f"closest contact {clear:.2f} A")

    # 10. same atom count and composition at both ends
    same = (len(ini) == len(fin)
            and sorted(ini.get_chemical_symbols()) ==
                sorted(fin.get_chemical_symbols()))
    t("10 composition preserved", same,
      f"{len(ini)} vs {len(fin)} atoms")

    return out


def main():
    print(f"Ten endpoint checks across {len(SBH10)} reactions. No GPU.\n")
    failures = 0
    for rid, spec in SBH10.items():
        print(f"=== {rid}")
        try:
            results = run(rid, spec)
        except Exception as exc:
            print(f"    CRASHED: {type(exc).__name__}: {exc}\n")
            failures += 1
            continue
        for name, ok, detail in results:
            print(f"  [{'ok  ' if ok else 'FAIL'}] {name:24s} {detail}")
            failures += 0 if ok else 1
        print()
    print("=" * 60)
    print("all clear" if not failures else f"{failures} check(s) failing")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
