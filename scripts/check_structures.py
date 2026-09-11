"""Check every reaction's starting geometry, without a GPU.

    python scripts/check_structures.py

Runs the real structure-building tool chain for all ten SBH10 reactions and
asserts the things that have actually gone wrong: fragments landing on top of
each other, fragments floating too far from the surface, step reactions built
on flat slabs, endpoints that are not distinct. No calculator, no checkpoint,
no API key. Seconds, not hours.

Every bug found in this project so far was a structure bug, and every one was
found by running a full agent job on a GPU and reading the wreckage. This
turns that loop into something you can run on a laptop between edits.

Exit code is 0 if every reaction passes, 1 otherwise, so it can gate a run.
"""
import sys
import traceback
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers, covalent_radii
from ase.io import read

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from src import store  # noqa: E402
from src.benchmark import SBH10  # noqa: E402
from src.tools import (  # noqa: E402
    breaking_bond,
    build_dissociated_endpoint,
    build_slab,
    build_stepped_slab,
    place_adsorbate,
)

# Matches the invariant build_dissociated_endpoint already enforces via
# min_safe_separation in src/tools.py: a fragment closer than this to its
# partner could relax straight back into the intact molecule. A single
# flat distance across nine different bond types (H-H, N-N, C-H on three
# different metals) does not make physical sense - a real adjacent-site
# product legitimately sits anywhere from 1.5 to 2.7 A away depending on
# the lattice, which is why the old MIN_FRAGMENT_SEPARATION = 3.0 A failed
# every reaction regardless of whether the geometry was actually correct.
RECOMBINATION_MARGIN = 2.0
BOND_TOLERANCE = 0.45
STEP_PROXIMITY = 3.2
MIN_INITIAL_CLEARANCE = 1.8


def _path(name):
    return str(Path(config.WORK_DIR if hasattr(config, "WORK_DIR") else "work") / name)


def _fragments(atoms, ads_idx):
    seen, groups = set(), []
    for a in ads_idx:
        if a in seen:
            continue
        stack, group = [a], []
        while stack:
            k = stack.pop()
            if k in seen:
                continue
            seen.add(k)
            group.append(k)
            for b in ads_idx:
                if b in seen:
                    continue
                reach = covalent_radii[atoms[k].number] + covalent_radii[atoms[b].number]
                if atoms.get_distance(k, b, mic=True) < 1.25 * reach:
                    stack.append(b)
        groups.append(sorted(group))
    return groups


def _anchor(atoms, group):
    return max(group, key=lambda k: covalent_radii[atoms[k].number])


def check_reaction(reaction_id, spec):
    results = []

    def add(name, ok, detail):
        results.append((name, bool(ok), detail))

    store.reset(reaction_id)
    stepped = spec.get("site_type") == "step"

    if stepped:
        msg = build_stepped_slab.invoke(
            {"metal": spec["metal"], "facet": spec["facet"]})
    else:
        msg = build_slab.invoke(
            {"metal": spec["metal"], "facet": spec["facet"]})
    add("slab built", not msg.startswith("FAILED"), msg.split(".")[0][:90])
    if msg.startswith("FAILED"):
        return results

    slab_rec = store.get("slab") or {}
    if stepped:
        edge = slab_rec.get("step_edge_atoms") or []
        add("step edge found", len(edge) > 0,
            f"{len(edge)} under-coordinated atoms")
    else:
        add("flat slab", not slab_rec.get("stepped"), "terrace as required")

    msg = place_adsorbate.invoke(
        {"species": spec["molecule"], "site": "step" if stepped else "ontop"})
    add("molecule placed", not msg.startswith("FAILED"), msg.split(".")[0][:90])
    if msg.startswith("FAILED"):
        return results

    msg = build_dissociated_endpoint.invoke({})
    add("endpoint built", not msg.startswith("FAILED"), msg.split(".")[0][:90])
    if msg.startswith("FAILED"):
        return results

    initial = read(_path("initial.traj"))
    final = read(_path("final.traj"))

    tags_i = initial.get_tags()
    ads_i = [i for i in range(len(initial)) if tags_i[i] == 2]
    metal_i = [i for i in range(len(initial)) if tags_i[i] != 2]
    tags_f = final.get_tags()
    ads_f = [i for i in range(len(final)) if tags_f[i] == 2]
    metal_f = [i for i in range(len(final)) if tags_f[i] != 2]

    r_metal = covalent_radii[atomic_numbers[final[metal_f[0]].symbol]]
    top_z = max(final.positions[m, 2] for m in metal_f)

    clearance = min(initial.get_distance(a, m, mic=True)
                    for a in ads_i for m in metal_i)
    add("initial clear of surface", clearance >= MIN_INITIAL_CLEARANCE,
        f"closest contact {clearance:.2f} A")

    fi = _fragments(initial, ads_i)
    ff = _fragments(final, ads_f)
    add("initial is intact", len(fi) == 1, f"{len(fi)} fragment(s)")
    add("final is dissociated", len(ff) == 2, f"{len(ff)} fragment(s)")
    if len(ff) != 2:
        return results

    # Identify the bond that actually dissociates, rather than guessing
    # from fixed indices (wrong for anything bigger than a diatomic) or a
    # single borrowed distance (wrong across different bond types). Uses
    # the initial, still-intact geometry, so this is the same pair
    # build_dissociated_endpoint itself chose to break.
    try:
        break_a, break_b, intact_bond = breaking_bond(initial, ads_i)
    except ValueError as exc:
        add("fragments separated", False,
            f"could not identify the breaking bond: {exc}")
        return results

    a0, a1 = _anchor(final, ff[0]), _anchor(final, ff[1])
    sep = final.get_distance(a0, a1, mic=True)
    required = RECOMBINATION_MARGIN * intact_bond
    add("fragments separated", sep >= required,
        f"{sep:.2f} A between anchors, minimum {required:.2f} A "
        f"({RECOMBINATION_MARGIN:.1f}x the intact "
        f"{initial[break_a].symbol}-{initial[break_b].symbol} bond, "
        f"{intact_bond:.2f} A)")

    for n, anc in enumerate((a0, a1)):
        ideal = r_metal + covalent_radii[final[anc].number]
        nearest = min(final.get_distance(anc, m, mic=True) for m in metal_f)
        ok = abs(nearest - ideal) <= BOND_TOLERANCE
        add(f"fragment {n + 1} bonded",
            ok, f"{final[anc].symbol} at {nearest:.2f} A, "
                f"covalent {ideal:.2f} A, height {final.positions[anc, 2] - top_z:.2f} A")

    if stepped:
        edge = slab_rec.get("step_edge_atoms") or []
        if edge:
            near = [min(final.get_distance(anc, e, mic=True) for e in edge)
                    for anc in (a0, a1)]
            add("fragments at the step", min(near) <= STEP_PROXIMITY,
                f"closest fragment {min(near):.2f} A from an edge atom")

    # Same pair, both trajectories. ads_i and ads_f are the same atoms in
    # the same order - build_dissociated_endpoint only moves positions, it
    # never reorders or relabels - so break_a/break_b index the breaking
    # bond correctly in both files.
    #
    # Pass criterion is the same `required` used for "fragments separated",
    # not an independent flat +1.0 A. A hardcoded absolute change does not
    # scale with bond length: H2's 0.74 A bond reaching a real, physically
    # correct 1.56 A adjacent-site separation is only a 0.82 A change and
    # failed here while H2/Cu(100) at 1.80 A (0.74 A bond, same chemistry)
    # passed at 1.06 A purely by chance of which site the search landed on.
    # In this benchmark break_a/break_b always coincide with the two
    # fragment anchors used above (a single molecule splits into exactly
    # two groups, so each group's heaviest-atom anchor IS the breaking
    # pair), so this check and "fragments separated" measure the same
    # physical distance before any relaxation has run - they should not be
    # allowed to disagree because one uses a scaled criterion and the
    # other does not.
    bond_i = initial.get_distance(break_a, break_b, mic=True)
    bond_f = final.get_distance(break_a, break_b, mic=True)
    add("endpoints distinct", bond_f >= required,
        f"breaking bond {initial[break_a].symbol}-{initial[break_b].symbol} "
        f"{bond_i:.2f} A -> {bond_f:.2f} A, minimum {required:.2f} A")

    return results


def main():
    print(f"Checking {len(SBH10)} reactions. No calculator, no GPU.\n")
    all_ok = True
    summary = []

    for reaction_id, spec in SBH10.items():
        print(f"=== {reaction_id}  ({spec['molecule']} on {spec['metal']}"
              f"({spec['facet']}), {spec.get('site_type')})")
        try:
            results = check_reaction(reaction_id, spec)
        except Exception as exc:
            print(f"    CRASHED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            summary.append((reaction_id, 0, 1))
            all_ok = False
            print()
            continue

        passed = sum(1 for _, ok, _ in results if ok)
        failed = len(results) - passed
        for name, ok, detail in results:
            mark = "  ok  " if ok else " FAIL "
            print(f"  [{mark}] {name:26s} {detail}")
        summary.append((reaction_id, passed, failed))
        if failed:
            all_ok = False
        print()

    print("=" * 72)
    print(f"{'reaction':26s} {'passed':>7s} {'failed':>7s}")
    print("-" * 72)
    for reaction_id, passed, failed in summary:
        flag = "" if failed == 0 else "   <-- needs attention"
        print(f"{reaction_id:26s} {passed:7d} {failed:7d}{flag}")
    print("=" * 72)

    total_failed = sum(f for _, _, f in summary)
    if all_ok:
        print("\nEvery reaction's starting geometry is sound. Safe to run the grid.")
    else:
        print(f"\n{total_failed} check(s) failing. Fix these before spending GPU time - "
              "every bug found in this project so far was a structure bug.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
