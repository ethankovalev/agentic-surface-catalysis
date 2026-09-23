"""
Option B: check connectivity INSIDE the retry loop, so a saddle that
belongs to a different process triggers another attempt rather than
being reported as the answer.

    python patch_connectivity_in_loop.py --check src/tools.py
    python patch_connectivity_in_loop.py src/tools.py

Apply after patch_drift_guard.py.

WHY, FROM REAL DATA
--------------------
CH4/Ni(100), 2026-09-19: refine_saddle_robust returned a first order
saddle at 2.387 A with one imaginary mode at 40.6 meV and a barrier of
0.629 eV. Pushing the breaking bond 0.35 A in BOTH directions relaxed
back to ~2.39 A, so it is a minimum along the C-H coordinate. The
40.6 meV mode points along some other motion entirely and 0.629 eV is
not the dissociation barrier.

One imaginary mode proves a first order saddle. It does not prove which
reaction that saddle belongs to. This is the documented reason
check_saddle_connects exists, and the same failure that produced a
confirmed saddle for subsurface H penetration on H2/Cu(111) earlier in
this project.

WHAT THE RECOVERY IS, AND WHY IT IS PRINCIPLED HERE
-----------------------------------------------------
On the seeded track peak.traj IS the published transition state. The
refinement then drifted 0.478 A away from it and landed on a different
saddle. The unrefined barrier at the published geometry was 0.900 eV
against a 0.760 reference, error +0.140; the refined one was 0.629,
error -0.131. Refinement moved AWAY from the right answer.

So when connectivity fails, the recovery is a smaller trust radius:
keep the steps local, stay in the neighbourhood of the geometry that
was already correct. That is the same recovery as basin collapse, and
for the same underlying reason - the optimiser went somewhere it should
not have.

COST, STATED PLAINLY
---------------------
Each connectivity check is two short relaxations (0.35 A push, 60 step
cap). Running it per attempt with max_attempts=3 adds up to six short
relaxations to a tool that previously ran three Sella calls and three
Hessians. Roughly a 30 to 50 percent increase in wall time per
reaction, not a tripling, because the Hessians dominate.

It is skipped entirely for attempts that already failed on mode count,
since there is no saddle to check connectivity for.
"""

import sys
from pathlib import Path


OLD_HELPER = '''def _geometry_is_still_a_saddle(atoms, r_seed):'''

NEW_HELPER = '''CONNECTIVITY_PUSH_A = 0.35
CONNECTIVITY_STEPS = 60


def _connectivity_by_bond(atoms, model_key, with_d3):
    """Push the breaking bond both ways; see where each side falls.

    Weaker than following the imaginary mode, and named accordingly.
    What it rules out is a saddle sitting somewhere other than between
    the intact molecule and the dissociated fragments.

    Returns (connects, detail). connects is None when the breaking bond
    cannot be identified, which is not the same as False and must not be
    treated as one.
    """
    found = _breaking_bond(atoms)
    if found is None:
        return None, "breaking bond could not be identified"

    anchor, terminal, r_saddle = found
    intact = (covalent_radii[atoms[anchor].number]
              + covalent_radii[atoms[terminal].number])

    axis = atoms.positions[terminal] - atoms.positions[anchor]
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        return None, "breaking bond has zero length"
    axis = axis / norm

    out = {}
    for label, sign in (("compressed", -1.0), ("stretched", +1.0)):
        trial = atoms.copy()
        trial.positions[terminal] += sign * CONNECTIVITY_PUSH_A * axis
        trial.calc = new_calculator(model_key, with_d3=with_d3)
        BFGS(trial, logfile=None).run(fmax=0.05, steps=CONNECTIVITY_STEPS)
        out[label] = float(trial.get_distance(anchor, terminal, mic=True))

    recombined = bool(out["compressed"] < intact * 1.35)
    dissociated = bool(out["stretched"] > max(r_saddle, intact) * 1.25)
    connects = bool(recombined and dissociated)

    if connects:
        detail = (f"compressed to {out['compressed']:.2f} A, stretched to "
                  f"{out['stretched']:.2f} A")
    else:
        why = []
        if not recombined:
            why.append(f"compressing left it at {out['compressed']:.2f} A "
                       f"instead of falling back toward {intact:.2f} A")
        if not dissociated:
            why.append(f"stretching left it at {out['stretched']:.2f} A "
                       "instead of running away to dissociation")
        detail = "; ".join(why)

    return connects, detail


def _geometry_is_still_a_saddle(atoms, r_seed):'''


OLD_ACCEPT = '''        geometry_ok, geometry_reason = _geometry_is_still_a_saddle(
            atoms, r_peak)
        attempts[-1]["geometry_ok"] = geometry_ok
        if geometry_reason:
            attempts[-1]["geometry_reason"] = geometry_reason

        if len(imag) == 1 and geometry_ok:
            break'''

NEW_ACCEPT = '''        geometry_ok, geometry_reason = _geometry_is_still_a_saddle(
            atoms, r_peak)
        attempts[-1]["geometry_ok"] = geometry_ok
        if geometry_reason:
            attempts[-1]["geometry_reason"] = geometry_reason

        # Connectivity only matters for something that is already a
        # first order saddle with a sane geometry. Checking it otherwise
        # spends two relaxations to confirm what is already known.
        connects = None
        connect_detail = ""
        if len(imag) == 1 and geometry_ok:
            connects, connect_detail = _connectivity_by_bond(
                atoms, model_key, with_d3)
            attempts[-1]["connects"] = connects
            attempts[-1]["connectivity_detail"] = connect_detail

        # connects is None when the bond could not be identified. That
        # is an unknown, not a failure, and must not silently reject a
        # result that may be correct.
        if len(imag) == 1 and geometry_ok and connects is not False:
            break'''


OLD_STRATEGY = '''        if len(imag) == 1 and not geometry_ok:
            delta0 = 0.005
            displacement = None
            strategy = ("retry with a smaller trust radius after the "
                        "geometry collapsed despite a single mode")'''

NEW_STRATEGY = '''        if len(imag) == 1 and geometry_ok and connects is False:
            # A real saddle, for the wrong reaction. On the seeded track
            # the starting geometry is the published transition state, so
            # the cure is to stop wandering away from it.
            delta0 = 0.005
            displacement = None
            strategy = ("retry with a smaller trust radius after landing "
                        "on a saddle that does not connect reactant to "
                        "product")
        elif len(imag) == 1 and not geometry_ok:
            delta0 = 0.005
            displacement = None
            strategy = ("retry with a smaller trust radius after the "
                        "geometry collapsed despite a single mode")'''


OLD_FINAL = '''    first_order = len(imag) == 1 and final.get("geometry_ok", True)'''

NEW_FINAL = '''    first_order = (len(imag) == 1
                   and final.get("geometry_ok", True)
                   and final.get("connects") is not False)'''


OLD_RECORD = '''        "r_b_drift_A": final["r_b_drift_A"],'''

NEW_RECORD = '''        "r_b_drift_A": final["r_b_drift_A"],
        "connects": final.get("connects"),
        "connectivity_detail": final.get("connectivity_detail", ""),
        "connectivity_method": "bond_displacement",
        "connectivity_note": (
            "weaker than mode following; confirms the saddle lies between "
            "reactant and product along the bond coordinate only"),'''


OLD_HEAD = '''    if len(imag) == 1 and not final.get("geometry_ok", True):'''

NEW_HEAD = '''    if len(imag) == 1 and final.get("connects") is False:
        head = (f"SADDLE FOR A DIFFERENT REACTION after {len(attempts)} "
                f"attempt(s): one imaginary mode at {imag[0]:.1f} meV and a "
                f"sane geometry, but it does not connect reactant to "
                f"product - {final.get('connectivity_detail', '')}. The "
                "barrier is not this reaction's.")
    elif len(imag) == 1 and not final.get("geometry_ok", True):'''


EDITS = [
    ("connectivity helper", OLD_HELPER, NEW_HELPER),
    ("accept decision", OLD_ACCEPT, NEW_ACCEPT),
    ("strategy selection", OLD_STRATEGY, NEW_STRATEGY),
    ("first_order definition", OLD_FINAL, NEW_FINAL),
    ("stored record", OLD_RECORD, NEW_RECORD),
    ("report head", OLD_HEAD, NEW_HEAD),
]


def main():
    check_only = "--check" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(args[0]) if args else Path("src/tools.py")

    if not path.exists():
        print(f"FAILED: {path} does not exist. Run from the repo root.")
        return 1

    text = path.read_text()
    if "_connectivity_by_bond" in text:
        print("Already patched. Nothing to do.")
        return 0
    if "_geometry_is_still_a_saddle" not in text:
        print("FAILED: apply patch_drift_guard.py first.")
        return 1

    for name, old, _ in EDITS:
        found = text.count(old)
        if found != 1:
            print(f"FAILED: the {name} anchor appears {found} times, "
                  "expected exactly 1. Nothing written.")
            return 1

    if check_only:
        print(f"All {len(EDITS)} anchors found. Patch would apply cleanly.")
        return 0

    backup = Path(str(path) + ".pre_connectivity_loop")
    if not backup.exists():
        backup.write_text(text)

    for name, old, new in EDITS:
        text = text.replace(old, new, 1)
        print(f"  patched: {name}")
    path.write_text(text)
    print(f"Patched {path}. Backup: {backup.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
