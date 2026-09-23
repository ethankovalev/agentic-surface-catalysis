"""
Add two things Sella already provides and this pipeline was not using:
IRC connectivity, and constrained refinement.

    python patch_irc_constraints.py --check src/tools.py
    python patch_irc_constraints.py src/tools.py

Apply after patch_connectivity_in_loop.py.

WHY, FROM REAL DATA
--------------------
CH4/Ni(100), seeded, 2026-09-19. peak.traj IS the published BEEF-vdW
transition state, so the starting geometry was already correct. Sella
then moved 0.478 A away from it and converged on a different saddle:

    at the published geometry, unrefined   0.900 eV, error +0.140
    after refinement                       0.629 eV, error -0.131

Refinement moved away from the right answer, and bond-displacement
connectivity confirmed the refined saddle does not connect reactant to
product. Two separate gaps caused that, and this patch closes both.

IRC, REPLACING BOND DISPLACEMENT AS THE PRIMARY TEST
------------------------------------------------------
_connectivity_by_bond pushes the breaking BOND and sees where it falls.
run_seeded.py's own docstring calls that the weaker test and says so
plainly: it confirms the saddle lies between reactant and product along
the bond coordinate, not that the reaction MODE connects them.

sella.IRC follows the imaginary mode itself, which is the stronger test.
It is run in both directions and the endpoints compared.

Two details taken from sella's source rather than guessed:

  - the argument is direction='forward' / 'reverse', and irun raises
    ValueError on anything else.
  - IRC caches its initial diagonalization in self.v0ts and REUSES it
    for the second direction. So the same IRC object must be run twice,
    forward then reverse. Constructing a second object would repeat the
    diagonalization and, worse, could pick a different sign convention
    for v0ts, silently giving two runs in the same direction.

Bond displacement is kept as the fallback, because IRC will not converge
on a near-flat surface, which is exactly the H2/Pt(111) and H2/Ru(0001)
regime where the reference barrier is about zero.

CONSTRAINED REFINEMENT
-----------------------
Sella's Constraints.fix_bond(indices, target=...) holds the breaking
bond at a chosen length while everything else relaxes. Used as a
recovery strategy: pin the bond at the value it had in peak.traj, refine,
then release and polish briefly unconstrained.

This addresses the wandering at source rather than detecting it
afterwards. The two-stage form matters: a constrained optimum is not a
true saddle in the full space, so the unconstrained polish is what makes
the result meaningful. The polish is capped short, so it can tighten a
good geometry without wandering off again.
"""

import sys
from pathlib import Path


OLD_HELPER = '''def _geometry_is_still_a_saddle(atoms, r_seed):'''

NEW_HELPER = '''IRC_FMAX = 0.05
IRC_STEPS = 60
# Deliberately short. Tested on EMT: 20 free steps after pinning the
# bond at 1.909 A let it run to 3.049 A, undoing the constraint
# completely. The drift guard would reject that, so it fails safe, but
# it would also make this strategy useless. The polish exists only to
# tighten a geometry the constrained stage already placed correctly.
CONSTRAINED_POLISH_STEPS = 8


def _connectivity_by_irc(atoms, model_key, with_d3):
    """Follow the imaginary mode both ways. The stronger connectivity test.

    Returns (connects, detail). connects is None when IRC cannot run or
    cannot converge, which is an unknown and NOT a failure: on a
    near-barrierless surface there is no meaningful mode to follow, and
    reporting that as "does not connect" would reject correct results.

    One IRC object is used for both directions on purpose. sella caches
    its initial diagonalization in self.v0ts and restores it when the
    direction changes; a second object would redo that diagonalization
    and could choose the opposite sign for v0ts, which would silently
    run the same direction twice.
    """
    found = _breaking_bond(atoms)
    if found is None:
        return None, "breaking bond could not be identified"
    anchor, terminal, r_saddle = found
    intact = (covalent_radii[atoms[anchor].number]
              + covalent_radii[atoms[terminal].number])

    try:
        from sella import IRC
    except ImportError:
        return None, "sella.IRC not available"

    ends = {}
    try:
        work = atoms.copy()
        work.calc = new_calculator(model_key, with_d3=with_d3)
        irc = IRC(work, trajectory=None, logfile=None, dx=0.1)
        for direction in ("forward", "reverse"):
            irc.run(fmax=IRC_FMAX, steps=IRC_STEPS, direction=direction)
            ends[direction] = float(
                work.get_distance(anchor, terminal, mic=True))
    except Exception as exc:
        return None, f"IRC did not run: {type(exc).__name__}: {exc}"

    lo, hi = sorted(ends.values())
    recombined = bool(lo < intact * 1.35)
    dissociated = bool(hi > max(r_saddle, intact) * 1.25)
    connects = bool(recombined and dissociated)

    if connects:
        detail = (f"IRC: one side to {lo:.2f} A, the other to {hi:.2f} A")
    else:
        why = []
        if not recombined:
            why.append(f"IRC's nearer end stopped at {lo:.2f} A instead of "
                       f"falling back toward {intact:.2f} A")
        if not dissociated:
            why.append(f"IRC's farther end stopped at {hi:.2f} A instead of "
                       "running away to dissociation")
        detail = "IRC: " + "; ".join(why)
    return connects, detail


def _check_connectivity(atoms, model_key, with_d3):
    """IRC first, bond displacement as the fallback.

    IRC is the stronger test but will not converge on a near-flat
    surface. When it returns an unknown, fall back rather than treat the
    unknown as a result.
    """
    connects, detail = _connectivity_by_irc(atoms, model_key, with_d3)
    if connects is not None:
        return connects, detail
    fallback, fdetail = _connectivity_by_bond(atoms, model_key, with_d3)
    return fallback, f"{detail}; fell back to bond displacement: {fdetail}"


def _refine_constrained(atoms, r_target, max_steps):
    """Refine with the breaking bond pinned, then polish it free.

    Stops the optimiser wandering out of the region it was asked about,
    which on the seeded track is the published transition state itself.
    A constrained optimum is not a saddle in the full space, so the
    short unconstrained polish afterwards is what makes the result
    meaningful; it is capped so it can tighten a good geometry without
    drifting away again.
    """
    from sella import Constraints, Sella

    found = _breaking_bond(atoms)
    if found is None:
        return False, 0, "breaking bond could not be identified"
    anchor, terminal, _ = found

    cons = Constraints(atoms)
    cons.fix_bond((anchor, terminal), target=r_target)
    dyn = Sella(atoms, order=1, internal=False, constraints=cons,
                trajectory=_path("saddle.traj"), logfile="-")
    dyn.run(fmax=SADDLE_FMAX, steps=max_steps)
    n_constrained = int(dyn.get_number_of_steps())

    free = Sella(atoms, order=1, internal=False,
                 trajectory=_path("saddle.traj"), logfile="-")
    converged = bool(free.run(fmax=SADDLE_FMAX,
                              steps=CONSTRAINED_POLISH_STEPS))
    total = n_constrained + int(free.get_number_of_steps())
    return converged, total, (
        f"bond pinned at {r_target:.3f} A for {n_constrained} steps, then "
        f"{int(free.get_number_of_steps())} free")


def _geometry_is_still_a_saddle(atoms, r_seed):'''


OLD_CONN_CALL = '''        if len(imag) == 1 and geometry_ok:
            connects, connect_detail = _connectivity_by_bond(
                atoms, model_key, with_d3)'''

NEW_CONN_CALL = '''        if len(imag) == 1 and geometry_ok:
            connects, connect_detail = _check_connectivity(
                atoms, model_key, with_d3)'''


OLD_SELLA_CALL = '''        converged, n_steps = _run_sella(atoms, max_steps, delta0)'''

NEW_SELLA_CALL = '''        if constrain_to is not None:
            converged, n_steps, how = _refine_constrained(
                atoms, constrain_to, max_steps)
            attempts_note = how
        else:
            converged, n_steps = _run_sella(atoms, max_steps, delta0)
            attempts_note = ""'''


OLD_INIT = '''    strategy = "from the NEB peak, default trust radius"
    delta0 = None
    displacement = None'''

NEW_INIT = '''    strategy = "from the NEB peak, default trust radius"
    delta0 = None
    displacement = None
    constrain_to = None'''


OLD_APPEND = '''            "r_b_drift_A": drift,
        })'''

NEW_APPEND = '''            "r_b_drift_A": drift,
            "note": attempts_note,
        })'''


OLD_STRATEGY = '''        if len(imag) == 1 and geometry_ok and connects is False:
            # A real saddle, for the wrong reaction. On the seeded track
            # the starting geometry is the published transition state, so
            # the cure is to stop wandering away from it.
            delta0 = 0.005
            displacement = None
            strategy = ("retry with a smaller trust radius after landing "
                        "on a saddle that does not connect reactant to "
                        "product")'''

NEW_STRATEGY = '''        if len(imag) == 1 and geometry_ok and connects is False:
            # A real saddle, for the wrong reaction. On the seeded track
            # the starting geometry is the published transition state, so
            # the cure is to stop wandering away from it. Pinning the
            # breaking bond does that directly; a smaller trust radius
            # only slows the wandering down.
            constrain_to = r_peak
            delta0 = None
            displacement = None
            strategy = ("retry with the breaking bond pinned at its "
                        "starting length, after landing on a saddle that "
                        "does not connect reactant to product")'''


EDITS = [
    ("IRC and constrained helpers", OLD_HELPER, NEW_HELPER),
    ("connectivity call", OLD_CONN_CALL, NEW_CONN_CALL),
    ("sella call", OLD_SELLA_CALL, NEW_SELLA_CALL),
    ("loop init", OLD_INIT, NEW_INIT),
    ("attempt record", OLD_APPEND, NEW_APPEND),
    ("strategy selection", OLD_STRATEGY, NEW_STRATEGY),
]


def main():
    check_only = "--check" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(args[0]) if args else Path("src/tools.py")

    if not path.exists():
        print(f"FAILED: {path} does not exist. Run from the repo root.")
        return 1

    text = path.read_text()
    if "_connectivity_by_irc" in text:
        print("Already patched. Nothing to do.")
        return 0
    if "_connectivity_by_bond" not in text:
        print("FAILED: apply patch_connectivity_in_loop.py first.")
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

    backup = Path(str(path) + ".pre_irc_constraints")
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
