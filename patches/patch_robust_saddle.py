"""
Add refine_saddle_robust: saddle refinement that recovers from the two
failure modes this project has actually hit, instead of reporting them
and stopping.

    python patch_robust_saddle.py --check src/tools.py src/prompt.py
    python patch_robust_saddle.py src/tools.py src/prompt.py

refine_saddle is left EXACTLY as it is. This adds a tool beside it.

THE TWO FAILURES THIS ADDRESSES, FROM NOTES.md
------------------------------------------------
1. BASIN COLLAPSE. Sella walks out of the saddle region into a
   minimum, and the mode count comes back 0.
     CH4_Ni100:      bond drifted -0.786 A, ended at 1.124 A against a
                     1.07 A intact bond. It fell back to the reactant.
     CH4_Ni111_step: drifted +0.509 A to 2.268 A, in the product basin.
   Cause: the first Sella step is large enough to leave the region
   where the saddle's negative curvature dominates. Recovery: retry
   with a smaller initial trust radius so early steps stay local.

2. RIDGE. Two imaginary modes, so the point is a second order saddle,
   not a transition state.
     CH4_Ru0001: second mode ~10 meV, stable across attempts
     H2_Cu100:   second mode ~23 meV, over 4x the noise floor
   Cause: the optimiser sits on a ridge between two equivalent lower
   symmetry saddles rather than on either one. Recovery: displace
   along the SECOND imaginary mode's eigenvector, which is the
   direction that descends off the ridge, then refine again. This is
   the standard treatment, and it is also the Cu(100) symmetry
   hypothesis the README lists as untested.

WHAT IT DOES NOT DO
-------------------
It does not help the near barrierless cases, H2/Pt(111) and
H2/Ru(0001), where the reference is about zero. There is no saddle of
any consequence to converge onto, and no amount of retrying invents
one. Those stay reported as what they are.

Every attempt is recorded. A run that needed two recoveries is not
presented as though it converged first time.
"""

import sys
from pathlib import Path


TOOLS_OLD = '''@tool
def refine_saddle(model_key: str = None, with_d3: bool = True,'''

TOOLS_NEW = '''def _imaginary_modes_with_vectors(atoms, indices, name):
    """Imaginary modes in meV, with their eigenvectors.

    _count_imaginary returns magnitudes only, which is all the checks
    need. Recovery from a ridge needs the direction as well: to step off
    a second order saddle you have to know which way the second
    negative curvature points.

    Returns (list of (meV, displacement array), n_modes). Each
    displacement is shaped like atoms.positions, zero everywhere except
    the analysed atoms, and normalised so its largest single atomic
    displacement is 1 A. The caller scales it.
    """
    import shutil
    shutil.rmtree(name, ignore_errors=True)
    try:
        vib = Vibrations(atoms, indices=indices, name=name)
        vib.run()
        energies = vib.get_energies()
        found = []
        for i, e in enumerate(energies):
            meV = abs(e.imag) * 1000.0
            if np.iscomplex(e) and meV > SADDLE_IMAG_MIN_meV:
                mode = np.asarray(vib.get_mode(i), dtype=float)
                largest = np.abs(mode).max()
                if largest > 0:
                    mode = mode / largest
                found.append((meV, mode))
    finally:
        shutil.rmtree(name, ignore_errors=True)
    found.sort(key=lambda pair: pair[0], reverse=True)
    return found, len(energies)


def _breaking_bond_length(atoms):
    """Length of the bond being broken, or None if it cannot be found."""
    try:
        tags = atoms.get_tags()
        ads = [i for i in range(len(atoms)) if tags[i] == 2]
        if len(ads) < 2:
            return None
        a, b, _ = breaking_bond(atoms, ads)
        return float(atoms.get_distance(a, b, mic=True))
    except Exception:
        return None


def _run_sella(atoms, max_steps, delta0=None):
    """One Sella run. delta0 is the initial trust radius; None is Sella's
    own default, a smaller value keeps early steps local."""
    from sella import Sella
    kwargs = {"order": 1, "internal": False,
              "trajectory": _path("saddle.traj"), "logfile": "-"}
    if delta0 is not None:
        kwargs["delta0"] = delta0
    dyn = Sella(atoms, **kwargs)
    converged = bool(dyn.run(fmax=SADDLE_FMAX, steps=max_steps))
    return converged, int(dyn.get_number_of_steps())


@tool
def refine_saddle_robust(model_key: str = None, with_d3: bool = True,
                         max_steps: int = 200, scope: str = "adsorbate",
                         max_attempts: int = 3) -> str:
    """Refine onto a transition state, recovering from a failed attempt.

    refine_saddle makes one attempt from the NEB peak and reports what it
    gets. When that lands in a basin (zero imaginary modes) or on a ridge
    (two or more), this retries with a strategy chosen to match the
    failure, instead of stopping.

    Zero imaginary modes means the optimiser left the saddle region and
    fell into a minimum. The retry uses a smaller initial trust radius so
    early steps stay local.

    Two or more means a second order saddle. The retry displaces along
    the second imaginary mode, the direction that descends off the ridge,
    and refines from there.

    Neither helps a genuinely barrierless reaction, where there is no
    saddle to find. Those are reported as such rather than retried.

    max_attempts caps the total number of Sella runs, including the
    first. Every attempt is recorded in the store under
    saddle_attempts, so a result that needed recovery is visible as
    such rather than looking like a clean first pass.

    Run after run_neb, in place of refine_saddle. Reads work/peak.traj,
    writes work/saddle.traj.
    """
    peak_file = Path(_path("peak.traj"))
    if not peak_file.exists():
        return ("FAILED: no peak.traj. Run run_neb first - refinement starts "
                "from the highest image of a converged band, not from scratch.")

    neb = store.get("neb")
    if not neb:
        return "FAILED: no NEB record in the store. Run run_neb first."

    if scope not in ("adsorbate", "mobile"):
        return (f"FAILED: unknown scope '{scope}'. Use 'adsorbate' or "
                "'mobile'. An unrecognised value must not silently fall "
                "back to a default.")

    if max_attempts < 1:
        return "FAILED: max_attempts must be at least 1."

    initial = store.get("initial_relaxed") or {}
    e_initial = initial.get("energy_eV")
    if e_initial is None:
        return ("FAILED: no relaxed initial state. The refined barrier is "
                "measured from it, so relax the initial endpoint first.")

    model_key = model_key or config.DEFAULT_MODEL

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        import sella  # noqa: F401
    except ImportError:
        return ("FAILED: sella is not installed. pip install sella. It is in "
                "requirements.txt; this environment predates that entry.")

    peak_atoms = read(str(peak_file))
    r_peak = _breaking_bond_length(peak_atoms)
    e_peak = None

    attempts = []
    atoms = None
    strategy = "from the NEB peak, default trust radius"
    delta0 = None
    displacement = None

    for attempt in range(1, max_attempts + 1):
        atoms = read(str(peak_file))
        atoms.calc = new_calculator(model_key, with_d3=with_d3)
        if e_peak is None:
            e_peak = float(atoms.get_potential_energy())
        if displacement is not None:
            atoms.positions = atoms.positions + displacement

        converged, n_steps = _run_sella(atoms, max_steps, delta0)
        e_after = float(atoms.get_potential_energy())

        tags = atoms.get_tags()
        ads = [i for i in range(len(atoms)) if tags[i] == 2]
        indices = ads if scope == "adsorbate" else _mobile_indices(atoms)
        if not indices:
            return ("FAILED: no atoms to analyse - the adsorbate is untagged "
                    "or every atom is constrained.")

        try:
            modes, n_modes = _imaginary_modes_with_vectors(
                atoms, indices, _path("vib_saddle"))
        except Exception as exc:
            return (f"FAILED: mode analysis raised "
                    f"{type(exc).__name__}: {exc}. Without a mode count "
                    "there is no way to tell a saddle from a minimum, so "
                    "this is not reported as a result.")

        imag = [meV for meV, _ in modes]
        r_now = _breaking_bond_length(atoms)
        drift = None if (r_now is None or r_peak is None) else r_now - r_peak

        attempts.append({
            "attempt": attempt,
            "strategy": strategy,
            "converged": converged,
            "n_steps": n_steps,
            "energy_eV": e_after,
            "imaginary_modes_meV": imag,
            "r_b_A": r_now,
            "r_b_drift_A": drift,
        })

        if len(imag) == 1:
            break

        if attempt == max_attempts:
            break

        # choose the next strategy from how this attempt failed
        if len(imag) == 0:
            # basin collapse: keep the steps local this time
            delta0 = 0.005
            displacement = None
            strategy = "retry with a smaller trust radius after basin collapse"
        else:
            # ridge: step off it along the second imaginary mode
            second = modes[1][1]
            displacement = 0.1 * second
            delta0 = None
            strategy = (f"retry displaced 0.1 A along the second imaginary "
                        f"mode ({imag[1]:.1f} meV) after a ridge")

    final = attempts[-1]
    imag = final["imaginary_modes_meV"]
    first_order = len(imag) == 1
    barrier = final["energy_eV"] - e_initial

    record = {
        "barrier_eV": barrier,
        "energy_eV": final["energy_eV"],
        "converged": final["converged"],
        "n_steps": final["n_steps"],
        "shift_from_neb_peak_eV": final["energy_eV"] - e_peak,
        "with_d3": effective_with_d3(with_d3),
        "fmax_target": SADDLE_FMAX,
        "scope": scope,
        "n_modes": n_modes,
        "imaginary_modes_meV": imag,
        "first_order_saddle": first_order,
        "neb_barrier_eV": neb.get("barrier_eV"),
        "r_b_drift_A": final["r_b_drift_A"],
        "n_attempts": len(attempts),
        "recovered": len(attempts) > 1 and first_order,
    }
    store.put("saddle", record)
    store.put("saddle_attempts", attempts)
    store.put("model_key", model_key)
    write(_path("saddle.traj"), atoms)

    lines = []
    for a in attempts:
        drift_txt = ("unknown" if a["r_b_drift_A"] is None
                     else f"{a['r_b_drift_A']:+.3f} A")
        lines.append(
            f"  attempt {a['attempt']} ({a['strategy']}): "
            f"{len(a['imaginary_modes_meV'])} imaginary "
            f"{[round(m, 1) for m in a['imaginary_modes_meV']]}, "
            f"bond drift {drift_txt}")
    trail = "\\n".join(lines)

    if first_order:
        head = (f"Refined to a first-order saddle after {len(attempts)} "
                f"attempt(s). Barrier {barrier:.3f} eV, one imaginary mode "
                f"at {imag[0]:.1f} meV.")
        if len(attempts) > 1:
            head += (" This needed recovery; the first attempt did not "
                     "give a transition state. Say so when reporting.")
    elif len(imag) == 0:
        head = (f"NOT A SADDLE after {len(attempts)} attempt(s): zero "
                "imaginary modes, so every attempt fell into a minimum. "
                "If the reference barrier for this reaction is near zero "
                "the reaction may simply be barrierless, which no amount "
                "of refinement will change.")
    else:
        head = (f"NOT A FIRST-ORDER SADDLE after {len(attempts)} attempt(s): "
                f"{len(imag)} imaginary modes {[round(m, 1) for m in imag]}. "
                "Stepping off the ridge did not reach a simple saddle.")

    return f"{head}\\n{trail}"


@tool
def refine_saddle(model_key: str = None, with_d3: bool = True,'''


REG_OLD = '''    compute_zpe_correction,
    check_run_quality,
    read_results,
]'''

REG_NEW = '''    compute_zpe_correction,
    refine_saddle_robust,
    check_run_quality,
    read_results,
]'''


PROMPT_OLD = '''After compute_zpe_correction succeeds, call check_run_quality once.'''

PROMPT_NEW = '''If refine_saddle does not give exactly one imaginary mode, call
refine_saddle_robust instead of accepting the result or rerunning the
band. It retries with a strategy matched to the failure: a smaller
trust radius when the optimiser fell into a minimum, a displacement
along the second imaginary mode when it landed on a ridge. It cannot
help a genuinely barrierless reaction, where there is no saddle to
find, so do not loop on one. If it reports that recovery was needed,
say so in your summary rather than presenting the result as a clean
first pass.

After compute_zpe_correction succeeds, call check_run_quality once.'''


EDITS = [
    (Path("src/tools.py"), "refine_saddle_robust + helpers", TOOLS_OLD, TOOLS_NEW),
    (Path("src/tools.py"), "SIMULATION_TOOLS registration", REG_OLD, REG_NEW),
    (Path("src/prompt.py"), "simulation prompt guidance", PROMPT_OLD, PROMPT_NEW),
]


def main():
    check_only = "--check" in sys.argv

    if "refine_saddle_robust" in Path("src/tools.py").read_text():
        print("Already patched. Nothing to do.")
        return 0

    for path, name, old, _ in EDITS:
        if not path.exists():
            print(f"FAILED: {path} does not exist. Run from the repo root.")
            return 1
        found = path.read_text().count(old)
        if found != 1:
            print(f"FAILED: the {name} anchor appears {found} times in "
                  f"{path}, expected exactly 1.")
            print("Nothing written. Apply by hand.")
            return 1

    if check_only:
        print(f"All {len(EDITS)} anchors found. Patch would apply cleanly.")
        return 0

    for path in {p for p, _, _, _ in EDITS}:
        backup = Path(str(path) + ".pre_robust_saddle")
        if not backup.exists():
            backup.write_text(path.read_text())

    for path, name, old, new in EDITS:
        path.write_text(path.read_text().replace(old, new, 1))
        print(f"Patched {path} ({name})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
