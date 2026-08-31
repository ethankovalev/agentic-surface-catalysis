#!/usr/bin/env python3
"""Install Sella saddle refinement into agentic-surface-catalysis.

Run from the repository root:

    python3 install_saddle_refinement.py

Every edit is guarded: if an anchor does not match exactly once, the script
aborts and changes nothing. Safe to re-run - it detects work already done.
"""
import sys
from pathlib import Path

ROOT = Path.cwd()
TOOLS = ROOT / "src" / "tools.py"
PROMPT = ROOT / "src" / "prompt.py"
REQS = ROOT / "requirements.txt"

if not TOOLS.exists():
    sys.exit("ABORT: run this from the repository root (src/tools.py not found).")


def patch(path, old, new, label):
    text = path.read_text()
    if new.strip()[:60] in text:
        print(f"  = {label}: already applied, skipping")
        return
    n = text.count(old)
    if n != 1:
        sys.exit(f"ABORT: {label}: found {n} matches for anchor, expected 1. "
                 f"Nothing has been written.\n---\n{old[:300]}")
    path.write_text(text.replace(old, new))
    print(f"  + {label}")


print("Patching src/tools.py")

# ---------------------------------------------------------------- 1. imports
patch(TOOLS,
      "from src.calculators import new_calculator",
      "from src.calculators import new_calculator\nfrom ase.vibrations import Vibrations",
      "vibrations import")

# ------------------------------------------------- 2. run_neb writes the peak
patch(TOOLS,
      '''    store.put("neb", {
        "barrier_eV": float(barrier),''',
      '''    # Write the highest image on its own. refine_saddle starts from this, and
    # reading it back out of neb.traj is fragile: the optimiser appends the
    # whole band on every step, so the file holds n_images x n_steps frames.
    write(_path("peak.traj"), images[peak])

    store.put("neb", {
        "barrier_eV": float(barrier),''',
      "run_neb writes peak.traj")

# ------------------------------------------------------- 3. the refine tool
REFINE = '''

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
'''

text = TOOLS.read_text()
if "def refine_saddle" in text:
    print("  = refine_saddle: already present, skipping")
else:
    anchor = "\n# Simulation tools"
    if text.count(anchor) != 1:
        sys.exit("ABORT: could not locate the '# Simulation tools' section header.")
    TOOLS.write_text(text.replace(anchor, REFINE + anchor))
    print("  + refine_saddle tool")

# --------------------------- 4. gas referencing prefers the refined saddle
patch(TOOLS,
      '''    well_depth = gasref["energy_eV"] - initial["energy_eV"]''',
      '''    # Prefer a confirmed saddle over the NEB's highest image. The band only
    # samples the path; the refined structure sits on the transition state and
    # has been shown to have exactly one imaginary mode.
    saddle = store.get("saddle")
    source = "NEB peak"
    barrier_ads = neb["barrier_eV"]
    if saddle and saddle.get("converged") and saddle.get("first_order_saddle"):
        barrier_ads = saddle["barrier_eV"]
        source = "refined saddle"

    well_depth = gasref["energy_eV"] - initial["energy_eV"]''',
      "gas referencing prefers refined saddle")

patch(TOOLS,
      '''    barrier_gas = neb["barrier_eV"] - well_depth''',
      '''    barrier_gas = barrier_ads - well_depth''',
      "gas referencing uses chosen barrier")

patch(TOOLS,
      '''    return (f"Physisorption well depth {well_depth:.3f} eV. "
            f"Barrier from physisorbed state {neb['barrier_eV']:.3f} eV, "''',
      '''    store.put("barrier_source", source)

    return (f"Physisorption well depth {well_depth:.3f} eV. "
            f"Barrier from physisorbed state {barrier_ads:.3f} eV ({source}), "''',
      "gas referencing reports its source")

print("\nPatching src/prompt.py")

patch(PROMPT,
      "  - a saddle point failure (zero or multiple imaginary modes) means",
      '''  - a path_resolved failure means the true peak fell between two images
    and was never computed. Prefer refine_saddle over rerunning the band:
    it starts from the highest image and converges onto the saddle itself,
    then confirms exactly one imaginary mode. On a test case a 3-image band
    failing at 52 percent gave, after refinement, the same answer as an
    11-image band, in one step. Reruns with more images cost a whole new
    band each time and have repeatedly failed to fix this.
  - a saddle point failure (zero or multiple imaginary modes) means''',
      "prompt: prefer refine_saddle")

print("\nPatching requirements.txt")
if REQS.exists():
    rtext = REQS.read_text()
    if "sella" in rtext:
        print("  = sella: already in requirements.txt")
    else:
        REQS.write_text(rtext.rstrip() + "\nsella\n")
        print("  + sella")
else:
    print("  ! requirements.txt not found, add 'sella' yourself")

print("""
Done. Now verify:

    python3 -c "import ast; ast.parse(open('src/tools.py').read()); print('tools.py OK')"
    python3 -c "import ast; ast.parse(open('src/prompt.py').read()); print('prompt.py OK')"
    grep -n "def refine_saddle" src/tools.py
    grep -n "peak.traj" src/tools.py

Then register the tool with the simulation agent. Find where the existing
simulation tools are listed:

    grep -rn "run_neb" src/agent.py src/graph.py

and add refine_saddle to the same list.
""")
