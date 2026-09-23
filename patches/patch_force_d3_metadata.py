"""
Fix the FORCE_D3 metadata bug (found 2026-09-15, CH4_Ni111_step).

    python patch_force_d3_metadata.py --check
    python patch_force_d3_metadata.py

THE BUG
-------
new_calculator's FORCE_D3 override reassigned its own local `with_d3`
parameter. Python passes booleans by value, so that never reached the
caller. The calculator was built correctly, but relax_structure, run_neb,
refine_saddle and compute_zpe_correction each stored their OWN with_d3
argument - what the agent asked for, not what was used. Result: a file
named ..._d3off.json whose every record claimed with_d3=True.

Physics was fine (that run's 0.675 eV classical barrier matches the
seeded D3-off value of 0.727 eV, nowhere near D3-on's 0.177 eV). Only
the labels lied. check_dispersion_consistent couldn't catch it because
every stage was wrong in the same direction.

THE FIX
-------
One helper, effective_with_d3(), is the single source of truth. The
calculator and all four store sites call it, so they cannot drift apart
again.
"""

import sys
from pathlib import Path

HELPER = '''def effective_with_d3(with_d3: bool) -> bool:
    """The dispersion setting actually in force, FORCE_D3 included.

    Every caller that records "with_d3" must call this before storing,
    rather than storing its own argument - the argument is what was
    requested, not necessarily what was used.
    """
    forced = os.environ.get("FORCE_D3")
    if forced is not None:
        return forced.lower() in ("1", "true", "on", "yes")
    return with_d3


def new_calculator('''

# (file, old, new). Applied in order, one replacement each.
EDITS = [
    # the helper, inserted just above new_calculator
    ("src/calculators.py", "def new_calculator(", HELPER),

    # the override now delegates to it
    ("src/calculators.py",
     '''    _forced = os.environ.get("FORCE_D3")
    if _forced is not None:
        with_d3 = _forced.lower() in ("1", "true", "on", "yes")''',
     '''    with_d3 = effective_with_d3(with_d3)'''),

    ("src/tools.py",
     "from src.calculators import new_calculator",
     "from src.calculators import effective_with_d3, new_calculator"),

    # three store sites; surrounding lines make each one unique
    ("src/tools.py",
     '''        "shift_from_neb_peak_eV": shift,
        "with_d3": bool(with_d3),''',
     '''        "shift_from_neb_peak_eV": shift,
        "with_d3": effective_with_d3(with_d3),'''),

    ("src/tools.py",
     '''        "converged": bool(converged),
        "with_d3": bool(with_d3),
        "fmax_target": fmax,''',
     '''        "converged": bool(converged),
        "with_d3": effective_with_d3(with_d3),
        "fmax_target": fmax,'''),

    ("src/tools.py",
     '''        "reaction_energy_eV": float(reaction_energy),
        "converged": bool(converged),
        "with_d3": bool(with_d3),''',
     '''        "reaction_energy_eV": float(reaction_energy),
        "converged": bool(converged),
        "with_d3": effective_with_d3(with_d3),'''),

    ("src/zpe.py",
     "from src.calculators import new_calculator",
     "from src.calculators import effective_with_d3, new_calculator"),

    ("src/zpe.py",
     '''        "model_key": model_key,
        "with_d3": bool(with_d3),''',
     '''        "model_key": model_key,
        "with_d3": effective_with_d3(with_d3),'''),
]

FILES = sorted({f for f, _, _ in EDITS})


def main():
    check_only = "--check" in sys.argv

    for path in FILES:
        if not Path(path).exists():
            print(f"FAILED: {path} does not exist. Run from the repo root.")
            return 1

    if all("effective_with_d3" in Path(f).read_text() for f in FILES):
        print("Already patched. Nothing to do.")
        return 0

    # Check every anchor first, so a missing one means nothing gets written.
    # Each match is blanked as it is found, so two edits cannot both claim
    # the same text.
    probe = {f: Path(f).read_text() for f in FILES}
    for path, old, _ in EDITS:
        if old not in probe[path]:
            print(f"FAILED: anchor not found in {path}:")
            print(f"  {old.splitlines()[0]}")
            print("Nothing written. Apply by hand.")
            return 1
        probe[path] = probe[path].replace(old, "", 1)

    if check_only:
        print(f"All {len(EDITS)} anchors found. Patch would apply cleanly.")
        return 0

    for path in FILES:
        backup = Path(path + ".pre_force_d3_fix")
        if not backup.exists():
            backup.write_text(Path(path).read_text())

    for path, old, new in EDITS:
        p = Path(path)
        p.write_text(p.read_text().replace(old, new, 1))

    print(f"Patched {len(EDITS)} sites across {len(FILES)} files.")
    print("Backups saved as *.pre_force_d3_fix")
    return 0


if __name__ == "__main__":
    sys.exit(main())
