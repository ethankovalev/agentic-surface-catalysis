"""
Tolerate a negative physisorption well depth that is within the model's
resolution.

    python patch_well_depth_tolerance.py --check
    python patch_well_depth_tolerance.py

Run from the repo root. Edits src/tools.py.

WHY
---
check_gas_reference_applied failed any negative well depth, on the
reasoning that the lifted molecule cannot relax below the physisorbed
one unless one of the two is not a real minimum. True in principle. But
with D3 off, physisorption wells on these systems are a few meV: 3 meV
for H2/Cu(111), 9 meV for N2/Ru(0001) terrace. UMA's benchmarked
resolution is about 0.05 eV.

N2/Ru(0001) step, blind sweep of 2026-09-23, had a connected first-order
saddle at 0.982 eV and failed on this check alone, with a well depth of
-0.011 eV. That is noise, not a structural failure.

The tolerance is 0.02 eV, well below the model's resolution, so a
genuinely wrong minimum, which shows up as tenths of an eV, still fails.
"""

import sys
from pathlib import Path

TOOLS = Path("src/tools.py")

OLD = '''    elif well_depth < 0:
        passed = False'''

NEW = '''    elif well_depth < -WELL_DEPTH_TOLERANCE_eV:
        passed = False'''

CONST_OLD = '''@tool
def check_gas_reference_applied() -> str:'''

CONST_NEW = '''# A negative well depth this small is noise. With D3 off, physisorption
# wells here are a few meV (3 meV H2/Cu(111), 9 meV N2/Ru(0001)), and
# UMA's resolution is about 0.05 eV. N2/Ru(0001) step failed on -0.011 eV
# with a connected saddle in hand.
WELL_DEPTH_TOLERANCE_eV = 0.02


@tool
def check_gas_reference_applied() -> str:'''


def main():
    if not TOOLS.exists():
        print(f"FAILED: {TOOLS} not found. Run from the repo root.")
        return 1
    text = TOOLS.read_text()
    if "WELL_DEPTH_TOLERANCE_eV" in text:
        print("Already patched. Nothing to do.")
        return 0
    for name, old in (("well depth comparison", OLD),
                      ("check_gas_reference_applied", CONST_OLD)):
        if text.count(old) != 1:
            print(f"FAILED: {name} anchor found {text.count(old)} times. "
                  "Nothing written.")
            return 1
    if "--check" in sys.argv:
        print("Anchors found. Patch would apply cleanly.")
        return 0
    backup = Path(str(TOOLS) + ".pre_well_depth")
    if not backup.exists():
        backup.write_text(text)
    text = text.replace(OLD, NEW, 1).replace(CONST_OLD, CONST_NEW, 1)
    TOOLS.write_text(text)
    print("Patched check_gas_reference_applied: tolerance 0.02 eV.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
