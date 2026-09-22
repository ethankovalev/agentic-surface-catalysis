"""
Make every dispersion record tell the truth, and make the consistency
check test consistency.

    python patch_dispersion_truth.py --check
    python patch_dispersion_truth.py

Run from the repo root. Edits src/calculators.py, src/tools.py, src/zpe.py.
Supersedes patch_force_d3_metadata.py, which covered four of the five
store sites and none of the comparisons. Do not apply both.

TWO BUGS THAT CANCELLED
-----------------------
1. check_dispersion_consistent tested set(values) == {True}. That is not
   a consistency test; it fails every consistent D3-off run. Its message
   said D3 "should be on for an RPBE-trained model", the opposite of this
   project's own headline finding.

2. Under FORCE_D3=off the calculator genuinely ran without dispersion,
   but every record stored the agent's with_d3 ARGUMENT, which defaults
   to True. The grid_v2 H2/Cu(111) run on 2026-09-22 was launched with
   --d3 off and its records read initial_relaxed=True, final_relaxed=True.

Bug 2 hid bug 1. Every D3-off agent result that validated, five of the
six on the autonomous track, passed dispersion_consistent because the
records lied, not because the check worked. The energies are sound -
every stage really was D3-off - but the validation was right for the
wrong reason. A pipeline that passes with_d3=False honestly, as
scripts/run_blind.py does, could never validate a D3-off run at all.

WHY ALL FOUR PARTS AT ONCE
--------------------------
Fixing the records alone would break the agent track. zpe.py compared
the stored NEB setting against the with_d3 ARGUMENT it was called with.
Under FORCE_D3=off with honest records, the NEB would record False while
the agent still passed True, and every D3-off zero-point correction
would refuse. So the comparison moves to the effective setting in the
same patch, and the patch applies all of it or none of it.

  - calculators.effective_with_d3, the single source of truth
  - all five store sites record it, including refine_saddle_robust's
    saddle record, added after the earlier patch was written
  - zpe compares against it
  - check_dispersion_consistent requires one recorded setting, on or
    off, across all four stages
"""

import sys
from pathlib import Path

CALC = Path("src/calculators.py")
TOOLS = Path("src/tools.py")
ZPE = Path("src/zpe.py")

STORE_SITE = '"with_d3": bool(with_d3),'
EFFECTIVE = '"with_d3": effective_with_d3(with_d3),'
EXPECTED_SITES = {TOOLS: 4, ZPE: 1}

CALC_OLD = '''    _forced = os.environ.get("FORCE_D3")
    if _forced is not None:
        with_d3 = _forced.lower() in ("1", "true", "on", "yes")'''
CALC_NEW = '''    with_d3 = effective_with_d3(with_d3)'''

CALC_HELPER_OLD = "def new_calculator("
CALC_HELPER_NEW = '''def effective_with_d3(with_d3: bool) -> bool:
    """The dispersion setting actually in force, FORCE_D3 included.

    The single source of truth. new_calculator builds with it, every tool
    records it, and zpe compares against it. Storing the with_d3 argument
    instead records what was asked for, not what was used, and under
    FORCE_D3 the two differ.
    """
    forced = os.environ.get("FORCE_D3")
    if forced is not None:
        return forced.lower() in ("1", "true", "on", "yes")
    return bool(with_d3)


def new_calculator('''

IMPORT_OLD = "from src.calculators import new_calculator"
IMPORT_NEW = "from src.calculators import effective_with_d3, new_calculator"

ZPE_CMP_OLD = 'if "with_d3" in neb and bool(neb["with_d3"]) != bool(with_d3):'
ZPE_CMP_NEW = ('if "with_d3" in neb and bool(neb["with_d3"]) != '
               'effective_with_d3(with_d3):')

CHECK_OLD = '''    passed = (set(settings.values()) == {True})

    detail = ", ".join(f"{k}={v}" for k, v in settings.items())
    if not passed:
        detail += (" - every stage must use the same setting, and D3 should "
                   "be on for an RPBE-trained model")'''
CHECK_NEW = '''    # One recorded setting, on or off, across all four stages. The earlier
    # test was set(values) == {True}, which failed every consistent D3-off
    # run; D3-off is the better setting on every model this project has
    # tested, so it must be able to pass.
    values = set(settings.values())
    passed = len(values) == 1 and values <= {True, False}

    detail = ", ".join(f"{k}={v}" for k, v in settings.items())
    if not passed:
        detail += (" - every stage must record the same dispersion setting; "
                   "a barrier assembled from mixed settings is not on any "
                   "single potential energy surface")'''


def main():
    check_only = "--check" in sys.argv
    for p in (CALC, TOOLS, ZPE):
        if not p.exists():
            print(f"FAILED: {p} not found. Run from the repo root.")
            return 1

    calc, tools, zpe = CALC.read_text(), TOOLS.read_text(), ZPE.read_text()

    if "def effective_with_d3" in calc and EFFECTIVE in tools:
        print("Already patched. Nothing to do.")
        return 0
    if "def effective_with_d3" in calc or EFFECTIVE in tools:
        print("FAILED: a partial earlier fix is present (probably "
              "patch_force_d3_metadata.py). Restore the .pre_ backups first.")
        return 1

    problems = []
    for text, old, where in ((calc, CALC_OLD, CALC),
                             (calc, CALC_HELPER_OLD, CALC),
                             (tools, IMPORT_OLD, TOOLS),
                             (zpe, IMPORT_OLD, ZPE),
                             (zpe, ZPE_CMP_OLD, ZPE),
                             (tools, CHECK_OLD, TOOLS)):
        if text.count(old) != 1:
            problems.append(f"{where}: anchor found {text.count(old)} times: "
                            f"{old.strip().splitlines()[0][:60]}")
    for path, text in ((TOOLS, tools), (ZPE, zpe)):
        n = text.count(STORE_SITE)
        if n != EXPECTED_SITES[path]:
            problems.append(f"{path}: {n} store sites, expected "
                            f"{EXPECTED_SITES[path]}")
    if problems:
        print("FAILED, nothing written:")
        for p in problems:
            print(f"  {p}")
        return 1

    if check_only:
        print("All anchors and all five store sites found. "
              "Patch would apply cleanly.")
        return 0

    for p, text in ((CALC, calc), (TOOLS, tools), (ZPE, zpe)):
        b = Path(str(p) + ".pre_dispersion_truth")
        if not b.exists():
            b.write_text(text)

    calc = calc.replace(CALC_OLD, CALC_NEW, 1)
    calc = calc.replace(CALC_HELPER_OLD, CALC_HELPER_NEW, 1)
    tools = tools.replace(IMPORT_OLD, IMPORT_NEW, 1)
    tools = tools.replace(STORE_SITE, EFFECTIVE)
    tools = tools.replace(CHECK_OLD, CHECK_NEW, 1)
    zpe = zpe.replace(IMPORT_OLD, IMPORT_NEW, 1)
    zpe = zpe.replace(STORE_SITE, EFFECTIVE)
    zpe = zpe.replace(ZPE_CMP_OLD, ZPE_CMP_NEW, 1)

    CALC.write_text(calc)
    TOOLS.write_text(tools)
    ZPE.write_text(zpe)
    print("Patched calculators.py, tools.py (4 sites, consistency check), "
          "zpe.py (1 site, comparison).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
