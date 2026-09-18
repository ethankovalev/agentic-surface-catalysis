"""
Wire the escalation assessment into run_one, after the graph returns.

    python patch_wire_escalation.py --check src/benchmark.py
    python patch_wire_escalation.py src/benchmark.py

Requires src/escalation.py to be present.

WHY AFTER THE GRAPH, NOT INSIDE IT
-----------------------------------
The assessment reads a finished run and reports a level. It is NOT a
tool the agent can call.

That is deliberate, for the same reason src/benchmark.py is
tool-unreachable: an agent that can read the verdict can optimise
against it. The point of a verification layer is that the thing being
verified cannot see the test. Keeping the assessment outside the
agent's reach preserves that, and as a side effect costs no API credit,
since it runs on data the graph already produced.

A mid-run variant, where the agent sees "your reaction coordinate mode
is weak, consider retrying", is genuinely the next step toward
intelligent planning. It needs an answer to "what stops the agent
gaming it" that this design does not yet have, so it is not built here.

WHAT IT ADDS TO EACH RESULT
----------------------------
    "assessment": {"level": ..., "blocking": [...], "review": [...],
                   "notes": [...]}

`validated` is untouched: exit_gate remains the authority on whether a
run counts. The assessment is a second, more granular opinion sitting
beside it, which is why a run can be validated=True and CAUTION at once.
"""

import sys
from pathlib import Path


OLD_IMPORT = "from src import store"
NEW_IMPORT = "from src import store\nfrom src.escalation import assess"

OLD_RETURN = '''    checks = store.validation()
    ref = spec.get("reference_eV")
    error = None if (computed is None or ref is None) else computed - ref

    return {
        "computed_eV": computed,
        "reference_eV": ref,
        "reference_leaked": bool(leaks),
        "reference_leaks": leaks,
        "error_eV": error,
        "reaction_class": spec.get("reaction_class"),
        "validation": checks,
        "validation_detail": store.get("validation_detail", {}),
        "validated": store.all_checks_passed(),
        "run_error": error_note,
        "trace": store.snapshot(),
    }'''

NEW_RETURN = '''    checks = store.validation()
    ref = spec.get("reference_eV")
    error = None if (computed is None or ref is None) else computed - ref

    # Post-hoc assessment of the finished run. Never sees the reference,
    # and the agent never sees it: see patch_wire_escalation.py on why
    # this sits outside the graph rather than inside it. Guarded because
    # a crash here must not throw away a completed GPU run.
    snapshot = store.snapshot()
    try:
        assessment = assess(snapshot)
    except Exception as exc:
        assessment = {
            "level": "BLOCKED",
            "blocking": [f"assessment itself failed: "
                        f"{type(exc).__name__}: {exc}"],
            "review": [],
            "notes": [],
        }

    return {
        "computed_eV": computed,
        "reference_eV": ref,
        "reference_leaked": bool(leaks),
        "reference_leaks": leaks,
        "error_eV": error,
        "reaction_class": spec.get("reaction_class"),
        "validation": checks,
        "validation_detail": store.get("validation_detail", {}),
        "validated": store.all_checks_passed(),
        "assessment": assessment,
        "run_error": error_note,
        "trace": snapshot,
    }'''

OLD_PRINT = '''        print(f"  computed={r['computed_eV']} ref={r['reference_eV']} "
              f"error={r['error_eV']} validated={r['validated']}")'''

NEW_PRINT = '''        print(f"  computed={r['computed_eV']} ref={r['reference_eV']} "
              f"error={r['error_eV']} validated={r['validated']} "
              f"assessment={r.get('assessment', {}).get('level')}")
        for reason in r.get("assessment", {}).get("review", []):
            print(f"    review: {reason}")'''


EDITS = [("import", OLD_IMPORT, NEW_IMPORT),
         ("run_one return", OLD_RETURN, NEW_RETURN),
         ("summary print", OLD_PRINT, NEW_PRINT)]


def main():
    check_only = "--check" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(args[0]) if args else Path("src/benchmark.py")

    if not path.exists():
        print(f"FAILED: {path} does not exist. Run from the repo root.")
        return 1
    if not Path("src/escalation.py").exists():
        print("FAILED: src/escalation.py not found. Add it first.")
        return 1

    text = path.read_text()
    if "from src.escalation import assess" in text:
        print("Already patched. Nothing to do.")
        return 0

    for name, old, _ in EDITS:
        if old not in text:
            print(f"FAILED: could not find the {name} block in {path}.")
            print("Nothing written. Apply by hand.")
            return 1

    if check_only:
        print(f"All {len(EDITS)} anchors found. Patch would apply cleanly.")
        return 0

    backup = Path(str(path) + ".pre_escalation")
    if not backup.exists():
        backup.write_text(text)

    for _, old, new in EDITS:
        text = text.replace(old, new, 1)
    path.write_text(text)

    print(f"Patched {path} ({len(EDITS)} sites).")
    print("Backup saved as", backup.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
