"""
Stop the Validation agent citing things no tool gave it.

    python patch_report_honesty.py --check
    python patch_report_honesty.py

Run from the repo root. Edits src/prompt.py.

WHY
---
On 2026-09-22 a validated N2/Ru(0001) run ended with a report saying the
0.43 eV underestimate was "within the known systematic tendency of
GGA-level MLFFs" and "consistent with published benchmarks for this
model class". No tool in this project retrieves literature, so both
claims were invented. The barrier itself was correct and validated; the
sentences explaining it were not supportable, and they read as credible.

The simulation prompt already forbids recalling benchmark values, but
the final report is written by the Validation agent, whose prompt had no
equivalent rule. This adds one there.
"""

import sys
from pathlib import Path

PROMPT = Path("src/prompt.py")

OLD = '''Never assert that a result is acceptable when a check has failed.
"""'''

NEW = '''Never assert that a result is acceptable when a check has failed.

Your final report states only what the tools in this run returned. Do not
explain a result by appeal to published benchmarks, literature, known
trends of a model class, or anything else no tool gave you: nothing
available to you can retrieve those, so any such claim would be invented,
however plausible it sounds. Do not compare against a reference value;
you have not been given one. If the reason for a number is not in the
tool outputs, say that the reason is not established by this run.
"""'''


def main():
    if not PROMPT.exists():
        print(f"FAILED: {PROMPT} not found. Run from the repo root.")
        return 1
    text = PROMPT.read_text()
    if "Your final report states only what the tools" in text:
        print("Already patched. Nothing to do.")
        return 0
    n = text.count(OLD)
    if n != 1:
        print(f"FAILED: anchor appears {n} times, expected exactly 1. "
              "Nothing written.")
        return 1
    if "--check" in sys.argv:
        print("Anchor found. Patch would apply cleanly.")
        return 0
    backup = Path(str(PROMPT) + ".pre_report_honesty")
    if not backup.exists():
        backup.write_text(text)
    PROMPT.write_text(text.replace(OLD, NEW, 1))
    print(f"Patched {PROMPT}. Backup: {backup.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
