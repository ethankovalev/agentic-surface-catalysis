"""
Fix scripts/run_grid.py: resume logic treats corrupted results as done.

Run:  python patch_grid_resume.py --check scripts/run_grid.py
      python patch_grid_resume.py scripts/run_grid.py

WHAT WAS WRONG
--------------
The resume check was `if rp.exists(): skip`. run_one() catches exceptions
INSIDE itself for at least one failure mode (an Anthropic API error during
a supervisor turn) and returns a result dict with computed_eV=None and a
run_error field describing the failure, rather than raising - so the
outer loop's own try/except never sees it, the result file gets written
exactly like a real one, and every future run of this script silently
treats "the API rejected every request for this reaction" the same as
"this reaction is done and validated".

That is what happened on 2026-09-12: a mid-sweep Anthropic credit
exhaustion produced 11 placeholder files, all skipped as "done" by
anything that ran afterward, discovered only because of a manual audit
script - not because run_grid.py itself noticed anything wrong.

THE FIX
-------
Skip only if the result file exists AND does not carry a run_error. A
placeholder from a real failure gets picked back up on the next run
automatically; nothing needs to be found and rm'd by hand again.
"""

import sys
from pathlib import Path

OLD = '''        if rp.exists():
            print(f"[{i}/{len(plan)}] {reaction_id} d3={d3_label} -> done, skipping")
            skipped += 1
            continue'''

NEW = '''        if rp.exists():
            # A result file existing is not the same as a result being
            # good: run_one() can catch an exception internally (an
            # Anthropic API error mid-supervisor-turn is the case that
            # actually happened) and return computed_eV=None with a
            # run_error field instead of raising, so the outer try/except
            # below never sees it and a corrupted placeholder gets written
            # exactly like a real result. Re-check the content, not just
            # the file's existence, or a credit-exhaustion placeholder
            # sits there forever looking done to every future run.
            try:
                prior = json.loads(rp.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                print(f"[{i}/{len(plan)}] {reaction_id} d3={d3_label} -> "
                      f"existing result file unreadable ({exc}), re-running")
                prior = {"run_error": f"unreadable prior result: {exc}"}

            if not prior.get("run_error"):
                print(f"[{i}/{len(plan)}] {reaction_id} d3={d3_label} -> done, skipping")
                skipped += 1
                continue

            print(f"[{i}/{len(plan)}] {reaction_id} d3={d3_label} -> "
                  f"prior result carries run_error "
                  f"({str(prior['run_error'])[:80]}), re-running")'''


def apply(path: Path, check_only: bool) -> int:
    text = path.read_text()

    if NEW in text:
        print("Already patched. Nothing to do.")
        return 0
    if OLD not in text:
        print("PATCH DID NOT APPLY: could not find the original skip block.")
        print("The file has moved on since this patch was written; apply by hand.")
        return 1

    if "^import json" not in text and "\nimport json\n" not in text:
        print("NOTE: scripts/run_grid.py does not appear to `import json` at "
              "module level. It is used elsewhere in the file for "
              "json.dump(result, f, ...), so this is almost certainly already "
              "imported - if the patched script fails with NameError: json, "
              "add `import json` near the top of the file.")

    if check_only:
        print("Skip block found. Patch would apply cleanly.")
        return 0

    text = text.replace(OLD, NEW, 1)
    backup = path.with_suffix(path.suffix + ".pre_resume_fix")
    if not backup.exists():
        backup.write_text(path.read_text())
    path.write_text(text)
    print(f"Patched {path}. Original saved to {backup.name}.")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    target = Path(args[0]) if args else Path("scripts/run_grid.py")
    if not target.exists():
        raise SystemExit(f"{target} does not exist. Pass the path to run_grid.py.")
    raise SystemExit(apply(target, "--check" in sys.argv))
