"""
Add --only and --limit to scripts/run_grid.py, for running a sweep on a
budget you can actually bound in advance.

Run:  python patch_grid_budget.py --check scripts/run_grid.py
      python patch_grid_budget.py scripts/run_grid.py

WHY
---
run_grid.py's only controls were --models, --d3 and --dry-run. There was
no way to say "attempt two reactions and stop". With 11 corrupted results
queued for re-attempt and a small API balance, the only options were to
run all of them or to Ctrl+C mid-reaction and lose that reaction's
partial work.

That is how the 2026-09-12 sweep drained its balance: it ran until the
API refused, wrote placeholder results for everything after that point,
and reported completed=10 failed=0.

  --only A,B     restrict the plan to named reactions
  --limit N      stop after N reactions have actually RUN (skips do not
                 count toward the limit, so resuming a mostly-done sweep
                 with --limit 2 attempts exactly two real reactions)

--limit counts real runs, not plan entries, specifically so that it
behaves predictably on a partially complete sweep. Counting plan entries
would let a run of 9 skips plus 1 real run satisfy --limit 10 while doing
almost nothing.
"""

import sys
from pathlib import Path


OLD_ARGS = '''    parser.add_argument("--models", default=config.DEFAULT_MODEL)
    parser.add_argument("--d3", choices=["on", "off", "both"], default="on")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()'''

NEW_ARGS = '''    parser.add_argument("--models", default=config.DEFAULT_MODEL)
    parser.add_argument("--d3", choices=["on", "off", "both"], default="on")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", default=None,
                        help="comma-separated reaction ids to restrict the "
                             "plan to, e.g. H2_Cu100,H2_Pt111")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many reactions have actually "
                             "RUN. Skipped (already-complete) reactions do "
                             "not count, so --limit 2 on a mostly-finished "
                             "sweep attempts exactly two real reactions.")
    args = parser.parse_args()'''


OLD_PLAN = '''    plan = []
    for model_key in models:
        for reaction_id, spec in SBH10.items():
            for d3_label in d3_labels:
                plan.append((model_key, reaction_id, spec, d3_label))'''

NEW_PLAN = '''    only = None
    if args.only:
        only = [r.strip() for r in args.only.split(",") if r.strip()]
        unknown_r = [r for r in only if r not in SBH10]
        if unknown_r:
            raise SystemExit(
                f"Unknown reaction id(s): {unknown_r}. Known: {list(SBH10)}")

    plan = []
    for model_key in models:
        for reaction_id, spec in SBH10.items():
            if only is not None and reaction_id not in only:
                continue
            for d3_label in d3_labels:
                plan.append((model_key, reaction_id, spec, d3_label))'''


OLD_LOOP = '''    for i, (model_key, reaction_id, spec, d3_label) in enumerate(plan, 1):
        rp = result_path(reaction_id, model_key, d3_label)
        ep = error_path(reaction_id, model_key, d3_label)'''

NEW_LOOP = '''    for i, (model_key, reaction_id, spec, d3_label) in enumerate(plan, 1):
        # Counts reactions that actually ran, not plan entries seen, so the
        # limit means the same thing on a fresh sweep and on a resumed one.
        if args.limit is not None and completed + failed >= args.limit:
            print(f"\\n[limit] {args.limit} reaction(s) have run; stopping "
                  f"before {reaction_id} d3={d3_label}. "
                  f"Re-run to continue where this left off.")
            break

        rp = result_path(reaction_id, model_key, d3_label)
        ep = error_path(reaction_id, model_key, d3_label)'''


def apply(path: Path, check_only: bool) -> int:
    text = path.read_text()

    if '"--limit"' in text and '"--only"' in text:
        print("Already patched. Nothing to do.")
        return 0

    problems = []
    for name, old in (("argument block", OLD_ARGS),
                      ("plan construction", OLD_PLAN),
                      ("main loop head", OLD_LOOP)):
        if old not in text:
            problems.append(f"  - could not find the {name}")

    if problems:
        print("PATCH DID NOT APPLY:")
        print("\n".join(problems))
        print("Apply patch_grid_resume.py first if you have not already.")
        return 1

    if check_only:
        print("All three blocks found. Patch would apply cleanly.")
        return 0

    text = text.replace(OLD_ARGS, NEW_ARGS, 1)
    text = text.replace(OLD_PLAN, NEW_PLAN, 1)
    text = text.replace(OLD_LOOP, NEW_LOOP, 1)

    backup = path.with_suffix(path.suffix + ".pre_budget_fix")
    if not backup.exists():
        backup.write_text(path.read_text())
    path.write_text(text)
    print(f"Patched {path}. Original saved to {backup.name}.")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    target = Path(args[0]) if args else Path("scripts/run_grid.py")
    if not target.exists():
        raise SystemExit(f"{target} does not exist.")
    raise SystemExit(apply(target, "--check" in sys.argv))
