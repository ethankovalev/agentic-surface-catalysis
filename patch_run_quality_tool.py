"""
Add check_run_quality: a mid-run tool the Simulation_Agent can call to
see the same physics-based assessment benchmark.py currently only
computes AFTER the graph returns.

    python patch_run_quality_tool.py --check src/tools.py src/prompt.py
    python patch_run_quality_tool.py src/tools.py src/prompt.py

Requires src/escalation.py.

THE GAMING QUESTION, ANSWERED BY DESIGN NOT BY TRUST
------------------------------------------------------
Giving an agent visibility into a quality signal creates an obvious
risk: can it write around a bad signal instead of fixing it? The
answer here is structural, not a hope that the agent behaves:

  - check_run_quality is READ-ONLY. It cannot write to the store, set
    validated, or touch exit_gate. Calling it, or not calling it,
    changes nothing about whether the run passes.
  - exit_gate still depends purely on check_saddle_connects, the
    imaginary mode counts, and the rest of REQUIRED_CHECKS, exactly as
    before this patch. Nothing about that machinery changed.
  - The only lever this hands the agent is EARLIER information about
    a signal that already existed and was already going to be computed
    post-hoc. It cannot be used to make a bad run look good; the worst
    it can do is be ignored, which is the same failure mode the project
    already caught once (CH4_Ni111_step) with no mid-run tool at all.

So this narrows, rather than widens, the chance of a misleading final
report: the agent now has the SAME information a moment earlier, while
its normal bounded retry budget (run_neb capped at two calls,
refine_saddle uncapped but inside the reaction's overall turn limit)
is still available to act on it.
"""

import sys
from pathlib import Path


# --- src/tools.py ---------------------------------------------------

TOOLS_OLD_ANCHOR = '''from src.zpe import check_zpe_applied, compute_zpe_correction  # noqa: E402

STRUCTURE_TOOLS = [build_slab, build_stepped_slab, place_adsorbate, build_dissociated_endpoint]
SIMULATION_TOOLS = [
    relax_structure,
    run_neb,
    refine_saddle,
    check_endpoints_are_minima,
    check_saddle_connects,
    build_gas_reference,
    compute_gas_referenced_barrier,
    compute_zpe_correction,
    read_results,
]'''

TOOLS_NEW_ANCHOR = '''from src.zpe import check_zpe_applied, compute_zpe_correction  # noqa: E402
# Imported here for the same one-way-dependency reason as zpe above:
# escalation reaches into store, never back into tools.
from src.escalation import assess  # noqa: E402


@tool
def check_run_quality() -> str:
    """Report on how confident this run should be, before finalising it.

    Runs the same physics based assessment benchmark.py applies after
    you finish, early: after compute_zpe_correction, while your normal
    retry budget is still available to act on it.

    This is informational only. It cannot mark a result validated and
    cannot override exit_gate: those still depend purely on
    check_saddle_connects, the imaginary mode counts, and the other
    checks you already run. Calling this tool, or not calling it,
    changes nothing about whether the run passes.

    What it is for: if the level comes back CAUTION or REVIEW, decide
    whether the reason is worth one more attempt within your existing
    retry budget. If you decide not to retry, report the concern
    plainly in your final summary rather than leaving it out. A
    CAUTION or REVIEW level reported honestly is a correct outcome. A
    concern smoothed over in the final report is not, and exit_gate
    will catch the underlying issue regardless of what the report says.
    """
    try:
        verdict = assess(store.snapshot())
    except Exception as exc:
        return (f"check_run_quality could not run: "
               f"{type(exc).__name__}: {exc}. This does not block "
               f"anything; continue as normal.")

    lines = [f"Run quality: {verdict['level']}"]
    for reason in verdict["blocking"]:
        lines.append(f"  blocking (exit_gate will refuse this): {reason}")
    for reason in verdict["review"]:
        lines.append(f"  review: {reason}")
    for note in verdict["notes"]:
        lines.append(f"  note: {note}")
    if verdict["level"] == "ACCEPTED":
        lines.append("  nothing measurable has been flagged.")
    return "\\n".join(lines)


STRUCTURE_TOOLS = [build_slab, build_stepped_slab, place_adsorbate, build_dissociated_endpoint]
SIMULATION_TOOLS = [
    relax_structure,
    run_neb,
    refine_saddle,
    check_endpoints_are_minima,
    check_saddle_connects,
    build_gas_reference,
    compute_gas_referenced_barrier,
    compute_zpe_correction,
    check_run_quality,
    read_results,
]'''


# --- src/prompt.py ----------------------------------------------------

PROMPT_OLD_ANCHOR = '''Report what you computed and stop. Do not judge whether the result is
correct (that is the validation agent's job.)
"""'''

PROMPT_NEW_ANCHOR = '''After compute_zpe_correction succeeds, call check_run_quality once. It
cannot change whether the run passes and cannot fix anything by itself:
it only tells you, a little earlier than you would otherwise find out,
whether something about this specific run looks unusual. CAUTION or
REVIEW is not a failure and not a reason to loop; it is a reason to
decide, once, whether the concern is worth one more attempt inside your
existing retry budget. Whatever you decide, put the concern in your
final summary in plain words. Do not write a report that reads as
settled when check_run_quality did not come back ACCEPTED.

Report what you computed and stop. Do not judge whether the result is
correct (that is the validation agent's job.)
"""'''


EDITS = [
    (Path("src/tools.py"), "check_run_quality tool + registration",
     TOOLS_OLD_ANCHOR, TOOLS_NEW_ANCHOR),
    (Path("src/prompt.py"), "simulation agent prompt guidance",
     PROMPT_OLD_ANCHOR, PROMPT_NEW_ANCHOR),
]


def main():
    check_only = "--check" in sys.argv

    if not Path("src/escalation.py").exists():
        print("FAILED: src/escalation.py not found. Add it first.")
        return 1

    already = all("check_run_quality" in p.read_text()
                  for p, _, _, _ in EDITS if p.exists())
    if already:
        print("Already patched. Nothing to do.")
        return 0

    # CLAUDE.md: "match an anchor exactly once, or change nothing". An
    # "at least once" check plus .replace(..., 1) would silently patch
    # the first of several occurrences and leave the rest, which is
    # exactly the plausible-looking-but-wrong failure this project
    # exists to catch.
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

    for path, name, old, new in EDITS:
        text = path.read_text()
        backup = Path(str(path) + ".pre_run_quality")
        if not backup.exists():
            backup.write_text(text)
        path.write_text(text.replace(old, new, 1))
        print(f"Patched {path} ({name}). Backup: {backup.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
