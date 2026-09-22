"""
Two guards, both prompted by the grid_v2 run of 2026-09-22.

    python patch_agent_guards.py --check
    python patch_agent_guards.py

Run from the repo root. Edits src/tools.py and src/benchmark.py.

GUARD 1: DO NOT REBUILD OVER FINISHED WORK
-------------------------------------------
H2/Cu(111) made 67 tool calls and ran out of supervisor turns. N2, which
validated, made 17. The sequence shows why: build_slab, place_adsorbate
and build_dissociated_endpoint each appear FOUR times. The agent built
the system, relaxed both endpoints, ran the band, refined a saddle,
computed a gas-referenced barrier and a zero-point correction, called
read_results, and then rebuilt the slab from scratch and started again.
Three times. It exhausted its budget redoing work it had already
finished, and the run was abandoned with path_resolved and zpe failing.

The validation prompt already warns against exactly this, calling it "an
escalating loop that recomputes the same thing at ever greater cost
while the real result sits already finished in the store". Prompt text
did not hold. This makes it structural: the three builder tools refuse
once the store holds relaxed endpoints or a saddle, and say what exists
and to call read_results instead.

The refusal is not absolute. A rebuild is sometimes correct, for
instance after discovering the wrong site was used, so each tool takes
force=True. The agent must then choose to discard the work explicitly
rather than drift into it. Scripted callers are unaffected: they call
store.reset() first, which clears the store.

GUARD 2: THE LEAK SCAN ONLY LOOKED FOR ONE NUMBER
--------------------------------------------------
The same run's N2 report said its barrier was "nowhere near the ~0.40 eV
step-site value". 0.40 eV is the SBH10 reference for N2_Ru0001_step. No
tool supplied it; the model recalled it. scan_for_reference_leak looked
only for the reference of the reaction being run, 1.84, so the run was
recorded as unleaked.

Two changes. The scan now checks every SBH10 reference, not just the
current one, and reports which reaction each hit belongs to. And the cue
list gains comparison phrasing, because the sentence above carries no
existing cue: "value" alone is not one, only "true value" and "actual
value" were. A number matching a dataset reference next to "nowhere
near" or "compared to" is a recalled value, not a coincidence.

This does not change any computed barrier. It changes what the record
says about how blind the run was, which the README currently overstates.
"""

import sys
from pathlib import Path

TOOLS = Path("src/tools.py")
BENCH = Path("src/benchmark.py")

GUARD_HELPER_OLD = "@tool\ndef build_slab(metal: str, facet: str = \"111\", nx: int = 3, ny: int = 3,"
GUARD_HELPER_NEW = '''def _work_already_done():
    """What the store holds that a rebuild would throw away.

    Returns a list of plain descriptions, empty when there is nothing to
    lose. Used by the three builder tools so that rebuilding over a
    finished calculation has to be deliberate.
    """
    done = []
    for key, label in (("initial_relaxed", "a relaxed initial state"),
                       ("final_relaxed", "a relaxed final state"),
                       ("saddle", "a refined saddle"),
                       ("barrier_eV", "a computed barrier")):
        value = store.get(key)
        if value is not None:
            done.append(label)
    return done


def _refuse_rebuild(tool_name, force):
    """None if the rebuild may proceed, else the refusal to return."""
    if force:
        return None
    done = _work_already_done()
    if not done:
        return None
    return (
        f"REFUSED: {tool_name} would rebuild the system, but this run already "
        f"has {', '.join(done)}. Rebuilding discards all of it and starts the "
        f"calculation again, which is how a run exhausts its turns without "
        f"finishing. Call read_results to see what exists, and continue from "
        f"there. If the existing work really is unusable, say why and call "
        f"this tool again with force=True.")


@tool
def build_slab(metal: str, facet: str = "111", nx: int = 3, ny: int = 3,'''

BUILD_SLAB_SIG_OLD = '''               layers: int = 4, vacuum: float = 10.0,
               dopant: str = "none", n_fixed_layers: int = 2) -> str:'''
BUILD_SLAB_SIG_NEW = '''               layers: int = 4, vacuum: float = 10.0,
               dopant: str = "none", n_fixed_layers: int = 2,
               force: bool = False) -> str:'''

PLACE_SIG_OLD = '''def place_adsorbate(species: str, height: float = 2.5,
                    site: str = "ontop", overhang: float = 0.5) -> str:'''
PLACE_SIG_NEW = '''def place_adsorbate(species: str, height: float = 2.5,
                    site: str = "ontop", overhang: float = 0.5,
                    force: bool = False) -> str:'''

ENDPOINT_SIG_OLD = '''def build_dissociated_endpoint(separation: float = None,
                               height: float = None) -> str:'''
ENDPOINT_SIG_NEW = '''def build_dissociated_endpoint(separation: float = None,
                               height: float = None,
                               force: bool = False) -> str:'''

LEAK_CUES_OLD = '''_LEAK_CUES = (
    "sbh10", "reference", "experimental", "experiment", "literature",
    "known", "reported", "published", "expected", "should be", "accepted",
    "benchmark value", "true value", "actual value",
)'''
LEAK_CUES_NEW = '''_LEAK_CUES = (
    "sbh10", "reference", "experimental", "experiment", "literature",
    "known", "reported", "published", "expected", "should be", "accepted",
    "benchmark value", "true value", "actual value",
    # Comparison phrasing. A number matching a dataset reference beside one
    # of these was recalled, not computed: on 2026-09-22 a report said its
    # barrier was "nowhere near the ~0.40 eV step-site value", quoting the
    # reference for a different reaction, and no cue above appears in it.
    "nowhere near", "compared to", "compared with", "versus", " vs ",
    "far from the", "close to the", "step-site", "step site",
    "terrace value", "regime",
)'''

SCAN_SIG_OLD = "def scan_for_reference_leak(messages, reference_eV):"
SCAN_SIG_NEW = "def scan_for_reference_leak(messages, reference_eV, others=None):"

SCAN_BODY_OLD = '''    if reference_eV is None:
        return []
    tol = max(_LEAK_TOL, abs(reference_eV) * _LEAK_TOL)
    hits = []'''
SCAN_BODY_NEW = '''    # Every reference in the set, not just this reaction's. The model has
    # read SBH10; quoting any of its numbers shows recall, whichever
    # reaction it belongs to.
    targets = []
    if reference_eV is not None:
        targets.append(("this reaction", float(reference_eV)))
    for name, value in (others or {}).items():
        if value is None or name == "this reaction":
            continue
        if reference_eV is not None and abs(float(value) - reference_eV) < 1e-9:
            continue
        targets.append((name, float(value)))
    if not targets:
        return []
    hits = []'''

SCAN_LOOP_OLD = '''            if abs(value - reference_eV) > tol:
                continue
            a = max(0, match.start() - 120)
            b = min(len(body), match.end() + 60)
            cue = next((c for c in _LEAK_CUES if c in low[a:b]), None)
            if cue is None:
                continue
            hits.append({
                "message_index": i,
                "speaker": getattr(message, "name", None)
                           or type(message).__name__,
                "value_eV": value,
                "cue": cue,
                "snippet": " ".join(body[a:b].split())[:200],
            })'''
SCAN_LOOP_NEW = '''            matched = None
            for name, target in targets:
                tol = max(_LEAK_TOL, abs(target) * _LEAK_TOL)
                if abs(value - target) <= tol:
                    matched = name
                    break
            if matched is None:
                continue
            a = max(0, match.start() - 120)
            b = min(len(body), match.end() + 60)
            cue = next((c for c in _LEAK_CUES if c in low[a:b]), None)
            if cue is None:
                continue
            hits.append({
                "message_index": i,
                "speaker": getattr(message, "name", None)
                           or type(message).__name__,
                "value_eV": value,
                "matched_reaction": matched,
                "cue": cue,
                "snippet": " ".join(body[a:b].split())[:200],
            })'''

CALL_OLD = '''        leaks = scan_for_reference_leak(
            (final_state or {}).get("messages"), spec.get("reference_eV"))'''
CALL_NEW = '''        leaks = scan_for_reference_leak(
            (final_state or {}).get("messages"), spec.get("reference_eV"),
            others={name: entry.get("reference_eV")
                    for name, entry in SBH10.items() if name != reaction_id})'''

TOOL_EDITS = [
    ("rebuild guard helpers", GUARD_HELPER_OLD, GUARD_HELPER_NEW),
    ("build_slab signature", BUILD_SLAB_SIG_OLD, BUILD_SLAB_SIG_NEW),
    ("place_adsorbate signature", PLACE_SIG_OLD, PLACE_SIG_NEW),
    ("build_dissociated_endpoint signature", ENDPOINT_SIG_OLD, ENDPOINT_SIG_NEW),
]
BENCH_EDITS = [
    ("leak cues", LEAK_CUES_OLD, LEAK_CUES_NEW),
    ("scan signature", SCAN_SIG_OLD, SCAN_SIG_NEW),
    ("scan targets", SCAN_BODY_OLD, SCAN_BODY_NEW),
    ("scan loop", SCAN_LOOP_OLD, SCAN_LOOP_NEW),
    ("run_one call site", CALL_OLD, CALL_NEW),
]

SLAB_GUARD = '''    refusal = _refuse_rebuild("build_slab", force)
    if refusal:
        return refusal

'''

PLACE_GUARD = '''    refusal = _refuse_rebuild("place_adsorbate", force)
    if refusal:
        return refusal

'''
ENDPOINT_GUARD = '''    refusal = _refuse_rebuild("build_dissociated_endpoint", force)
    if refusal:
        return refusal

'''


def insert_after_docstring(text, func_signature, guard):
    """Put the guard immediately after the function's docstring."""
    start = text.index(func_signature)
    q = text.index('"""', start)
    end = text.index('"""', q + 3) + 3
    line_end = text.index("\n", end) + 1
    return text[:line_end] + guard + text[line_end:]


def main():
    check_only = "--check" in sys.argv
    for p in (TOOLS, BENCH):
        if not p.exists():
            print(f"FAILED: {p} not found. Run from the repo root.")
            return 1
    tools, bench = TOOLS.read_text(), BENCH.read_text()

    if "_refuse_rebuild" in tools and "matched_reaction" in bench:
        print("Already patched. Nothing to do.")
        return 0

    problems = []
    for name, old, _ in TOOL_EDITS:
        if tools.count(old) != 1:
            problems.append(f"src/tools.py: {name} anchor found "
                            f"{tools.count(old)} times")
    for name, old, _ in BENCH_EDITS:
        if bench.count(old) != 1:
            problems.append(f"src/benchmark.py: {name} anchor found "
                            f"{bench.count(old)} times")
    if problems:
        print("FAILED, nothing written:")
        for p in problems:
            print(f"  {p}")
        return 1

    if check_only:
        print(f"All {len(TOOL_EDITS) + len(BENCH_EDITS)} anchors found. "
              "Patch would apply cleanly.")
        return 0

    for p, text in ((TOOLS, tools), (BENCH, bench)):
        b = Path(str(p) + ".pre_agent_guards")
        if not b.exists():
            b.write_text(text)

    for name, old, new in TOOL_EDITS:
        tools = tools.replace(old, new, 1)
    tools = insert_after_docstring(tools, BUILD_SLAB_SIG_NEW, SLAB_GUARD)
    tools = insert_after_docstring(tools, PLACE_SIG_NEW, PLACE_GUARD)
    tools = insert_after_docstring(tools, ENDPOINT_SIG_NEW, ENDPOINT_GUARD)
    for name, old, new in BENCH_EDITS:
        bench = bench.replace(old, new, 1)

    TOOLS.write_text(tools)
    BENCH.write_text(bench)
    print("Patched src/tools.py (rebuild guard on three builders) and "
          "src/benchmark.py (leak scan over every reference).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
