"""
One table across every track, and a direct test of whether unphysical
structures go with large errors.

    python scripts/compare_tracks.py
    python scripts/compare_tracks.py --d3 off --model uma-s-1p1

No GPU, no API. Reads result files only.

Columns:
  seeded   barrier at the published TS geometry (no search)
  blind    scripted blind pipeline, no LLM          (results/blind)
  agent    autonomous agent, latest sweep           (results/grid_v2)
  old      autonomous agent, original sweep         (results/grid)

A result is shown with * when validated. "flags" counts structural
problems plus energy flags in the blind run's saved structures.

The final block asks the question directly: among blind runs, is the
mean absolute error larger where structures were flagged than where
they were clean? With ten reactions this is suggestive at most, and the
script says so.
"""

import argparse
import json
import os
from pathlib import Path

BASE = Path(os.environ.get("RESULTS_BASE",
                           "/workspace/agentic-surface-catalysis/results"))

REACTIONS = [
    "H2_Cu111", "H2_Cu100", "H2_Pt111", "H2_Ru0001", "N2_Ru0001_terrace",
    "N2_Ru0001_step", "CH4_Ru0001", "CH4_Ni100", "CH4_Ni111_terrace",
    "CH4_Ni111_step",
]


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def cell(record, key="computed_eV"):
    if not record:
        return "n/a", None
    v = record.get(key)
    if v is None:
        return "none", None
    mark = "*" if record.get("validated") else " "
    return f"{v:.3f}{mark}", v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="uma-s-1p1")
    ap.add_argument("--d3", choices=["on", "off"], default="off")
    args = ap.parse_args()
    tag = f"{args.model}_d3{args.d3}"

    seeded = load(BASE / "seeded" / f"seeded_{tag}.json") or {}

    head = (f"{'reaction':20s} {'ref':>6s} {'seeded':>8s} {'blind':>9s} "
            f"{'agent':>9s} {'old':>9s} {'flags':>6s}")
    print(f"model {args.model}, D3 {args.d3}.  * = validated\n")
    print(head)
    print("-" * len(head))

    counts = {"blind": 0, "agent": 0, "old": 0}
    clean_err, flagged_err = [], []

    for rid in REACTIONS:
        s = seeded.get(rid) or {}
        ref = s.get("reference_eV")
        s_txt, _ = cell(s, "barrier_at_reference_geometry_eV")
        s_txt = s_txt.strip()

        blind = load(BASE / "blind" / f"{rid}_{tag}.json")
        agent = load(BASE / "grid_v2" / f"{rid}_{tag}.json")
        old = load(BASE / "grid" / f"{rid}_{tag}.json")

        b_txt, b_val = cell(blind)
        a_txt, _ = cell(agent)
        o_txt, _ = cell(old)
        for name, rec in (("blind", blind), ("agent", agent), ("old", old)):
            if rec and rec.get("validated"):
                counts[name] += 1

        if ref is None:
            for rec in (blind, agent, old):
                if rec and rec.get("reference_eV") is not None:
                    ref = rec["reference_eV"]
                    break

        flags = "n/a"
        if blind:
            n = (blind.get("n_structural_problems", 0)
                 + len(blind.get("energy_flags", [])))
            flags = str(n)
            if b_val is not None and ref is not None:
                (flagged_err if n else clean_err).append(abs(b_val - ref))

        ref_txt = f"{ref:.2f}" if ref is not None else "n/a"
        print(f"{rid:20s} {ref_txt:>6s} {s_txt:>8s} {b_txt:>9s} "
              f"{a_txt:>9s} {o_txt:>9s} {flags:>6s}")

    print("-" * len(head))
    print(f"validated: blind {counts['blind']}/10, agent {counts['agent']}/10, "
          f"old {counts['old']}/10")

    print("\nDo flagged structures go with larger errors? (blind runs)")
    if clean_err and flagged_err:
        mc = sum(clean_err) / len(clean_err)
        mf = sum(flagged_err) / len(flagged_err)
        print(f"  clean:   n={len(clean_err)}, mean |error| {mc:.3f} eV")
        print(f"  flagged: n={len(flagged_err)}, mean |error| {mf:.3f} eV")
        print("  With ten reactions this is suggestive at most. It is not a "
              "significance test.")
    else:
        print(f"  not enough data: {len(clean_err)} clean and "
              f"{len(flagged_err)} flagged runs with a barrier")


if __name__ == "__main__":
    main()
