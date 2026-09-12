"""
Write the seeded track's headline table to a durable markdown file.

    python scripts/save_seeded_summary.py

Reads results/seeded/seeded_<model>_d3<on|off>.json for both D3 settings
and writes results/seeded/SEEDED_SUMMARY.md - the single most important
number in this project as of 2026-09-12: UMA's barrier at the PUBLISHED
SBH10 transition-state geometry, D3 on vs off, all 10 reactions, zero
missing data.

This is provenance="seeded": PES accuracy at a known saddle, not
transition-state search capability. Never quote this MAE as evidence the
autonomous pipeline works - it is evidence the underlying potential energy
surface is accurate. See run_seeded.py's module docstring.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.benchmark import SBH10

RESULTS = Path("/workspace/agentic-surface-catalysis/results/seeded")

def main():
    off = json.loads((RESULTS / "seeded_uma-s-1p1_d3off.json").read_text())
    on = json.loads((RESULTS / "seeded_uma-s-1p1_d3on.json").read_text())

    rows, off_errs, on_errs = [], [], []
    for name in sorted(off):
        ref = SBH10.get(name, {}).get("reference_eV")
        a = off[name].get("barrier_at_reference_geometry_eV")
        b = on[name].get("barrier_at_reference_geometry_eV")
        ea = (a - ref) if isinstance(a, (int, float)) and ref is not None else None
        eb = (b - ref) if isinstance(b, (int, float)) and ref is not None else None
        if ea is not None:
            off_errs.append(abs(ea))
        if eb is not None:
            on_errs.append(abs(eb))
        rows.append((name, ref, a, ea, b, eb))

    mae_off = sum(off_errs) / len(off_errs) if off_errs else None
    mae_on = sum(on_errs) / len(on_errs) if on_errs else None

    lines = [
        "# Seeded track: UMA barrier at the published SBH10 transition state",
        "",
        "**provenance: seeded.** PES accuracy at a known saddle, computed with",
        "zero geometry search - the literature BEEF-vdW transition state is",
        "read directly and evaluated. This measures whether the potential",
        "energy surface is right, not whether the engine can find a",
        "transition state on its own. See scripts/run_seeded.py.",
        "",
        f"| reaction | ref (eV) | D3-off | err | D3-on | err |",
        f"|---|---:|---:|---:|---:|---:|",
    ]
    for name, ref, a, ea, b, eb in rows:
        fr = f"{ref:.3f}" if ref is not None else "--"
        fa = f"{a:.3f}" if isinstance(a, (int, float)) else "--"
        fea = f"{ea:+.3f}" if ea is not None else "--"
        fb = f"{b:.3f}" if isinstance(b, (int, float)) else "--"
        feb = f"{eb:+.3f}" if eb is not None else "--"
        lines.append(f"| {name} | {fr} | {fa} | {fea} | {fb} | {feb} |")

    lines += [
        "",
        f"**MAE, D3-off: {mae_off:.3f} eV** ({len(off_errs)}/10 reactions)",
        f"**MAE, D3-on: {mae_on:.3f} eV** ({len(on_errs)}/10 reactions)",
        "",
        "For reference, BEEF-vdW (the best DFT functional in the original",
        "SBH10 paper) reports MAE ~0.12-0.14 eV against the same references.",
    ]

    out = RESULTS / "SEEDED_SUMMARY.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten to {out}")

if __name__ == "__main__":
    main()
