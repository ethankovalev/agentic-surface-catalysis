"""
Chart the seeded-track headline result. Runs entirely on the Mac - no
pod, no calculator, no venv-uma. Reads results_seeded_summary.md, which
is already committed to the repo.

    python plot_seeded_results.py

Writes seeded_d3_comparison.png to the repo root.

If matplotlib is missing:  pip3 install matplotlib
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_summary_table(path: Path):
    """Pull the reaction/ref/D3-off/D3-on rows out of the markdown table."""
    text = path.read_text()
    rows = []
    for line in text.splitlines():
        if not line.startswith("| ") or line.startswith("|---") or "reaction" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 6:
            continue
        name, ref, off, _, on, _ = cells
        try:
            ref_v = float(ref)
        except ValueError:
            continue
        off_v = float(off) if off != "--" else None
        on_v = float(on) if on != "--" else None
        rows.append((name, ref_v, off_v, on_v))
    if not rows:
        raise ValueError(
            f"no data rows parsed from {path}. Table format may have changed - "
            "check the header still reads '| reaction | ref (eV) | D3-off | "
            "err | D3-on | err |'.")
    return rows


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results_seeded_summary.md")
    if not src.exists():
        raise SystemExit(
            f"{src} not found. Run this from the repo root, or pass the path "
            "explicitly: python plot_seeded_results.py path/to/summary.md")

    rows = parse_summary_table(src)
    names = [r[0] for r in rows]
    ref = [r[1] for r in rows]
    off = [r[2] for r in rows]
    on = [r[3] for r in rows]

    mae_off = np.mean([abs(o - r) for o, r in zip(off, ref) if o is not None])
    mae_on = np.mean([abs(o - r) for o, r in zip(on, ref) if o is not None])

    x = np.arange(len(names))
    width = 0.28

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width, ref, width, label="SBH10 reference", color="#444444")
    ax.bar(x, [v if v is not None else 0 for v in off], width,
          label=f"UMA, D3 off (MAE {mae_off:.3f} eV)", color="#2a7f62")
    ax.bar(x + width, [v if v is not None else 0 for v in on], width,
          label=f"UMA, D3 on (MAE {mae_on:.3f} eV)", color="#b3541e")

    ax.set_ylabel("Barrier at published transition state (eV)")
    ax.set_title("UMA barrier accuracy at literature SBH10 transition states\n"
                 "(seeded track: zero geometry search, provenance=seeded)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out = Path("seeded_d3_comparison.png")
    fig.savefig(out, dpi=160)
    print(f"MAE D3-off: {mae_off:.3f} eV")
    print(f"MAE D3-on:  {mae_on:.3f} eV")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
