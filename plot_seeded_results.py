"""
Chart the seeded track: UMA and MACE, dispersion on and off, against the
SBH10 reference barriers.

Runs anywhere, no pod, no calculator, no GPU. Reads the four seeded
result files written by scripts/run_seeded.py.

    python plot_seeded_results.py
    python plot_seeded_results.py --results /path/to/results/seeded

Writes seeded_d3_comparison.png next to this script.

The quantity plotted is barrier_at_reference_geometry_eV: a single point
at the published BEEF vdW transition state with nothing moved. It exists
for every reaction whether or not the saddle search, connectivity check
or Hessian succeeded, which is why the chart has no gaps.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REFERENCE = {
    "H2_Cu111": 0.630, "H2_Cu100": 0.740, "H2_Pt111": 0.000,
    "H2_Ru0001": 0.000, "N2_Ru0001_terrace": 1.840, "N2_Ru0001_step": 0.400,
    "CH4_Ru0001": 0.800, "CH4_Ni100": 0.760, "CH4_Ni111_terrace": 1.010,
    "CH4_Ni111_step": 0.800,
}

ORDER = ["H2_Cu111", "H2_Cu100", "H2_Pt111", "H2_Ru0001",
         "N2_Ru0001_terrace", "N2_Ru0001_step",
         "CH4_Ru0001", "CH4_Ni100", "CH4_Ni111_terrace", "CH4_Ni111_step"]

LABEL = {
    "H2_Cu111": "H$_2$/Cu(111)", "H2_Cu100": "H$_2$/Cu(100)",
    "H2_Pt111": "H$_2$/Pt(111)", "H2_Ru0001": "H$_2$/Ru(0001)",
    "N2_Ru0001_terrace": "N$_2$/Ru(0001)\nterrace",
    "N2_Ru0001_step": "N$_2$/Ru(0001)\nstep",
    "CH4_Ru0001": "CH$_4$/Ru(0001)", "CH4_Ni100": "CH$_4$/Ni(100)",
    "CH4_Ni111_terrace": "CH$_4$/Ni(111)\nterrace",
    "CH4_Ni111_step": "CH$_4$/Ni(111)\nstep",
}

SERIES = [
    ("uma-s-1p1", "off", "UMA, D3 off", "#1b6b4f"),
    ("uma-s-1p1", "on", "UMA, D3 on", "#7fbfa6"),
    ("mace-mh-1", "off", "MACE, D3 off", "#8c3a12"),
    ("mace-mh-1", "on", "MACE, D3 on", "#d9926a"),
]

KEY = "barrier_at_reference_geometry_eV"


def load(results_dir, model, d3):
    path = Path(results_dir) / f"seeded_{model}_d3{d3}.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Pass the right directory with --results, or "
            "run scripts/run_seeded.py for this model and dispersion setting "
            "first.")
    raw = json.loads(path.read_text())
    values, missing = {}, []
    for name in ORDER:
        entry = raw.get(name) or {}
        v = entry.get(KEY)
        if isinstance(v, (int, float)):
            values[name] = float(v)
        else:
            missing.append(name)
    if missing:
        print(f"  note: {path.name} has no {KEY} for {missing}")
    return values


def mae(values):
    shared = [k for k in values if k in REFERENCE]
    return sum(abs(values[k] - REFERENCE[k]) for k in shared) / len(shared)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/workspace/agentic-surface-catalysis/results/seeded")
    ap.add_argument("--out", default="seeded_d3_comparison.png")
    args = ap.parse_args()

    data, maes = {}, {}
    for model, d3, label, _ in SERIES:
        vals = load(args.results, model, d3)
        data[(model, d3)] = vals
        maes[(model, d3)] = mae(vals)

    x = np.arange(len(ORDER))
    n = len(SERIES) + 1
    width = 0.85 / n

    fig, (ax, axe) = plt.subplots(
        2, 1, figsize=(14, 9), height_ratios=[2.1, 1], sharex=True)

    ax.bar(x - 2 * width, [REFERENCE[k] for k in ORDER], width,
           label="SBH10 reference", color="#333333")
    for i, (model, d3, label, colour) in enumerate(SERIES):
        vals = data[(model, d3)]
        ax.bar(x + (i - 1) * width, [vals.get(k, 0.0) for k in ORDER], width,
               label=f"{label}  (MAE {maes[(model, d3)]:.3f} eV)", color=colour)

    ax.set_ylabel("Barrier at the published transition state (eV)")
    ax.set_title(
        "Foundation MLIP accuracy at literature SBH10 transition states\n"
        "seeded track: published BEEF vdW geometry, no search, "
        "10 of 10 reactions for every series")
    ax.legend(ncol=2, fontsize=9, loc="upper left")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.grid(axis="y", alpha=0.25)

    # Same x offsets as the top panel so a column lines up with the bar
    # it is the error of.
    for i, (model, d3, label, colour) in enumerate(SERIES):
        vals = data[(model, d3)]
        axe.bar(x + (i - 1) * width,
                [vals.get(k, 0.0) - REFERENCE[k] for k in ORDER],
                width, color=colour)
    axe.axhline(0, color="black", linewidth=0.9)
    axe.set_ylabel("Error vs reference (eV)")
    axe.grid(axis="y", alpha=0.25)
    axe.set_xticks(x)
    axe.set_xticklabels([LABEL[k] for k in ORDER], fontsize=8.5)

    fig.tight_layout()
    fig.savefig(args.out, dpi=170)

    print()
    for model, d3, label, _ in SERIES:
        print(f"{label:16s} MAE {maes[(model, d3)]:.3f} eV")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
