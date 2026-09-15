"""
Chart the seeded track: four foundation MLIPs, dispersion on and off,
against the SBH10 reference barriers.

Runs anywhere. No pod, no calculator, no GPU. Reads the eight seeded
result files written by scripts/run_seeded.py.

    python plot_seeded_results.py
    python plot_seeded_results.py --results /path/to/results/seeded

Writes seeded_d3_comparison.png next to this script.

The quantity plotted is barrier_at_reference_geometry_eV, a single point
at the published BEEF vdW transition state with nothing moved. It exists
for every reaction whether or not the saddle search, connectivity check
or Hessian succeeded afterwards, which is why there are no gaps.
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

# model key, short name, in domain for surfaces, D3 damping functional
MODELS = [
    ("uma-s-1p1", "UMA S 1.1", True, "rpbe", "#14543c", "#7fbfa6"),
    ("mace-mh-1", "MACE mh 1", True, "pbe", "#8c3a12", "#e0a887"),
    ("orb-v3-cons-inf-omat", "Orb v3", False, "pbe", "#2f4b7c", "#9db4d4"),
    ("mace-mpa-0", "MACE mpa 0", False, "pbe", "#5c3566", "#bda2c4"),
]

KEY = "barrier_at_reference_geometry_eV"


def load(results_dir, model, d3):
    path = Path(results_dir) / f"seeded_{model}_d3{d3}.json"
    if not path.exists():
        raise SystemExit(f"{path} not found. Use --results, or run that sweep first.")
    raw = json.loads(path.read_text())
    out = {}
    for name in ORDER:
        v = (raw.get(name) or {}).get(KEY)
        if isinstance(v, (int, float)):
            out[name] = float(v)
    missing = [k for k in ORDER if k not in out]
    if missing:
        print(f"  note: {path.name} missing {KEY} for {missing}")
    return out


def mae(values):
    shared = [k for k in values if k in REFERENCE]
    return sum(abs(values[k] - REFERENCE[k]) for k in shared) / len(shared)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results",
                    default="/workspace/agentic-surface-catalysis/results/seeded")
    ap.add_argument("--out", default="seeded_d3_comparison.png")
    args = ap.parse_args()

    data, maes = {}, {}
    for key, short, indomain, xc, dark, light in MODELS:
        for d3 in ("off", "on"):
            vals = load(args.results, key, d3)
            data[(key, d3)] = vals
            maes[(key, d3)] = mae(vals)

    x = np.arange(len(ORDER))
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.25], hspace=0.42,
                          wspace=0.22)

    # Panel A: mean absolute error, the headline
    axa = fig.add_subplot(gs[0, 0])
    names, vals, colours = [], [], []
    for key, short, indomain, xc, dark, light in MODELS:
        for d3, colour in (("off", dark), ("on", light)):
            names.append(f"{short}\nD3 {d3}")
            vals.append(maes[(key, d3)])
            colours.append(colour)
    bars = axa.bar(range(len(vals)), vals, color=colours)
    axa.axhline(0.14, color="black", linestyle=":", linewidth=1.4)
    axa.text(len(vals) - 0.4, 0.155, "BEEF vdW 0.14", fontsize=8,
             ha="right", va="bottom")
    for b, v in zip(bars, vals):
        axa.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                 ha="center", fontsize=7.5)
    axa.set_xticks(range(len(names)))
    axa.set_xticklabels(names, fontsize=6.5)
    axa.set_ylabel("MAE against reference (eV)")
    axa.set_title("A. Accuracy by model and dispersion setting", fontsize=10,
                  loc="left")
    axa.grid(axis="y", alpha=0.25)
    axa.set_ylim(0, max(vals) * 1.18)

    # Panel B: the D3 shift is a property of the damping, not the model
    axb = fig.add_subplot(gs[0, 1])
    # The three pbe damped models coincide to within 1 meV, so plotted at
    # equal weight only one would be visible. Decreasing linewidth and
    # different markers make the overlap legible rather than hidden.
    weights = {"uma-s-1p1": (3.0, "o", 7), "mace-mh-1": (5.0, "s", 11),
               "orb-v3-cons-inf-omat": (2.6, "^", 7),
               "mace-mpa-0": (1.2, "x", 6)}
    for key, short, indomain, xc, dark, light in MODELS:
        shift = [data[(key, "on")].get(k, np.nan) - data[(key, "off")].get(k, np.nan)
                 for k in ORDER]
        lw, marker, ms = weights[key]
        axb.plot(x, shift, marker=marker, color=dark, markersize=ms,
                 linewidth=lw, linestyle="-" if xc == "rpbe" else "--",
                 alpha=1.0 if xc == "rpbe" else 0.85,
                 label=f"{short} ({xc})")
    axb.set_ylabel("Shift from turning D3 on (eV)")
    axb.set_title("B. The dispersion shift tracks the damping parameters,\n"
                  "not the model: three pbe damped models coincide to 1 meV",
                  fontsize=10, loc="left")
    axb.axhline(0, color="black", linewidth=0.8)
    axb.legend(fontsize=8)
    axb.grid(alpha=0.25)
    axb.set_xticks(x)
    axb.set_xticklabels([LABEL[k].replace("\n", " ") for k in ORDER],
                        rotation=40, ha="right", fontsize=7)

    # Panel C: in domain against out of domain, dispersion off
    axc = fig.add_subplot(gs[1, :])
    w = 0.16
    axc.bar(x - 2 * w, [REFERENCE[k] for k in ORDER], w,
            label="SBH10 reference", color="#333333")
    for i, (key, short, indomain, xc, dark, light) in enumerate(MODELS):
        vals = data[(key, "off")]
        tag = "in domain" if indomain else "out of domain"
        axc.bar(x + (i - 1) * w, [vals.get(k, 0.0) for k in ORDER], w,
                color=dark,
                label=f"{short}, {tag}  (MAE {maes[(key,'off')]:.3f})")
    axc.set_ylabel("Barrier at published TS (eV)")
    axc.set_title("C. Dispersion off: training domain separates the models "
                  "far more than architecture does", fontsize=10, loc="left")
    axc.axhline(0, color="black", linewidth=0.8)
    axc.legend(ncol=3, fontsize=8)
    axc.grid(axis="y", alpha=0.25)
    axc.set_xticks(x)
    axc.set_xticklabels([LABEL[k] for k in ORDER], fontsize=7.5)

    # Panel D: signed error, every series
    axd = fig.add_subplot(gs[2, :])
    w2 = 0.105
    for i, (key, short, indomain, xc, dark, light) in enumerate(MODELS):
        for j, (d3, colour) in enumerate((("off", dark), ("on", light))):
            vals = data[(key, d3)]
            pos = x + (2 * i + j - 3.5) * w2
            axd.bar(pos, [vals.get(k, 0.0) - REFERENCE[k] for k in ORDER], w2,
                    color=colour,
                    label=f"{short} D3 {d3}" if True else None)
    axd.axhline(0, color="black", linewidth=1.0)
    axd.set_ylabel("Error against reference (eV)")
    axd.set_title("D. Signed error, all eight series. Every bar below the line "
                  "with dispersion on, for all four models", fontsize=10,
                  loc="left")
    axd.legend(ncol=4, fontsize=7.5)
    axd.grid(axis="y", alpha=0.25)
    axd.set_xticks(x)
    axd.set_xticklabels([LABEL[k] for k in ORDER], fontsize=7.5)

    fig.suptitle("Foundation MLIP accuracy at literature SBH10 transition "
                 "states\nseeded track: published BEEF vdW geometry, no "
                 "search, 10 of 10 reactions in every series",
                 fontsize=12.5, y=0.985)
    fig.savefig(args.out, dpi=165, bbox_inches="tight")

    print()
    for key, short, indomain, xc, dark, light in MODELS:
        tag = "in domain " if indomain else "out domain"
        print(f"{short:12s} {tag}  D3 off {maes[(key,'off')]:.3f}   "
              f"D3 on {maes[(key,'on')]:.3f}")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
