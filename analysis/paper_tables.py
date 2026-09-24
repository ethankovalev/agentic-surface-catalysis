"""
Every table and number the paper quotes, regenerated from the result files.

    python analysis/paper_tables.py
    python analysis/paper_tables.py --results /workspace/agentic-surface-catalysis/results

Writes two files next to this script:

    paper_tables.md     the tables, ready to read or paste
    paper_numbers.json  every number, so the text can quote them exactly

No GPU, no API. Reads JSON only.

WHY THIS EXISTS
---------------
Numbers in this project moved repeatedly in one week as bugs were found
and fixed. A number typed into a paper from memory, or copied from a
terminal on the day it was produced, can quietly go stale. Every number
the paper states should come out of this script, and the output records
which input files it was computed from, when they were last written, and
the git commit of the code.

WHAT IT DOES NOT DO
-------------------
It does not decide what is true. It reports what the files say, with
the caveats each quantity needs printed beside it. In particular:

  - MAE over VALIDATED blind runs is over a selected subset and cannot be
    compared with an MAE over all ten reactions. It is labelled as such.
  - Seeded status counts reflect the checks as they were when each file
    was written. The file dates are printed so that can be judged.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.benchmark import SBH10  # noqa: E402

REACTIONS = list(SBH10)
D3_SETTINGS = ("off", "on")

# every input file read, with its hash and date, for the provenance table
INPUTS = []


# ---------------------------------------------------------------- helpers

def load(path):
    """Read a JSON file and record it as an input. None if missing."""
    path = Path(path)
    if not path.exists():
        INPUTS.append({"file": str(path), "found": False})
        return None
    raw = path.read_bytes()
    INPUTS.append({
        "file": str(path),
        "found": True,
        "sha256": hashlib.sha256(raw).hexdigest()[:16],
        "modified": datetime.datetime.fromtimestamp(
            path.stat().st_mtime, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"),
    })
    return json.loads(raw)


def reference(rid):
    return SBH10[rid].get("reference_eV")


def error_stats(errors):
    """n, MAE, RMSE, mean signed error, and how many fall below zero."""
    errors = [e for e in errors if e is not None]
    n = len(errors)
    if n == 0:
        return {"n": 0, "mae": None, "rmse": None, "mse": None, "below": 0}
    mae = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    mse = sum(errors) / n
    below = sum(1 for e in errors if e < 0)
    return {"n": n, "mae": mae, "rmse": rmse, "mse": mse, "below": below}


def f(x, digits=3, signed=False):
    """Format a number for a table cell."""
    if x is None:
        return "n/a"
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def table(headers, rows):
    """A markdown table from a header list and a list of row lists."""
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def domain(model):
    return "in" if config.MODELS[model].get("in_domain_for_surfaces") else "out"


def git_state():
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return commit or "unknown", bool(dirty)
    except Exception:
        return "unknown", None


# --------------------------------------------------------------- loading

def load_seeded(results):
    """{(model, d3): {reaction: record}} for every seeded file present."""
    seeded = {}
    for model in config.MODELS:
        for d3 in D3_SETTINGS:
            data = load(results / "seeded" / f"seeded_{model}_d3{d3}.json")
            if data:
                seeded[(model, d3)] = data
    return seeded


def load_per_reaction(results, folder):
    """{(model, d3): {reaction: record}} for one-file-per-reaction tracks."""
    found = {}
    for model in config.MODELS:
        for d3 in D3_SETTINGS:
            records = {}
            for rid in REACTIONS:
                path = results / folder / f"{rid}_{model}_d3{d3}.json"
                if path.exists():
                    records[rid] = load(path)
            if records:
                found[(model, d3)] = records
    return found


# ---------------------------------------------------------------- tables

def t_references():
    rows = [[rid, f(reference(rid), 2),
             f(SBH10[rid].get("reference_uncertainty_eV"), 2)] for rid in REACTIONS]
    numbers = {rid: {"reference_eV": reference(rid),
                     "uncertainty_eV": SBH10[rid].get("reference_uncertainty_eV")}
               for rid in REACTIONS}
    return ("SBH10 reference barriers (eV)",
            table(["Reaction", "Reference", "Uncertainty"], rows), numbers)


def t_seeded_accuracy(seeded):
    """Finding 1: single points at the published geometry, per model and D3."""
    rows, numbers = [], {}
    for (model, d3), data in sorted(seeded.items()):
        errors = []
        for rid in REACTIONS:
            value = (data.get(rid) or {}).get("barrier_at_reference_geometry_eV")
            ref = reference(rid)
            errors.append(None if value is None or ref is None else value - ref)
        s = error_stats(errors)
        numbers[f"{model}_d3{d3}"] = s
        rows.append([model, d3, domain(model), config.MODELS[model].get("d3_xc") or "none",
                     s["n"], f(s["mae"]), f(s["rmse"]), f(s["mse"], signed=True),
                     f"{s['below']}/{s['n']}"])
    note = ("Barrier at the published BEEF-vdW transition state, nothing moved. "
            "Error is computed minus reference. No search, so no check in this "
            "project's pipeline affects these numbers.")
    return ("Seeded accuracy at the published geometry",
            note + "\n\n" + table(["Model", "D3", "Domain", "Damping", "n", "MAE",
                                   "RMSE", "Mean signed", "Below ref"], rows), numbers)


def t_seeded_per_reaction(seeded):
    keys = sorted(seeded)
    headers = ["Reaction", "Ref"] + [f"{m} {d}" for m, d in keys]
    rows, numbers = [], {}
    for rid in REACTIONS:
        row = [rid, f(reference(rid), 2)]
        numbers[rid] = {}
        for model, d3 in keys:
            value = (seeded[(model, d3)].get(rid) or {}).get(
                "barrier_at_reference_geometry_eV")
            row.append(f(value))
            numbers[rid][f"{model}_d3{d3}"] = value
        rows.append(row)
    return ("Seeded barrier at the published geometry, every reaction (eV)",
            table(headers, rows), numbers)


def t_dispersion_shift(seeded):
    """Finding 1b: the D3 shift per reaction, and its spread within a damping family."""
    models = sorted({m for m, _ in seeded if (m, "on") in seeded and (m, "off") in seeded})
    rows, numbers = [], {}
    for rid in REACTIONS:
        shifts = {}
        for model in models:
            on = (seeded[(model, "on")].get(rid) or {}).get("barrier_at_reference_geometry_eV")
            off = (seeded[(model, "off")].get(rid) or {}).get("barrier_at_reference_geometry_eV")
            shifts[model] = None if on is None or off is None else on - off
        spreads = {}
        for damping in sorted({config.MODELS[m].get("d3_xc") for m in models} - {None}):
            family = [shifts[m] for m in models
                      if config.MODELS[m].get("d3_xc") == damping and shifts[m] is not None]
            spreads[damping] = (max(family) - min(family)) if len(family) >= 2 else None
        numbers[rid] = {"shift_eV": shifts, "spread_within_damping_eV": spreads}
        rows.append([rid] + [f(shifts[m], signed=True) for m in models]
                    + [f(spreads.get("pbe"), 4)])
    note = ("Barrier with D3 minus barrier without, at the published geometry. The last "
            "column is the largest difference between models sharing pbe damping; a "
            "spread near zero means the shift belongs to the damping, not the model.")
    return ("Dispersion shift per reaction (eV)",
            note + "\n\n" + table(["Reaction"] + models + ["pbe spread"], rows), numbers)


def t_d3_preference(seeded):
    rows, numbers = [], {}
    for model in sorted({m for m, _ in seeded}):
        if (model, "on") not in seeded or (model, "off") not in seeded:
            continue
        better_off = compared = 0
        for rid in REACTIONS:
            ref = reference(rid)
            on = (seeded[(model, "on")].get(rid) or {}).get("barrier_at_reference_geometry_eV")
            off = (seeded[(model, "off")].get(rid) or {}).get("barrier_at_reference_geometry_eV")
            if None in (ref, on, off):
                continue
            compared += 1
            better_off += abs(off - ref) < abs(on - ref)
        numbers[model] = {"better_without_d3": better_off, "compared": compared}
        rows.append([model, domain(model), f"{better_off}/{compared}"])
    return ("Reactions closer to the reference without D3",
            table(["Model", "Domain", "D3 off closer"], rows), numbers)


def t_domain(seeded):
    """Finding 2: in-domain against out-of-domain, error and saddle counts."""
    rows, numbers = [], {}
    for d3 in D3_SETTINGS:
        for side in ("in", "out"):
            errors = []
            for (model, setting), data in seeded.items():
                if setting != d3 or domain(model) != side:
                    continue
                for rid in REACTIONS:
                    value = (data.get(rid) or {}).get("barrier_at_reference_geometry_eV")
                    ref = reference(rid)
                    if value is not None and ref is not None:
                        errors.append(value - ref)
            s = error_stats(errors)
            numbers[f"{side}_domain_d3{d3}"] = s
            rows.append([f"{side} domain", d3, s["n"], f(s["mae"])])

    saddle_rows = []
    for (model, d3), data in sorted(seeded.items()):
        first_order = sum(1 for rid in REACTIONS
                          if (data.get(rid) or {}).get("first_order_saddle")
                          and (data.get(rid) or {}).get("saddle_converged"))
        ok = sum(1 for rid in REACTIONS if (data.get(rid) or {}).get("status") == "ok")
        numbers[f"saddles_{model}_d3{d3}"] = {"first_order": first_order, "status_ok": ok}
        saddle_rows.append([model, d3, domain(model), f"{first_order}/10", f"{ok}/10"])

    note = ("Pooled over every model on each side of the training domain split in "
            "config.MODELS. Saddle counts: a converged first order saddle after "
            "refinement from the published geometry, and a run whose status is ok "
            "(saddle, zero point correction and connectivity all present). Status "
            "counts reflect the checks as they were when each file was written; see "
            "the input dates below.")
    return ("Training domain",
            note + "\n\n" + table(["Group", "D3", "n", "MAE"], rows) + "\n\n"
            + table(["Model", "D3", "Domain", "First order saddle", "Status ok"],
                    saddle_rows), numbers)


def t_blind(blind):
    """The blind scripted track, per reaction, for every model and D3 present."""
    parts, numbers = [], {}
    for (model, d3), records in sorted(blind.items()):
        rows, validated_errors = [], []
        n_validated = 0
        for rid in REACTIONS:
            rec = records.get(rid)
            if rec is None:
                rows.append([rid, f(reference(rid), 2), "not run", "", "", "", ""])
                continue
            value, ref = rec.get("computed_eV"), reference(rid)
            error = None if value is None or ref is None else value - ref
            failed = [k for k, v in (rec.get("validation") or {}).items() if not v]
            if rec.get("validated"):
                n_validated += 1
                validated_errors.append(error)
            rows.append([rid, f(ref, 2), f(value), f(error, signed=True),
                         "yes" if rec.get("validated") else "no",
                         ", ".join(failed) or "",
                         rec.get("saddle_chosen_from") or ""])
        s = error_stats(validated_errors)
        numbers[f"{model}_d3{d3}"] = {"validated": n_validated, "of": len(REACTIONS),
                                     "validated_subset_stats": s}
        parts.append(
            f"**{model}, D3 {d3}: {n_validated}/{len(REACTIONS)} validated.** "
            f"MAE over the validated subset {f(s['mae'])} eV (n={s['n']}). This is a "
            f"selected subset and is not comparable with an MAE over all ten reactions.\n\n"
            + table(["Reaction", "Ref", "Blind", "Error", "Validated", "Failed checks",
                     "Saddle from"], rows))
    return ("Blind scripted track", "\n\n".join(parts) if parts else "No blind results found.",
            numbers)


def t_blind_failures(blind):
    counts, numbers = {}, {}
    for (model, d3), records in blind.items():
        for rid, rec in records.items():
            if rec.get("validated"):
                continue
            for check, passed in (rec.get("validation") or {}).items():
                if not passed:
                    counts[check] = counts.get(check, 0) + 1
    rows = [[check, n] for check, n in sorted(counts.items(), key=lambda kv: -kv[1])]
    numbers = dict(counts)
    return ("Why blind runs failed validation",
            table(["Check", "Failed in this many runs"], rows) if rows else "None failed.",
            numbers)


def t_cross_track(blind, agent_new, agent_old):
    """Agreement where the blind track and an agent run both produced a number."""
    rows, numbers = [], {}
    for (model, d3), records in sorted(blind.items()):
        for rid, rec in records.items():
            b = rec.get("computed_eV")
            for label, agent in (("agent (grid_v2)", agent_new), ("agent (grid)", agent_old)):
                a_rec = (agent.get((model, d3)) or {}).get(rid)
                if not a_rec or b is None or a_rec.get("computed_eV") is None:
                    continue
                a = a_rec["computed_eV"]
                numbers[f"{rid}_{model}_d3{d3}_{label}"] = abs(b - a)
                rows.append([rid, model, d3, label, f(b), f(a), f(abs(b - a))])
    note = ("Differences reflect both run to run variation and genuinely different "
            "saddles; a large difference is not by itself an error.")
    return ("Blind against agent, same reaction",
            note + "\n\n" + table(["Reaction", "Model", "D3", "Compared with", "Blind",
                                   "Agent", "Difference"], rows) if rows
            else "No reaction has both a blind and an agent result.", numbers)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path(os.environ.get(
        "RESULTS_BASE", "/workspace/agentic-surface-catalysis/results")))
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()

    if not args.results.exists():
        raise SystemExit(f"{args.results} not found. Run this on the pod, or pass "
                         "--results with the path to the results folder.")

    seeded = load_seeded(args.results)
    blind = load_per_reaction(args.results, "blind")
    agent_new = load_per_reaction(args.results, "grid_v2")
    agent_old = load_per_reaction(args.results, "grid")

    sections = [
        t_references(),
        t_seeded_accuracy(seeded),
        t_seeded_per_reaction(seeded),
        t_dispersion_shift(seeded),
        t_d3_preference(seeded),
        t_domain(seeded),
        t_blind(blind),
        t_blind_failures(blind),
        t_cross_track(blind, agent_new, agent_old),
    ]

    commit, dirty = git_state()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = (f"# Paper tables\n\nGenerated {stamp} by analysis/paper_tables.py at "
              f"commit {commit}{' (uncommitted changes present)' if dirty else ''}. "
              f"Do not edit by hand; rerun the script.\n")

    found = [i for i in INPUTS if i["found"]]
    inputs_table = table(["File", "Modified (UTC)", "sha256 prefix"],
                         [[Path(i["file"]).relative_to(args.results), i["modified"],
                           i["sha256"]] for i in found])

    md = [header]
    for title, body, _ in sections:
        md.append(f"## {title}\n\n{body}\n")
    md.append(f"## Inputs\n\n{len(found)} result files read.\n\n{inputs_table}\n")
    (args.out / "paper_tables.md").write_text("\n".join(md))

    numbers = {"generated": stamp, "commit": commit, "uncommitted_changes": dirty,
               "inputs": found}
    for title, _, nums in sections:
        numbers[title] = nums
    (args.out / "paper_numbers.json").write_text(
        json.dumps(numbers, indent=1, default=str))

    print("\n".join(md))
    print(f"\nWrote {args.out / 'paper_tables.md'} and {args.out / 'paper_numbers.json'}")


if __name__ == "__main__":
    main()
