"""Resumable grid runner: reactions x models x D3 setting.

Model and dispersion are pinned per run via FORCE_MODEL / FORCE_D3, which
override whatever the agent chooses. Results are written per run, so the
script can be killed and restarted without losing completed work.

Usage:
    python scripts/run_grid.py --models uma-s-1p1 --dry-run
    python scripts/run_grid.py --models uma-s-1p1 --d3 on
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.benchmark import SBH10, run_one
from src.graph import create_graph

RESULTS_DIR = Path(__file__).parent.parent / "results" / "grid"


def result_path(reaction_id, model_key, d3_label):
    return RESULTS_DIR / f"{reaction_id}_{model_key}_d3{d3_label}.json"


def error_path(reaction_id, model_key, d3_label):
    return RESULTS_DIR / f"{reaction_id}_{model_key}_d3{d3_label}.error"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=config.DEFAULT_MODEL)
    parser.add_argument("--d3", choices=["on", "off", "both"], default="on")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    d3_labels = ["on", "off"] if args.d3 == "both" else [args.d3]

    unknown = [m for m in models if m not in config.MODELS]
    if unknown:
        raise SystemExit(f"Unknown model key(s): {unknown}. Known: {list(config.MODELS)}")

    plan = []
    for model_key in models:
        for reaction_id, spec in SBH10.items():
            for d3_label in d3_labels:
                plan.append((model_key, reaction_id, spec, d3_label))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Models: {models}  D3: {d3_labels}  Planned runs: {len(plan)}")

    if args.dry_run:
        for model_key, reaction_id, spec, d3_label in plan:
            done = result_path(reaction_id, model_key, d3_label).exists()
            print(f"  {model_key:14s} {reaction_id:24s} d3={d3_label:3s} -> "
                  f"{'SKIP (done)' if done else 'would run'}")
        return

    # Initialise torch.det's lazy LAPACK backend on the MAIN thread, before
    # any agent work starts. LangGraph runs tools in a thread pool and this
    # lazy init is not thread-safe - UMA calls torch.det on the cell matrix
    # during forward, which raises "lazy wrapper should be called at most
    # once" if it first happens inside a worker thread. invoke.py does the
    # same thing; this script is a second entry point and needs it too.
    import torch
    if torch.cuda.is_available():
        torch.det(torch.eye(3, device="cuda"))
    torch.det(torch.eye(3))

    graph = create_graph()
    completed = skipped = failed = 0

    for i, (model_key, reaction_id, spec, d3_label) in enumerate(plan, 1):
        rp = result_path(reaction_id, model_key, d3_label)
        ep = error_path(reaction_id, model_key, d3_label)

        if rp.exists():
            print(f"[{i}/{len(plan)}] {reaction_id} d3={d3_label} -> done, skipping")
            skipped += 1
            continue

        print(f"[{i}/{len(plan)}] {model_key} {reaction_id} d3={d3_label} -> running "
              f"({datetime.now(timezone.utc).isoformat()})", flush=True)

        os.environ["FORCE_MODEL"] = model_key
        os.environ["FORCE_D3"] = d3_label

        try:
            result = run_one(graph, reaction_id, spec)
            result["_grid_model_key"] = model_key
            result["_grid_d3_label"] = d3_label
            result["_grid_timestamp"] = datetime.now(timezone.utc).isoformat()
            with open(rp, "w") as f:
                json.dump(result, f, indent=2, default=str)
            if ep.exists():
                ep.unlink()
            status = "validated" if result.get("validated") else "ran, NOT validated"
            print(f"    -> {status}, computed_eV={result.get('computed_eV')}", flush=True)
            completed += 1
        except Exception as exc:
            with open(ep, "w") as f:
                f.write(f"{datetime.now(timezone.utc).isoformat()}\n{traceback.format_exc()}")
            print(f"    -> FAILED: {type(exc).__name__}: {exc}", flush=True)
            failed += 1
            continue

    print(f"\nDone. completed={completed} skipped={skipped} failed={failed} of {len(plan)}")


if __name__ == "__main__":
    main()
