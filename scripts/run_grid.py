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
import signal
import sys
import traceback
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.benchmark import SBH10, run_one
from src.graph import create_graph

# Write to the network volume, not the repo. The repo lives on the pod's
# local disk, which is wiped when the pod goes away - a full sweep of
# results was lost that way. /workspace persists across pods.
RESULTS_DIR = Path(os.environ.get(
    "GRID_RESULTS_DIR",
    "/workspace/agentic-surface-catalysis/results/grid"))


# A reaction that fails path_resolved makes the agent rerun the NEB with more
# images, and each rerun is a full NEB from scratch. There is no cap on that
# loop, so one pathological reaction can block every reaction behind it for
# hours. Abandon it after a fixed budget, log it, and move on: a logged
# failure is a data point, a fourth NEB is not.
REACTION_BUDGET_S = int(os.environ.get("REACTION_BUDGET_S", "2700"))


class ReactionTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise ReactionTimeout(f"per-reaction budget of {REACTION_BUDGET_S}s exceeded")


def result_path(reaction_id, model_key, d3_label):
    return RESULTS_DIR / f"{reaction_id}_{model_key}_d3{d3_label}.json"


def error_path(reaction_id, model_key, d3_label):
    return RESULTS_DIR / f"{reaction_id}_{model_key}_d3{d3_label}.error"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=config.DEFAULT_MODEL)
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
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    d3_labels = ["on", "off"] if args.d3 == "both" else [args.d3]

    unknown = [m for m in models if m not in config.MODELS]
    if unknown:
        raise SystemExit(f"Unknown model key(s): {unknown}. Known: {list(config.MODELS)}")

    only = None
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
        # Counts reactions that actually ran, not plan entries seen, so the
        # limit means the same thing on a fresh sweep and on a resumed one.
        if args.limit is not None and completed + failed >= args.limit:
            print(f"\n[limit] {args.limit} reaction(s) have run; stopping "
                  f"before {reaction_id} d3={d3_label}. "
                  f"Re-run to continue where this left off.")
            break

        rp = result_path(reaction_id, model_key, d3_label)
        ep = error_path(reaction_id, model_key, d3_label)

        if rp.exists():
            # A result file existing is not the same as a result being
            # good: run_one() can catch an exception internally (an
            # Anthropic API error mid-supervisor-turn is the case that
            # actually happened) and return computed_eV=None with a
            # run_error field instead of raising, so the outer try/except
            # below never sees it and a corrupted placeholder gets written
            # exactly like a real result. Re-check the content, not just
            # the file's existence, or a credit-exhaustion placeholder
            # sits there forever looking done to every future run.
            try:
                prior = json.loads(rp.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                print(f"[{i}/{len(plan)}] {reaction_id} d3={d3_label} -> "
                      f"existing result file unreadable ({exc}), re-running")
                prior = {"run_error": f"unreadable prior result: {exc}"}

            if not prior.get("run_error"):
                print(f"[{i}/{len(plan)}] {reaction_id} d3={d3_label} -> done, skipping")
                skipped += 1
                continue

            print(f"[{i}/{len(plan)}] {reaction_id} d3={d3_label} -> "
                  f"prior result carries run_error "
                  f"({str(prior['run_error'])[:80]}), re-running")

        print(f"[{i}/{len(plan)}] {model_key} {reaction_id} d3={d3_label} -> running "
              f"({datetime.now(timezone.utc).isoformat()})", flush=True)

        os.environ["FORCE_MODEL"] = model_key
        os.environ["FORCE_D3"] = d3_label

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(REACTION_BUDGET_S)

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
        finally:
            signal.alarm(0)

    print(f"\nDone. completed={completed} skipped={skipped} failed={failed} of {len(plan)}")


if __name__ == "__main__":
    main()
