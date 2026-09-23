"""
Run every SBH10 reaction blind, through the same tools the agent uses,
in a fixed order, with no language model.

    python -u scripts/run_blind.py --d3 off
    python -u scripts/run_blind.py --d3 off --only H2_Cu111,N2_Ru0001_terrace
    python -u scripts/run_blind.py --d3 off --force        # rerun finished ones

GPU only. No Anthropic API calls at all.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
Provenance is "blind_scripted". It is blind in the same sense as the
autonomous track: every structure is built from scratch, no published
geometry is read, and the reference barrier is never used by the
pipeline. It is NOT the autonomous track: no agent decides anything.
Results are written to their own directory and must never be pooled with
agent results or described as agentic.

Its job is to separate two questions the agent track answers together:

  - can the TOOLS find a validated barrier blind?        (this script)
  - does the AGENT call those tools in a way that does?  (run_grid.py)

Where this succeeds and the agent fails, the fault is orchestration, and
this script's step list is the policy the agent should have followed.
Where this also fails, the fault is in the tools or the model.

THE POLICY
----------
Exactly the sequence the simulation prompt describes, with the recovery
rule applied unconditionally rather than left to judgement: after the
band, refine_saddle_robust always runs. On 2026-09-22 the agent applied
that rule on N2/Ru(0001) and validated, and did not on H2/Cu(111) and
did not.

STRUCTURES ARE KEPT, AND AUDITED
--------------------------------
Every structure is copied out of work/ before the next reaction
overwrites it, and each is run through the structural sanity checks
(desorption, fused atoms, damaged slab). Energies are screened too:
a barrier above 3 eV or below -0.2 eV, or a reaction energy beyond 4 eV
in either direction, is flagged. These energy limits are heuristics
chosen to be far outside anything in SBH10, not derived bounds.
Unphysical structures and unreasonable energies are the larger MLIP
failure mode, and this makes them countable across a whole sweep rather
than visible only one run at a time.
"""

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from ase.io import read  # noqa: E402
from src import store  # noqa: E402
from src.benchmark import SBH10  # noqa: E402
from src.tools import (  # noqa: E402
    _structural_problems,
    build_dissociated_endpoint, build_gas_reference, build_slab,
    build_stepped_slab, check_convergence, check_dispersion_consistent,
    check_dispersion_relevance, check_endpoints_are_minima,
    check_endpoints_distinct, check_fragments_sensible, check_gas_reference_applied,
    check_geometry, check_noise_floor, check_path_resolved,
    check_reaction_consistency, check_saddle_connects,
    compute_gas_referenced_barrier, place_adsorbate, refine_saddle_robust,
    relax_structure, run_neb, validation_summary,
)
from src.zpe import check_zpe_applied, compute_zpe_correction  # noqa: E402

RESULTS = Path(os.environ.get(
    "BLIND_RESULTS_DIR",
    "/workspace/agentic-surface-catalysis/results/blind"))

STRUCTURES = ("initial", "final", "gasref", "peak", "saddle")

BARRIER_MAX_eV = 3.0
BARRIER_MIN_eV = -0.2
REACTION_ABS_MAX_eV = 4.0

VALIDATION = [
    check_convergence, check_noise_floor, check_dispersion_relevance,
    check_dispersion_consistent, check_gas_reference_applied,
    check_fragments_sensible, check_geometry, check_reaction_consistency,
    check_endpoints_distinct, check_path_resolved, check_zpe_applied,
]


def call(tool, steps, **kwargs):
    """Invoke one tool, record its output, and return that output."""
    name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))
    t0 = time.time()
    out = tool.invoke(kwargs)
    text = str(out)
    steps.append({"tool": name, "seconds": round(time.time() - t0, 1),
                  "output": text[:600]})
    print(f"  [{name}] {text.splitlines()[0][:110] if text else ''}",
          flush=True)
    return text


def failed(text):
    return text.startswith("FAILED") or text.startswith("NOT")


def _keep_peak(work, label, neb, peaks):
    """Copy the current NEB peak aside so a later band cannot overwrite it."""
    src = work / "peak.traj"
    if not src.exists():
        return
    dst = work / f"peak_{label}.traj"
    shutil.copy(src, dst)
    neb = neb or {}
    peaks.append({"label": label, "file": dst,
                  "band_barrier_eV": neb.get("barrier_eV"),
                  "converged": bool(neb.get("converged"))})


def run_one(rid, spec, model_key, with_d3, n_images):
    """The fixed policy for one reaction. Returns (steps, stopped_at)."""
    steps = []
    kw = {"model_key": model_key, "with_d3": with_d3}
    stepped = spec.get("site_type") == "step"

    if stepped:
        out = call(build_stepped_slab, steps,
                   metal=spec["metal"], facet=spec["facet"])
    else:
        args = {"metal": spec["metal"], "facet": spec["facet"]}
        if spec.get("cell"):
            args["nx"], args["ny"] = spec["cell"]
        out = call(build_slab, steps, **args)
    if failed(out):
        return steps, "build_slab"

    out = call(place_adsorbate, steps, species=spec["molecule"],
               site="step" if stepped else "ontop")
    if failed(out):
        return steps, "place_adsorbate"

    out = call(build_dissociated_endpoint, steps)
    if failed(out):
        return steps, "build_dissociated_endpoint"

    for structure in ("initial", "final"):
        out = call(relax_structure, steps, structure=structure, **kw)
        if failed(out):
            return steps, f"relax {structure}"

    call(build_gas_reference, steps)
    call(relax_structure, steps, structure="gasref", **kw)
    call(check_endpoints_are_minima, steps, **kw)

    out = call(run_neb, steps, n_images=n_images, **kw)
    if out.startswith("FAILED"):
        return steps, "run_neb"

    # Keep every band's peak. On 2026-09-23 N2/Ru(0001) terrace ran two
    # bands: the first peaked at 1.450 eV and was never refined, the second
    # converged at 1.739 eV and was. The agent had refined the first and
    # found a connected saddle at 1.411 eV. Two connected saddles for one
    # reaction, and the policy kept whichever came last.
    work = Path(config.WORK_DIR)
    peaks = []
    _keep_peak(work, "band1", store.get("neb"), peaks)

    # One finer band when the first did not converge. MAX_NEB_ATTEMPTS
    # allows two. It is not guaranteed to help - on N2/Ru(0001) two bands
    # were once both unconverged - but it is the one retry the tool permits.
    if not (store.get("neb") or {}).get("converged"):
        out = call(run_neb, steps, n_images=n_images + 6, **kw)
        if out.startswith("FAILED"):
            return steps, "run_neb (second band)"
        _keep_peak(work, "band2", store.get("neb"), peaks)

    # Refine every plausible peak; keep the LOWEST connected saddle. The
    # lowest connected saddle sets the rate. This rule never looks at the
    # reference, and it can move a result further from it: N2 terrace is
    # expected to go from 1.704 back toward 1.41, further from 1.84.
    tried = [pk for pk in peaks
             if pk["band_barrier_eV"] is None
             or pk["band_barrier_eV"] <= BARRIER_MAX_eV]
    if not tried and peaks:
        tried = peaks[-1:]
    for pk in peaks:
        if pk not in tried:
            steps.append({"tool": "policy", "output": (
                f"skipped {pk['label']} peak: band barrier "
                f"{pk['band_barrier_eV']:.2f} eV is above {BARRIER_MAX_eV}")})

    candidates = []
    for pk in tried:
        shutil.copy(pk["file"], work / "peak.traj")
        call(refine_saddle_robust, steps, **kw)
        saddle = store.get("saddle") or {}
        summary = {"band": pk["label"],
                   "band_barrier_eV": pk["band_barrier_eV"],
                   "first_order": bool(saddle.get("first_order_saddle")),
                   "connects": None, "barrier_eV": saddle.get("barrier_eV")}
        if saddle.get("first_order_saddle"):
            call(check_saddle_connects, steps, **kw)
            conn = store.get("saddle_connectivity") or {}
            summary["connects"] = conn.get("connects")
            if conn.get("connects") is True:
                kept = work / f"saddle_{pk['label']}.traj"
                shutil.copy(work / "saddle.traj", kept)
                candidates.append({"energy": saddle.get("energy_eV"),
                                   "saddle": dict(saddle),
                                   "connectivity": dict(conn),
                                   "file": kept, "band": pk["label"]})
        store.put("saddle_candidates",
                  (store.get("saddle_candidates") or []) + [summary])

    if candidates:
        best = min(candidates, key=lambda c: c["energy"])
        store.put("saddle", best["saddle"])
        store.put("saddle_connectivity", best["connectivity"])
        shutil.copy(best["file"], work / "saddle.traj")
        store.put("saddle_chosen_from", best["band"])

    saddle = store.get("saddle") or {}
    call(compute_gas_referenced_barrier, steps)
    if saddle.get("first_order_saddle"):
        call(compute_zpe_correction, steps, **kw)

    return steps, None


def audit(folder):
    """Structural sanity over every saved structure."""
    found = {}
    for name in STRUCTURES:
        f = folder / f"{name}.traj"
        if not f.exists():
            continue
        label = "saddle" if name == "peak" else name
        try:
            found[name] = _structural_problems(read(str(f)), label)
        except Exception as exc:
            found[name] = [f"audit could not run: {type(exc).__name__}: {exc}"]
    return found


def energy_flags(barrier, neb):
    flags = []
    if barrier is not None:
        if barrier > BARRIER_MAX_eV:
            flags.append(f"barrier {barrier:.2f} eV above {BARRIER_MAX_eV}")
        if barrier < BARRIER_MIN_eV:
            flags.append(f"barrier {barrier:.2f} eV below {BARRIER_MIN_eV}")
    rxn = (neb or {}).get("reaction_energy_eV")
    if rxn is not None and abs(rxn) > REACTION_ABS_MAX_eV:
        flags.append(f"reaction energy {rxn:.2f} eV beyond "
                     f"+/-{REACTION_ABS_MAX_eV}")
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="uma-s-1p1")
    ap.add_argument("--d3", choices=["on", "off", "both"], default="off")
    ap.add_argument("--only", default=None)
    ap.add_argument("--n-images", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    reactions = list(SBH10)
    if args.only:
        wanted = [r.strip() for r in args.only.split(",") if r.strip()]
        bad = [r for r in wanted if r not in SBH10]
        if bad:
            raise SystemExit(f"unknown reaction(s): {bad}")
        reactions = [r for r in reactions if r in wanted]
    d3s = {"on": [True], "off": [False], "both": [False, True]}[args.d3]

    RESULTS.mkdir(parents=True, exist_ok=True)
    work = Path(config.WORK_DIR)
    done = ran = errored = 0

    for with_d3 in d3s:
        label = "on" if with_d3 else "off"
        os.environ["FORCE_D3"] = label
        os.environ["FORCE_MODEL"] = args.model
        for rid in reactions:
            tag = f"{rid}_{args.model}_d3{label}"
            out_json = RESULTS / f"{tag}.json"
            if out_json.exists() and not args.force:
                prior = json.loads(out_json.read_text())
                if not prior.get("run_error"):
                    print(f"{tag}: done, skipping")
                    done += 1
                    continue

            print(f"\n{'=' * 72}\n{tag}\n{'=' * 72}", flush=True)
            spec = SBH10[rid]
            store.reset(rid)
            t0 = time.time()
            steps, stopped, run_error = [], None, None
            try:
                steps, stopped = run_one(rid, spec, args.model, with_d3,
                                         args.n_images)
            except Exception as exc:
                run_error = f"{type(exc).__name__}: {exc}"
                print(f"  RUN ERROR: {run_error}", flush=True)
                traceback.print_exc()

            # validate whatever exists, even after an early stop, so the
            # record says which checks failed rather than nothing
            for check in VALIDATION:
                try:
                    call(check, steps)
                except Exception as exc:
                    steps.append({"tool": getattr(check, "name", "check"),
                                  "output": f"check raised {exc}"})
            try:
                call(validation_summary, steps)
            except Exception:
                pass

            folder = RESULTS / tag
            folder.mkdir(exist_ok=True)
            for name in STRUCTURES:
                src = work / f"{name}.traj"
                if src.exists():
                    shutil.copy(src, folder / f"{name}.traj")

            barrier = store.get("barrier_eV")
            ref = spec.get("reference_eV")
            structural = audit(folder)
            n_structural = sum(len(v) for v in structural.values())
            e_flags = energy_flags(barrier, store.get("neb"))
            validated = bool(store.all_checks_passed())

            record = {
                "provenance": "blind_scripted",
                "reaction": rid, "model": args.model, "with_d3": with_d3,
                "computed_eV": barrier, "reference_eV": ref,
                "error_eV": (None if barrier is None or ref is None
                             else barrier - ref),
                "validated": validated,
                "validation": store.validation(),
                "validation_detail": store.get("validation_detail", {}),
                "stopped_at": stopped,
                "run_error": run_error,
                "saddle_candidates": store.get("saddle_candidates"),
                "saddle_chosen_from": store.get("saddle_chosen_from"),
                "structural_problems": structural,
                "n_structural_problems": n_structural,
                "energy_flags": e_flags,
                "seconds": round(time.time() - t0, 1),
                "steps": steps,
                "structures_dir": str(folder),
            }
            out_json.write_text(json.dumps(record, indent=1, default=str))

            ran += 1
            errored += bool(run_error)
            status = "VALIDATED" if validated else "not validated"
            print(f"\n  -> {status}, barrier={barrier}, "
                  f"structural problems={n_structural}, "
                  f"energy flags={len(e_flags)}, "
                  f"{record['seconds']:.0f}s", flush=True)
            if stopped:
                print(f"  -> stopped at: {stopped}")

    print(f"\nDone. ran={ran} skipped={done} run_errors={errored}")
    print(f"Results in {RESULTS}")


if __name__ == "__main__":
    main()
