"""
SBH10 benchmark harness.

Reference barriers from Sharada, Bligaard, Luntz, Kroes and Norskov,
J. Phys. Chem. C 2017, 121(36), 19807-19815. Ten dissociation barriers
on transition metal surfaces, derived from molecular beam scattering,
laser-assisted associative desorption and thermal experiments - six via
SRP-DFT fitted to experiment, four from more ad-hoc procedures.

Two things about this benchmark matter.

The references are experimental, not computational. Agreement means
agreement with reality, not with another calculation.

The best-performing functional in the original study was BEEF-vdW, at
0.14 eV mean error - a dispersion-corrected GGA, beating both a
meta-GGA and a screened hybrid. That is the reverse of the gas-phase
pattern, and it is why the dispersion treatment is the axis this
benchmark discriminates on.

THE AGENT NEVER SEES THESE NUMBERS. There is no tool that reaches this
module. Comparison happens here, after the graph has returned. Any other
arrangement measures curve-fitting, not calculation.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src import store


# Fill reference_eV from Table 1 of the paper. Keep the keys stable;
# results are recorded against them.
SBH10 = {
    "H2_Cu111": {
        "molecule": "H2", "metal": "Cu", "facet": "111",
        "site_type": "terrace",
        "reaction_class": "dissociation",
        "reference_eV": 0.63,
    },
    "H2_Cu100": {
        "molecule": "H2", "metal": "Cu", "facet": "100",
        "site_type": "terrace",
        "reaction_class": "dissociation",
        "reference_eV": 0.74,
    },
    "H2_Pt111": {
        "molecule": "H2", "metal": "Pt", "facet": "111",
        "site_type": "terrace",
        "reaction_class": "dissociation",
        "reference_eV": 0.00,
    },
    "H2_Ru0001": {
        "molecule": "H2", "metal": "Ru", "facet": "0001",
        "site_type": "terrace",
        "reaction_class": "dissociation",
        "reference_eV": 0.00,
    },
    "N2_Ru0001_terrace": {
        "molecule": "N2", "metal": "Ru", "facet": "0001",
        "site_type": "terrace",
        "reaction_class": "dissociation",
        "reference_eV": 1.84,
    },
    "N2_Ru0001_step": {
        "molecule": "N2", "metal": "Ru", "facet": "0001",
        "site_type": "step",
        "reaction_class": "dissociation",
        "reference_eV": 0.40,
    },
    "CH4_Ru0001": {
        "molecule": "CH4", "metal": "Ru", "facet": "0001",
        "site_type": "terrace",
        "reaction_class": "dissociation",
        "reference_eV": 0.80,
    },
    "CH4_Ni100": {
        "molecule": "CH4", "metal": "Ni", "facet": "100",
        "site_type": "terrace",
        "reaction_class": "dissociation",
        "reference_eV": 0.76,
    },
    "CH4_Ni111_terrace": {
        "molecule": "CH4", "metal": "Ni", "facet": "111",
        "site_type": "terrace",
        "reaction_class": "dissociation",
        "reference_eV": 1.01,
    },
    "CH4_Ni111_step": {
        "molecule": "CH4", "metal": "Ni", "facet": "111",
        "site_type": "step",
        "reaction_class": "dissociation",
        "reference_eV": 0.80,
    },
}

def run_one(graph, reaction_id: str, spec: dict) -> dict:
    """Run the agent on a single reaction, blind, then score it."""
    store.reset(reaction_id)

    task = (
        f"Compute the dissociation barrier for {spec['molecule']} on "
        f"{spec['metal']}({spec['facet']}). Build the slab and both "
        f"endpoints, relax them, run the nudged elastic band, and "
        f"validate the result before reporting."
    )

    computed, error_note = None, None
    try:
        graph.invoke(
            {
                "messages": [("user", task)],
                "next": "",
                "reaction_id": reaction_id,
                "attempts": 0,
            },
            {
                "configurable": {"thread_id": reaction_id},
                "recursion_limit": config.RECURSION_LIMIT,
            },
        )
        computed = store.get("barrier_eV")
    except Exception as exc:
        error_note = f"{type(exc).__name__}: {exc}"
        print(f"  run failed: {error_note}")

    checks = store.validation()
    ref = spec.get("reference_eV")
    error = None if (computed is None or ref is None) else computed - ref

    return {
        "computed_eV": computed,
        "reference_eV": ref,
        "error_eV": error,
        "reaction_class": spec.get("reaction_class"),
        "validation": checks,
        "validation_detail": store.get("validation_detail", {}),
        "validated": store.all_checks_passed(),
        "run_error": error_note,
        "trace": store.snapshot(),
    }


def run_benchmark(graph, subset=None, out_name="sbh10_results.json"):
    """Run every reaction and write a scored results file."""
    reactions = subset or list(SBH10)
    results = {}

    for i, rid in enumerate(reactions, 1):
        print(f"\n{'=' * 60}\n[{i}/{len(reactions)}] {rid}\n{'=' * 60}")
        results[rid] = run_one(graph, rid, SBH10[rid])
        r = results[rid]
        print(f"  computed={r['computed_eV']} ref={r['reference_eV']} "
              f"error={r['error_eV']} validated={r['validated']}")

    out_path = config.OUTPUT_DIR / out_name
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwritten to {out_path}")

    summarise(results)
    return results


def summarise(results: dict):
    """Headline numbers.

    Validated and unvalidated results are reported separately on
    purpose. Averaging them together hides the more interesting half:
    how often the agent produced a number it should not have trusted.
    """
    print("\n" + "=" * 60)

    produced = [r for r in results.values() if r["computed_eV"] is not None]
    scored = [r for r in produced if r["error_eV"] is not None]
    validated = [r for r in scored if r["validated"]]

    print(f"reactions attempted : {len(results)}")
    print(f"barriers produced   : {len(produced)}")
    print(f"scored vs reference : {len(scored)}")
    print(f"passed validation   : {len(validated)}")

    if scored:
        mae = sum(abs(r["error_eV"]) for r in scored) / len(scored)
        print(f"MAE (all scored)    : {mae:.3f} eV")
    if validated:
        mae_v = sum(abs(r["error_eV"]) for r in validated) / len(validated)
        print(f"MAE (validated)     : {mae_v:.3f} eV")

    print("\nreference: BEEF-vdW reached 0.14 eV MAE on this set")
    print("(Sharada et al., J. Phys. Chem. C 2017, 121, 19807)")

    failed_checks = {}
    for r in results.values():
        for name, ok in r["validation"].items():
            if not ok:
                failed_checks[name] = failed_checks.get(name, 0) + 1
    if failed_checks:
        print("\nmost common check failures:")
        for name, n in sorted(failed_checks.items(), key=lambda x: -x[1]):
            print(f"  {name}: {n}")
    print("=" * 60)
