"""
Day-1 wiring. Run from the repo root:  python wire_day1.py

Six edits across five files, each one anchored on an exact string. If any
anchor is missing the script writes nothing at all and tells you which one,
rather than leaving the repo half-edited. Idempotent: running it twice is a
no-op and says so.

  config.py      REQUIRED_CHECKS; noise floor split into two honest numbers
  src/tools.py   import zpe, register its two tools, use REQUIRED_CHECKS
  src/graph.py   exit_gate enforces the required set, not just "what ran"
  src/prompt.py  step 7 in the simulation workflow
  src/benchmark.py  per-reaction experimental uncertainty
"""

import sys
from pathlib import Path

EDITS = []


def edit(path, old, new, why):
    EDITS.append((Path(path), old, new, why))


# --- config.py --------------------------------------------------------

edit("config.py", '''FMAX = 0.02              # eV/A for endpoint relaxations''',
     '''# Every check the run must pass before it can finish. validation_summary
# reports against this and exit_gate enforces it, so the two can never
# disagree about what "validated" means. A check that exists but is not
# listed here is optional, and an optional check is one the agent can skip
# on the run where it would have failed.
REQUIRED_CHECKS = (
    "convergence",
    "noise_floor",
    "dispersion",
    "dispersion_consistent",
    "gas_reference",
    "fragments",
    "geometry",
    "reaction_consistency",
    "endpoints_distinct",
    "path_resolved",
    "zpe",
)

FMAX = 0.02              # eV/A for endpoint relaxations''',
     "REQUIRED_CHECKS, shared by the summary and the gate")

edit("config.py", '''        "in_domain_for_surfaces": True,
        "noise_floor_eV": 0.3,
        "gated": True,
    },
    "mace-mh-1": {''',
     '''        "in_domain_for_surfaces": True,
        "noise_floor_eV": 0.3,
        "model_resolution_eV": 0.05,
        "reproducibility_eV": None,
        "gated": True,
    },
    "mace-mh-1": {''',
     "split UMA's noise floor")

edit("config.py", '''DEFAULT_MODEL = os.environ.get("MLIP_MODEL", "uma-s-1p1")''',
     '''# noise_floor_eV conflated two different failures and so measured
# neither. They are now separate fields.
#
#   model_resolution_eV  - the model's own energy resolution. Below this a
#       number is not distinguishable from zero however tightly the
#       optimiser converged. A property of the checkpoint.
#
#   reproducibility_eV   - the spread of the whole pipeline over repeat
#       runs from perturbed starting geometries. A property of the harness,
#       dominated by which basin the fragments land in, and in practice
#       much larger than the model resolution. None means NOT YET MEASURED,
#       and any claim about a difference smaller than it is unsupported.
#
# Both default to the legacy noise_floor_eV where a model has not been
# measured, so nothing silently becomes more confident than it was.
for _spec in MODELS.values():
    _spec.setdefault("model_resolution_eV", _spec.get("noise_floor_eV"))
    _spec.setdefault("reproducibility_eV", None)

DEFAULT_MODEL = os.environ.get("MLIP_MODEL", "uma-s-1p1")''',
     "defaults for the split noise floor")


# --- src/tools.py -----------------------------------------------------

edit("src/tools.py",
     '''STRUCTURE_TOOLS = [build_slab, build_stepped_slab, place_adsorbate, build_dissociated_endpoint]''',
     '''# Imported here rather than at the top of the file to keep the dependency
# one-way: zpe reaches into calculators and store, never back into tools.
from src.zpe import check_zpe_applied, compute_zpe_correction  # noqa: E402

STRUCTURE_TOOLS = [build_slab, build_stepped_slab, place_adsorbate, build_dissociated_endpoint]''',
     "import the zpe tools")

edit("src/tools.py", '''    compute_gas_referenced_barrier,
    read_results,
]''',
     '''    compute_gas_referenced_barrier,
    compute_zpe_correction,
    read_results,
]''',
     "register compute_zpe_correction")

edit("src/tools.py", '''    check_path_resolved,
    validation_summary,
]''',
     '''    check_path_resolved,
    check_zpe_applied,
    validation_summary,
]''',
     "register check_zpe_applied")

edit("src/tools.py", '''    expected = {"convergence", "noise_floor", "dispersion",
                "dispersion_consistent", "gas_reference", "fragments",
                "geometry", "reaction_consistency", "endpoints_distinct",
                "path_resolved"}''',
     '''    expected = set(config.REQUIRED_CHECKS)''',
     "summary reads the shared required set")


# --- src/graph.py -----------------------------------------------------

edit("src/graph.py", '''    if store.all_checks_passed():
        print("\\n[gate] all checks passed, finishing")
        return END

    checks = store.validation()
    if not checks:
        print("\\n[gate] supervisor said FINISH but nothing is validated "
              "-> Validation_Agent")
    else:
        failed = [k for k, ok in checks.items() if not ok]
        print(f"\\n[gate] supervisor said FINISH but these failed: {failed}")
    return "Validation_Agent"''',
     '''    # all_checks_passed() is all(checks.values()), which is vacuously
    # satisfiable: a run that executed three checks and passed them looks
    # identical to one that executed eleven. The gate exists precisely to
    # stop a model deciding it is done, so it has to require the full set,
    # not merely the absence of failures among whatever happened to run.
    checks = store.validation()
    missing = [name for name in config.REQUIRED_CHECKS if name not in checks]
    failed = [k for k, ok in checks.items() if not ok]

    if not missing and not failed:
        print("\\n[gate] all required checks passed, finishing")
        return END

    if not checks:
        print("\\n[gate] supervisor said FINISH but nothing is validated "
              "-> Validation_Agent")
    else:
        if missing:
            print(f"\\n[gate] supervisor said FINISH but these never ran: {missing}")
        if failed:
            print(f"\\n[gate] supervisor said FINISH but these failed: {failed}")
    return "Validation_Agent"''',
     "exit_gate enforces the required set")


# --- src/prompt.py ----------------------------------------------------

edit("src/prompt.py", '''  6. Call compute_gas_referenced_barrier
''',
     '''  6. Call compute_gas_referenced_barrier
  7. Call compute_zpe_correction

Step 7 is not optional and the barrier is not final without it. The band
gives a classical electronic barrier. The quantity this benchmark is scored
against is the zero-point corrected one, which is a different number by
0.03 to 0.15 eV on every reaction of this kind, always in the same
direction. compute_zpe_correction needs a confirmed first-order saddle, so
if refine_saddle has not run and succeeded, run it before step 7.
''',
     "tell the simulation agent to correct the barrier")


# --- src/benchmark.py -------------------------------------------------

UNCERTAINTY = {
    "H2_Cu111": (0.05, "SRP-DFT fitted to molecular beam; no conflicting value reported"),
    "H2_Cu100": (0.07, "beam/SRP-DFT 0.74 against thermal 0.60-0.62"),
    "H2_Pt111": (0.03, "quantum dynamics gives +0.06 and -0.008; set to 0"),
    "H2_Ru0001": (0.03, "dynamics ~4 meV, LAAD near zero; set to 0"),
    "N2_Ru0001_terrace": (0.48, "thermal 1.3, LAAD 1.8, NN-PES 1.84, QCT 2.27"),
    "N2_Ru0001_step": (0.10, "single thermal measurement, step-dominated kinetics"),
    "CH4_Ru0001": (0.23, "beam 0.38, thermal 0.53, LAAD 0.80, beam threshold 0.85"),
    "CH4_Ni100": (0.08, "thermal 0.61 against beam dynamics 0.76"),
    "CH4_Ni111_terrace": (0.12, "thermal 0.77 against beam + lattice-coupled 1.01"),
    "CH4_Ni111_step": (0.15, "DERIVED as 1.01 - 0.21, not measured directly"),
}


def patch_benchmark(text):
    """Insert reference_uncertainty_eV after each reference_eV line."""
    out, changed = [], 0
    current = None
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        for key in UNCERTAINTY:
            if stripped.startswith(f'"{key}":'):
                current = key
        out.append(line)
        if stripped.startswith('"reference_eV":') and current:
            if "reference_uncertainty_eV" in text and f'# {current}' in text:
                pass
            indent = line[:len(line) - len(line.lstrip())]
            band, why = UNCERTAINTY[current]
            out.append(f'{indent}"reference_uncertainty_eV": {band},\n')
            out.append(f'{indent}"uncertainty_note": "{why}",\n')
            changed += 1
            current = None
    if changed != len(UNCERTAINTY):
        raise SystemExit(
            f"benchmark.py: inserted {changed} uncertainties, expected "
            f"{len(UNCERTAINTY)}. Not writing. Check the SBH10 dict keys.")
    return "".join(out)


def main():
    root = Path(".")
    if not (root / "config.py").exists():
        raise SystemExit("Run this from the repo root (config.py not found).")

    already = 0
    staged = {}

    for path, old, new, why in EDITS:
        text = staged.get(path, path.read_text())
        # `new` contains `old`, so testing for `new` is the only reliable
        # idempotence check: testing for the first line of `new`, or for the
        # absence of `old`, both report "not yet applied" on an already
        # patched file and duplicate the block.
        if new in text:
            already += 1
            continue
        if old not in text:
            raise SystemExit(
                f"ANCHOR NOT FOUND in {path}: {why}\n"
                f"  looked for: {old.splitlines()[0][:70]}...\n"
                "Nothing has been written. Apply this edit by hand.")
        staged[path] = text.replace(old, new, 1)

    bench = Path("src/benchmark.py")
    btext = staged.get(bench, bench.read_text())
    if "reference_uncertainty_eV" in btext:
        already += 1
    else:
        staged[bench] = patch_benchmark(btext)

    for path, text in staged.items():
        backup = path.with_suffix(path.suffix + ".pre_day1")
        if not backup.exists():
            backup.write_text(path.read_text())
        path.write_text(text)
        print(f"  wrote {path}")

    if already:
        print(f"  {already} edit(s) already applied, left alone")
    print("Wiring done.")


if __name__ == "__main__":
    sys.exit(main())
