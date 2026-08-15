# Agentic SBH10 benchmark

An agent that autonomously builds, relaxes, and computes dissociation
barriers on transition metal surfaces, then benchmarks itself against
the SBH10 experimental reference set.

## Why this benchmark

SBH10 (Sharada, Bligaard, Luntz, Kroes, Nørskov; *J. Phys. Chem. C*
2017, 121, 19807) is ten dissociation barriers on transition metal
surfaces, referenced to molecular beam scattering, laser-assisted
associative desorption, and thermal experiments.

Two properties make it the right target.

**The references are experimental, not computational.** Agreement means
agreement with reality, not with another calculation.

**Dispersion is the discriminating axis.** In the original study the
dispersion-corrected BEEF-vdW reached 0.14 eV mean error, beating both a
meta-GGA and a screened hybrid functional — the reverse of the typical
gas-phase pattern. The model used here (UMA, `oc20` task) is trained on
RPBE, which has no dispersion term at all. That is a sharp, testable
hypothesis, not a vague gap.

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

export ANTHROPIC_API_KEY=...
export UMA_CHECKPOINT=/path/to/uma-s-1p2.pt   # or place it in data/
export UMA_DEVICE=cuda                        # if you have a GPU
```

The UMA checkpoint is gated: request access at
[huggingface.co/facebook/UMA](https://huggingface.co/facebook/UMA).

## Running

```bash
python invoke.py --single H2_Cu111     # one reaction, verbose — start here
python invoke.py --all                 # the full SBH10 set
python invoke.py --task "..."          # ad-hoc task, not scored
```

Start with `--single`. If the agent gets H₂ on Cu(111) badly wrong, that
is worth knowing on day one, not after ten runs.

## Architecture

A supervisor picks which worker acts next; workers report back to it.

- **Structure_Agent** — builds the slab, places the adsorbate,
  constructs the dissociated endpoint
- **Simulation_Agent** — relaxes the endpoints and the gas-phase
  reference, runs the climbing-image NEB, converts the barrier to a
  gas-phase reference
- **Validation_Agent** — runs the physical sanity checks

**One deliberate departure from the standard supervisor pattern: the
supervisor cannot end the run.** `exit_gate` in `src/graph.py` reads the
validation record directly and refuses to exit until every check has
actually run and passed. A model that talks itself into "looks good to
me" is exactly the failure this project exists to catch, so the
guarantee lives in code, not in a prompt.

### The checks

Every check is designed to work without knowing what the answer should
be. None of them compares against the reference value.

| Check | Catches |
|---|---|
| `convergence` | energies from non-converged optimisations |
| `noise_floor` | results below the model's own ~0.009 eV MAE |
| `dispersion` | adsorbate drifting away because RPBE has no vdW term |
| `geometry` | atoms driven into the surface; barrier peak at an endpoint |
| `reaction_consistency` | endothermic reaction with a barrier below its reaction energy; chemically impossible magnitudes |
| `endpoints_distinct` | a "barrier" between two states that turned out to be the same minimum |
| `path_resolved` | a band too coarse to sample the transition state — one image-to-image step carrying most of the climb |

`endpoints_distinct` is what makes a zero barrier interpretable. Two of
the ten reactions (H₂ on Pt(111) and Ru(0001)) are genuinely
non-activated, so a near-zero result is correct for those. Checking that
the endpoints are physically different states distinguishes a real
non-activated reaction from a failed calculation, without the validator
being told which is which.

### Blind evaluation

The agent never sees the reference barrier. No tool reaches
`src/benchmark.py`, and none of the prompts mentions it. Comparison
happens in the runner, after the graph has returned. Any other
arrangement measures how well the agent can curve-fit, not how well it
can calculate.

### Gas-phase referencing

SBH10 barriers are measured relative to a **free molecule**, not to a
physisorbed one. A NEB run between a physisorbed initial state and a
dissociated final state therefore measures against the wrong reference,
short by the physisorption well depth.

The OC20 task is not trained on isolated molecules, so the fix is not to
remove the slab. Instead `build_gas_reference` lifts the molecule ~8 Å
clear of the same slab in the same cell, and the slab contribution
cancels in the difference:

well_depth = E_gasref − E_initial
barrier_gas = barrier_neb + well_depth


`compute_gas_referenced_barrier` performs this conversion and reports the
well depth as a diagnostic. For H₂ on Cu the well is shallow and the two
conventions nearly agree; for CH₄ on Ni or Ru, where dispersion binding
is stronger, the correction is expected to matter more.

## Current status

One reaction has been run end to end. H₂/Cu(111) gave a barrier of
**0.672 eV** against an SBH10 reference of 0.63 eV — but that run
predates both the gas-phase referencing and the `path_resolved` check,
and its energy profile jumped 0.525 eV in a single image-to-image step,
which is 78% of the barrier. It needs rerunning with more images and
proper referencing before it means anything.

Treat this as evidence the pipeline works, not as a result.

## Project layout

```text
agentic-surface-catalysis/
  README.md
  requirements.txt
  .gitignore
  config.py
  prompts.py
  main.py
  src/
    __init__.py
    tools.py
    store.py
    calculators.py
  benchmarks/
    sbh10.py
  tests/
    test_tools.py
    test_checks.py
    test_endpoint_bonding.py
  docs/
    VALIDATION.md
```


## What is not finished

- **Two of the ten reactions need stepped surfaces.** `N2_Ru0001_step`
  and `CH4_Ni111_step` are measured at step sites, where under-
  coordinated edge atoms lower the barrier substantially — N₂ on
  Ru(0001) drops from 1.84 eV on the terrace to 0.40 eV at a step.
  `build_slab` only produces flat terraces, so these two cannot
  currently be computed correctly. No validation check will catch this:
  both step values sit comfortably inside a plausible range.
- **No site or orientation sampling.** One configuration per reaction,
  so the reported barrier is not a minimum over configuration space.
  Fragments can settle at atop sites when a hollow site is more stable.
- **Site placement is approximate.** `place_adsorbate` picks ontop,
  bridge, or hollow from the first few top-layer atoms rather than doing
  a proper geometric site search.
- **The fragment-radius estimate is crude for polyatomics.**
  `build_dissociated_endpoint` derives the starting height from the mean
  covalent radius of all adsorbate atoms. For CH₄ that averages one
  carbon with four hydrogens and underestimates where a CH₃ fragment
  should sit, so the four CH₄ reactions start from a worse geometry than
  the H₂ ones.
- **No repeat runs.** Every number so far is a single run with no
  estimate of its own variability.

## Known limitations of the underlying model

- OC20/RPBE has no dispersion term. This is why D3 is included in the
  relaxation loop by default (`with_d3=True`), not applied afterward as
  a single-point correction — the geometry shift under dispersion is
  most of the effect.
- D3 is known to overbind on some metal surfaces. Agreement with
  experiment should be read as suggestive, not confirmatory, without a
  second independent check.
- UMA's benchmarked MAE against its own reference DFT is roughly
  0.009 eV. Any computed quantity below that is not resolvable from
  zero, regardless of how tightly the optimiser converges.

## Troubleshooting

**`create_react_agent() got unexpected keyword arguments:
{'state_modifier': ...}`** — LangGraph renamed this argument to
`prompt`. Change all three occurrences in `src/graph.py`.

**`KeyError: 'messages'`** — some LangGraph versions yield node-keyed
chunks from `.stream()` rather than full state. `run_worker` in
`src/graph.py` passes `stream_mode="values"` and guards on the key
being present; if you see this, check both are in place.

**`zsh: bad assignment`** when setting the API key — shell variable
names cannot contain hyphens. It is `ANTHROPIC_API_KEY` with
underscores, and no spaces around the `=`.

## Reference

Sharada, S. M.; Bligaard, T.; Luntz, A. C.; Kroes, G.-J.; Nørskov, J. K.
*SBH10: A Benchmark Database of Barrier Heights on Transition Metal
Surfaces.* J. Phys. Chem. C 2017, 121 (36), 19807–19815.