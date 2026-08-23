# Agentic surface catalysis: a cross-model barrier benchmark

An agent that autonomously builds, relaxes, and computes dissociation barriers on
transition metal surfaces — then benchmarks **every public foundational MLIP**
against the same ten experimentally referenced barriers, with dispersion as an
explicit variable rather than a buried default.

The agent is the harness. The result is the comparison.

---

## The question

Universal machine-learning interatomic potentials are now good enough that
people use them to screen catalysts. Reaction rates depend on barriers
exponentially — rate ∝ exp(−E<sub>a</sub>/k<sub>B</sub>T) — so at 500 K a
0.2 eV barrier error is roughly a 100× rate error. Chemical accuracy is
1 kcal/mol ≈ 0.043 eV.

Most published MLIP evaluation reports energies and forces on equilibrium
structures. Barriers are different: the transition state is, by construction,
a configuration the model was probably never trained on.

**So: how accurate are foundational MLIPs on barriers, referenced to experiment
rather than to more DFT, and does adding dispersion help or hurt?**

That question is not settled, and the answer is not the same for every model.

---

## Why SBH10

SBH10 (Sharada, Bligaard, Luntz, Kroes, Nørskov; *J. Phys. Chem. C* 2017, 121,
19807) is ten dissociation barriers on transition metal surfaces, referenced to
molecular beam scattering, laser-assisted associative desorption, and thermal
experiments.

Two properties make it the right target.

**The references are experimental, not computational.** Agreement means
agreement with reality, not with whichever DFT setup generated a model's
training data. Almost every MLIP benchmark in circulation compares against DFT,
which measures how well a model reproduces its teacher — a different and easier
question.

**Dispersion is the discriminating axis.** In the original study the
dispersion-corrected BEEF-vdW reached 0.14 eV mean error, beating both a
meta-GGA and a screened hybrid — the reverse of the typical gas-phase pattern.
Several of the models tested here are trained on RPBE or PBE data with no
dispersion term at all. That is a sharp, testable hypothesis, not a vague gap.

The set is small enough to run exhaustively across many models, and hard enough
that models disagree.

---

## The models

| Key | Backend | Training domain | Surfaces in domain? | Access |
|---|---|---|---|---|
| `uma-s-1p2` | fairchem | OC20 + OMat24 + OMol25 + ODAC23 + OMC25 | **yes** (`oc20` task) | gated |
| `mace-mh-1` | mace | OMat24 pretrain, multi-head | **yes** (`oc20_usemppbe` head) | open, ASL — academic only |
| `mace-mpa-0` | mace | Materials Project + Alexandria | no — bulk crystals | open |
| `orb-v3-cons-inf-omat` | orb | OMat24 | no — bulk crystals | open |
| `esen-sm-cons-omol` | fairchem | OMol25 (isolated molecules, ωB97M-V) | no — molecular | gated |

`training_domain` and `in_domain_for_surfaces` are recorded in `config.MODELS`
and carried into every results table. They are not decoration. OC20 is
adsorbates on slabs; OMat24 is bulk inorganic crystals; OMol25 is isolated
molecules at a molecular level of theory. Running a molecular model on a
periodic slab is out of domain **by construction** — including those models is
the point, but a table that doesn't say so reads as a fair fight when it isn't.

The interesting result is not a leaderboard. It is *which training domain
transfers to surface barriers, and how much the mismatch costs.*

### Dispersion is per model, not global

D3 damping parameters are fitted to a specific functional — RPBE for OC20-based
models, PBE for OMat24-based ones. `d3_xc` is therefore a per-model field.
Using one setting across models would silently invalidate every dispersion
comparison in the benchmark.

Three mechanisms are handled separately:

- **`torch_dftd`** — D3 added as a second additive calculator (UMA, Orb), so it
  contributes forces *during* relaxation rather than as a single-point
  correction afterwards. The geometry shift under dispersion is most of the
  effect.
- **`builtin`** — MACE applies dispersion internally via `dispersion=True`.
- **`none`** — ωB97M-V already includes dispersion; adding D3 would double-count
  it, so `new_calculator` raises rather than silently returning the bare model.

---

## Setup

Two environments are required. **UMA and MACE have conflicting `e3nn` version
requirements and cannot coexist.** Orb lives with MACE.

```bash
# environment 1: UMA
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# environment 2: MACE + Orb
python3.13 -m venv .venv-mace
source .venv-mace/bin/activate
pip install --upgrade pip
export CMAKE_POLICY_VERSION_MINIMUM=3.5        # see Troubleshooting
pip install ase numpy torch-dftd mace-torch orb-models
```

```bash
export ANTHROPIC_API_KEY=...
export UMA_CHECKPOINT=/path/to/uma-s-1p2.pt    # or place it in data/
export MLIP_DEVICE=cuda                        # if you have a GPU
```

### Checkpoints

FAIR Chemistry weights are gated: request access at
[huggingface.co/facebook/UMA](https://huggingface.co/facebook/UMA) and
[/facebook/OMol25](https://huggingface.co/facebook/OMol25), agree to the
licence, then place the file in `data/`.

MACE weights are open:

```bash
hf download mace-foundations/mace-mh-1  --local-dir data/mace-mh-1
hf download mace-foundations/mace-mpa-0 --local-dir data/mace-mpa-0
ls data/mace-mh-1 data/mace-mpa-0       # confirm filenames, update config.MODELS
```

Orb resolves its own weights by tag. `config.py` sets `HF_HOME` to
`data/hf_cache/` so every download lands inside the project and later runs are
network-free — which matters, because a benchmark whose weights silently
redownload or update between runs is not reproducible.

### Verify before computing anything

```bash
python -c "
from ase.build import fcc111
from src.calculators import new_calculator
slab = fcc111('Cu', size=(3,3,4), vacuum=10.0)
slab.pbc = True
for key in ['mace-mh-1', 'mace-mpa-0', 'orb-v3-cons-inf-omat']:
    for d3 in (False, True):
        slab.calc = new_calculator(key, with_d3=d3)
        print(f'{key:<24} d3={str(d3):<5} E={slab.get_potential_energy():10.4f} eV')
"
```

Every `d3=True` row should be more negative than its `d3=False` counterpart by a
physically sane amount. If a row is off by tens of eV, the damping/functional
pairing for that model is wrong, and every dispersion number downstream would be
meaningless.

---

## Running

```bash
python invoke.py --single H2_Cu111     # one reaction, verbose — start here
python invoke.py --all                 # the full SBH10 set
python invoke.py --task "..."          # ad-hoc task, not scored
MLIP_MODEL=mace-mh-1 python invoke.py --all    # a different model
```

Start with `--single`. If the agent gets H₂ on Cu(111) badly wrong, that is
worth knowing on day one, not after ten runs.

Because the two environments are separate, a full cross-model grid is run once
per environment and the results merged afterwards.

## Architecture

A supervisor picks which worker acts next; workers report back to it.

- **Structure_Agent** — builds the slab, places the adsorbate, constructs the
  dissociated endpoint
- **Simulation_Agent** — relaxes the endpoints and the gas-phase reference, runs
  the climbing-image NEB, converts the barrier to a gas-phase reference
- **Validation_Agent** — runs the physical sanity checks

**One deliberate departure from the standard supervisor pattern: the supervisor
cannot end the run.** `exit_gate` in `src/graph.py` reads the validation record
directly and refuses to exit until every check has actually run and passed. A
model that talks itself into "looks good to me" is exactly the failure this
project exists to catch, so the guarantee lives in code, not in a prompt.

### The checks

Every check is designed to work without knowing what the answer should be.
None of them compares against the reference value.

| Check | Catches |
|---|---|
| `convergence` | energies from non-converged optimisations |
| `noise_floor` | results below the model's own resolvable floor |
| `dispersion` | adsorbate drifting away because the model has no vdW term |
| `geometry` | atoms driven into the surface; barrier peak at an endpoint |
| `reaction_consistency` | endothermic reaction with a barrier below its reaction energy; chemically impossible magnitudes |
| `endpoints_distinct` | a "barrier" between two states that turned out to be the same minimum |
| `path_resolved` | a band too coarse to sample the transition state — one image-to-image step carrying most of the climb |

`endpoints_distinct` is what makes a zero barrier interpretable. Two of the ten
reactions (H₂ on Pt(111) and Ru(0001)) are genuinely non-activated, so a
near-zero result is correct for those. Checking that the endpoints are
physically different states distinguishes a real non-activated reaction from a
failed calculation, without the validator being told which is which.

### Blind evaluation

The agent never sees the reference barrier. No tool reaches `src/benchmark.py`,
and none of the prompts mentions it. Comparison happens in the runner, after the
graph has returned. Any other arrangement measures how well the agent can
curve-fit, not how well it can calculate.

### Gas-phase referencing

SBH10 barriers are measured relative to a **free molecule**, not to a
physisorbed one. A NEB run between a physisorbed initial state and a
dissociated final state therefore measures against the wrong reference, short
by the physisorption well depth.

The OC20 task is not trained on isolated molecules, so the fix is not to remove
the slab. Instead `build_gas_reference` lifts the molecule ~8 Å clear of the
same slab in the same cell, and the slab contribution cancels in the
difference:

```
well_depth  = E_gasref − E_initial
barrier_gas = barrier_neb + well_depth
```

`compute_gas_referenced_barrier` performs this conversion and reports the well
depth as a diagnostic. For H₂ on Cu the well is shallow and the two conventions
nearly agree; for CH₄ on Ni or Ru, where dispersion binding is stronger, the
correction is expected to matter more.

## Current status

**No cross-model results yet.** The harness runs, the model registry is
verified across all five backends, but the grid has not been executed. Nothing
in this repository should be cited as a benchmark result today.

What is actually true right now:

- **The registry works.** All three open models (`mace-mh-1`, `mace-mpa-0`,
  `orb-v3-cons-inf-omat`) construct and evaluate, with and without dispersion,
  and D3 deltas agree to 0.01 eV across models — the correct behaviour for an
  additive geometric term.
- **One reaction has been run end to end, and it does not count yet.**
  H₂/Cu(111) gave a barrier of **0.672 eV** against an SBH10 reference of
  0.63 eV — but that run predates the gas-phase referencing, the
  `path_resolved` check, and the entire multi-model refactor, and its energy
  profile jumped 0.525 eV in a single image-to-image step, which is 78% of the
  barrier. It is being rerun.
- **A silent head-selection bug was found and fixed.** `mace-mh-1` was being
  requested with head `"oc20"`, which does not exist; MACE fell back to
  `omat_pbe` — a *bulk crystal* head — with only a warning. Every MACE surface
  number would have been wrong in a way that looked entirely normal. The
  correct head is `oc20_usemppbe`. Treat this as the representative failure
  mode of this whole exercise: the errors that matter do not raise exceptions.
- **`esen-sm-cons-omol` is not yet runnable** — the OMol25 gate is separate
  from the UMA gate and access has not been granted.

Treat all of the above as evidence the pipeline works, not as results.

## Project layout

```text
agentic-surface-catalysis/
├── .gitignore
├── README.md
├── config.py                  # model registry, thresholds, paths
├── invoke.py                  # entry point
├── requirements.txt
├── data/                      # checkpoints and HF cache (gitignored)
└── src/
    ├── __init__.py
    ├── agent.py               # builds the three agents
    ├── benchmark.py           # SBH10 reference barriers — never reachable by a tool
    ├── calculators.py         # multi-backend calculator registry + dispersion
    ├── graph.py               # supervisor routing and the exit gate
    ├── prompt.py              # the three agent prompts
    ├── store.py               # run state and check results
    └── tools.py               # structure, simulation and validation tools
```

## What is not finished

- **Two of the ten reactions need stepped surfaces.** `N2_Ru0001_step` and
  `CH4_Ni111_step` are measured at step sites, where under-coordinated edge
  atoms lower the barrier substantially — N₂ on Ru(0001) drops from 1.84 eV on
  the terrace to 0.40 eV at a step. `build_slab` only produces flat terraces,
  so these two cannot currently be computed correctly. No validation check
  will catch this: both step values sit comfortably inside a plausible range.
  That 1.44 eV spread is also the single most discriminating pair in the set,
  so this is the most valuable missing piece, not the least.
- **Noise floors are placeholders.** `noise_floor_eV: 0.3` is a guess in every
  registry entry. The real per-model figure requires repeat runs with
  perturbed initial conditions, and is itself one of the intended results.
- **No site or orientation sampling.** One configuration per reaction, so the
  reported barrier is not a minimum over configuration space. Fragments can
  settle at atop sites when a hollow site is more stable.
- **Site placement is approximate.** `place_adsorbate` picks ontop, bridge, or
  hollow from the first few top-layer atoms rather than doing a proper
  geometric site search.
- **The fragment-radius estimate is crude for polyatomics.**
  `build_dissociated_endpoint` derives the starting height from the mean
  covalent radius of all adsorbate atoms. For CH₄ that averages one carbon
  with four hydrogens and underestimates where a CH₃ fragment should sit, so
  the four CH₄ reactions start from a worse geometry than the H₂ ones.
- **No repeat runs.** Every number so far is a single run with no estimate of
  its own variability.
- **No out-of-distribution set yet.** Magnetic metals, oxides, and
  metal–oxide interfaces are where universal MLIPs are known to degrade most,
  and none are currently covered.

## Known limitations of the underlying models

- OC20/RPBE has no dispersion term. This is why D3 is included in the
  relaxation loop by default (`with_d3=True`), not applied afterward as a
  single-point correction — the geometry shift under dispersion is most of
  the effect.
- D3 is known to overbind on some metal surfaces. Agreement with experiment
  should be read as suggestive, not confirmatory, without a second
  independent check.
- UMA's benchmarked MAE against its own reference DFT is roughly 0.009 eV.
  Any computed quantity below that is not resolvable from zero, regardless
  of how tightly the optimiser converges. The equivalent floor for the other
  models is not yet measured.
- Absolute D3 energies on dense metal slabs are large (order 0.4 eV/atom) and
  mostly cancel in a barrier, which is an energy *difference* along a path.
  The quantity of interest is how dispersion changes that difference, not
  the magnitude of the correction itself.

## Troubleshooting

**`dm-tree` fails to build: `CMake must be installed`** — install cmake
(`brew install cmake`). If it then fails with *"Compatibility with CMake < 3.5
has been removed"*, set `export CMAKE_POLICY_VERSION_MINIMUM=3.5` before
installing.

**`AttributeError: module 'config' has no attribute 'DEVICE'`** — `DEVICE`
must be defined in `config.py` before `MODELS`.

**`ModuleNotFoundError: orb_models.forcefield.calculator`** — newer
`orb-models` moved this to `orb_models.forcefield.inference.calculator`.
`_build_orb` tries both.

**`NotImplementedError: We do not support periodicity along a subset of
axes`** — Orb requires full 3D periodicity. `ase.build.fcc111(...,
vacuum=...)` returns `pbc=[True, True, False]`. `build_slab` sets
`slab.pbc = True`; any ad-hoc script must do the same, or models are being
compared under different boundary conditions.

**`WARNING: Head <name> not found in available heads [...], defaulting to
the last head`** — do not ignore this. MACE silently falls back to a
different head. Check the printed list and set the exact name in
`config.MODELS`.

**`create_react_agent() got unexpected keyword arguments:
{'state_modifier': ...}`** — LangGraph renamed this argument to `prompt`.
Change all three occurrences in `src/graph.py`.

**`KeyError: 'messages'`** — some LangGraph versions yield node-keyed chunks
from `.stream()` rather than full state. `run_worker` in `src/graph.py`
passes `stream_mode="values"` and guards on the key being present; if you see
this, check both are in place.

**`zsh: bad assignment`** when setting the API key — shell variable names
cannot contain hyphens. It is `ANTHROPIC_API_KEY` with underscores, and no
spaces around the `=`.

## Reference

Sharada, S. M.; Bligaard, T.; Luntz, A. C.; Kroes, G.-J.; Nørskov, J. K.
*SBH10: A Benchmark Database of Barrier Heights on Transition Metal
Surfaces.* J. Phys. Chem. C 2017, 121 (36), 19807–19815.