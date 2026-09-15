# Agentic surface catalysis: a cross model barrier benchmark

An agent that autonomously builds, relaxes, and computes dissociation barriers on
transition metal surfaces, then benchmarks **every public foundational MLIP**
against the same ten experimentally referenced barriers, with dispersion as an
explicit variable rather than a buried default.

The agent is the harness. The result is the comparison.

---

## The question

Universal machine learning interatomic potentials are now good enough that
people use them to screen catalysts. Reaction rates depend on barriers
exponentially: rate ∝ exp(−E<sub>a</sub>/k<sub>B</sub>T). At 500 K a 0.2 eV
barrier error is roughly a 100× rate error. Chemical accuracy is
1 kcal/mol ≈ 0.043 eV.

Most published MLIP evaluation reports energies and forces on equilibrium
structures. Barriers are different: the transition state is, by construction,
a configuration the model was probably never trained on.

**So: how accurate are foundational MLIPs on barriers, referenced to experiment
rather than to more DFT, and does adding dispersion help or hurt?**

That question is not settled. The current best answer from this repository, now
measured on two independent models, is that **dispersion hurts, substantially
and consistently.** See [Current status](#current-status).

---

## Two tracks, and why they must never be averaged together

A barrier benchmark conflates two different questions. This repository separates
them, and stamps `provenance` on every stored result so they cannot be silently
recombined.

| | **Seeded track** | **Autonomous track** |
|---|---|---|
| Starting geometry | the published BEEF vdW transition state, from the SBH10 SI | built from scratch by the agent |
| Question answered | is the model's energy right at a known saddle? | can the engine find the saddle at all? |
| What it measures | accuracy of the potential energy surface | capability of the search |
| Entry point | `scripts/run_seeded.py` | `invoke.py`, `scripts/run_grid.py` |
| `provenance` | `seeded` | `autonomous` |
| Failure risk | very low: no search to fail | high: this is the hard part |
| Anthropic API | never called | called on every supervisor turn |

**The autonomous track is the product claim.** A customer with a novel catalyst
has no SBH10 entry to start from, by definition. A headline number that quietly
depended on literature geometries would not be evidence that the engine works.

**The seeded track is the diagnostic.** It isolates PES accuracy from search
capability, so that when a reaction fails it is possible to say which of the two
broke. It also guarantees a number on every reaction even when the search does
not converge, which is what makes the failures interpretable rather than blank.
Because it calls tools directly rather than through the agent, it also costs
nothing in API credit, which matters more than it sounds: the autonomous sweep
is the expensive one.

Both are separately comparable to the SBH10 reference. Neither is comparable to
the other without stating the cell and the geometry source. See
[Cell sensitivity](#cell-sensitivity).

---

## Why SBH10

SBH10 (Sharada, Bligaard, Luntz, Kroes, Nørskov; *J. Phys. Chem. C* 2017, 121,
19807) is ten dissociation barriers on transition metal surfaces, referenced to
molecular beam scattering, laser assisted associative desorption, and thermal
experiments.

Three properties make it the right target.

**The references are experimental, not computational.** Agreement means
agreement with reality, not with whichever DFT setup generated a model's
training data. Almost every MLIP benchmark in circulation compares against DFT,
which measures how well a model reproduces its teacher, a different and easier
question.

**Dispersion is the discriminating axis.** In the original study the
dispersion corrected BEEF vdW reached 0.14 eV mean error, beating both a
meta GGA and a screened hybrid, the reverse of the typical gas phase pattern.
Several of the models tested here are trained on RPBE or PBE data with no
dispersion term at all. That is a sharp, testable hypothesis, not a vague gap.

**The SI publishes the transition state geometries.** All ten BEEF vdW saddles
are given as VASP POSCARs, plus Table S1 listing the final adsorption sites.
This is what makes the seeded track possible at all. `data_sbh10_si/poscars.txt`
holds the transcribed POSCARs; `build_seeds.py` parses, tags, and validates them.

The set is small enough to run exhaustively across many models, and hard enough
that models disagree.

### Reference values are not exact, and the table now says so

Several SBH10 references are contested in the source literature.
`src/benchmark.py` carries a `reference_uncertainty_eV` and an
`uncertainty_note` per reaction, because a model cannot beat the experiment's
own spread and a table that pretends otherwise is the first thing a serious
reader attacks.

| Reaction | Reference | Spread in the literature |
|---|---|---|
| `N2_Ru0001_terrace` | 1.84 | 1.3 thermal, 1.8 LAAD, 1.84 chosen, 2.27 QCT |
| `CH4_Ru0001` | 0.80 | 0.38 beam, 0.53 thermal, 0.80 LAAD, 0.85 beam threshold |
| `CH4_Ni111_terrace` | 1.01 | 0.77 thermal against 1.01 beam and lattice coupled |
| `CH4_Ni111_step` | 0.80 | **derived** as 1.01 minus 0.21, not measured directly |
| `H2_Cu100` | 0.74 | 0.74 beam and SRP DFT against 0.60 to 0.62 thermal |

SBH17 (Kroes et al., *J. Chem. Theory Comput.* 2023) revises
`CH4_Ni111_step` to **0.699 eV**, a computed SRP DFT value rather than a
subtraction. This repository still uses SBH10's 0.80 for strict comparability
with the original paper. **Open decision**, logged in `NOTES.md`.

---

## The models

| Key | Backend | Training domain | Surfaces in domain? | Access |
|---|---|---|---|---|
| `uma-s-1p1` | fairchem | OC20 + OMat24 + OMol25 + ODAC23 + OMC25 | **yes** (`oc20` task) | gated |
| `mace-mh-1` | mace | OMat24 pretrain, multi head | **yes** (`oc20_usemppbe` head) | open, ASL academic only |
| `mace-mpa-0` | mace | Materials Project and Alexandria | no, bulk crystals | open |
| `orb-v3-cons-inf-omat` | orb | OMat24 | no, bulk crystals | open |
| `esen-sm-cons-omol` | fairchem | OMol25 (isolated molecules, ωB97M V) | no, molecular | gated |

`training_domain` and `in_domain_for_surfaces` are recorded in `config.MODELS`
and carried into every results table. They are not decoration. OC20 is
adsorbates on slabs; OMat24 is bulk inorganic crystals; OMol25 is isolated
molecules at a molecular level of theory. Running a molecular model on a
periodic slab is out of domain **by construction**. Including those models is
the point, but a table that does not say so reads as a fair fight when it is not.

The interesting result is not a leaderboard. It is *which training domain
transfers to surface barriers, and how much the mismatch costs.*

**On UMA S 1.2.** The 1.2 checkpoint is incompatible with every installable
`fairchem-core` version tested (2.14.0, 2.13.0, 2.5.0, 1.10.0): the saved
`eSCNMDBackbone` config carries parameters the current library does not accept.
This benchmark therefore uses UMA S 1.1, which works with `fairchem-core`
2.14.0. This is a documented limitation, not a preference.

### Dispersion is per model, not global

D3 damping parameters are fitted to a specific functional, RPBE for OC20 based
models, PBE for OMat24 based ones. `d3_xc` is therefore a per model field.
Using one setting across models would silently invalidate every dispersion
comparison in the benchmark.

Three mechanisms are handled separately:

- **`torch_dftd`**: D3 added as a second additive calculator (UMA, Orb), so it
  contributes forces *during* relaxation rather than as a single point
  correction afterwards. The geometry shift under dispersion is most of the
  effect.
- **`builtin`**: MACE applies dispersion internally via `dispersion=True`.
- **`none`**: ωB97M V already includes dispersion; adding D3 would double count
  it, so `new_calculator` raises rather than silently returning the bare model.

A useful invariant falls out of the additive mechanism: two models that share a
`d3_xc` value must produce an identical D3 contribution on identical geometry,
because the correction depends only on positions and damping parameters, never
on the network's own prediction. Measured on a Cu(111) 3x3x4 slab, `mace-mh-1`
and `mace-mpa-0` both give a correction of 0.4293 eV per atom, agreeing to six
decimal places. Worth keeping as a sanity check.

---

## Setup

Two environments are required. **UMA and MACE have conflicting `e3nn` version
requirements and cannot coexist.** `mace-torch` pins `e3nn==0.4.4`;
`fairchem-core` requires `e3nn>=0.5`. Installing MACE into a working UMA
environment breaks UMA outright, with an import failure deep inside the
checkpoint loader rather than at install time. Orb lives with MACE.

```bash
# environment 1: UMA
python3.10 -m venv .venv-uma
source .venv-uma/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# environment 2: MACE and Orb
python3.10 -m venv .venv-mace
source .venv-mace/bin/activate
pip install --upgrade pip
export CMAKE_POLICY_VERSION_MINIMUM=3.5        # see Troubleshooting
pip install -r requirements-mace.txt
```

```bash
export ANTHROPIC_API_KEY=...
export UMA_CHECKPOINT=/path/to/uma-s-1p1.pt    # or place it in data/
export MLIP_DEVICE=cuda                        # if you have a GPU
export GRID_RESULTS_DIR=/workspace/agentic-surface-catalysis/results/grid
export SEEDED_RESULTS_DIR=/workspace/agentic-surface-catalysis/results/seeded
```

**Set one export per line.** `export A=1 export B=2` on a single line does not
set `B`; it exports a variable literally named `export B`. `MLIP_DEVICE` falling
back to its `cpu` default this way costs an entire overnight sweep and raises no
error. A preflight that asserts on `MLIP_DEVICE == "cuda"` is worth running at
the start of every session.

### GPU architecture matters

`.venv-mace` pins `torch 2.6.0+cu124`, which supports compute capability up to
`sm_90`. **Blackwell cards are `sm_120` and will not run it**, failing with
"no kernel image is available for execution on the device". On Runpod that
rules out RTX 5090, RTX PRO 4500, RTX PRO 6000 and B200. Ada and Ampere cards
work: RTX 4090 and RTX 2000 Ada are both `sm_89`, A100 is `sm_80`, A6000 is
`sm_86`. The Runpod GPU id string is exact, for instance
`"NVIDIA RTX 2000 Ada Generation"`; check with `runpodctl gpu list`.

Note that `torch.cuda.is_available()` can return `True` on an unsupported
architecture while every kernel launch still fails. Force a real computation to
test properly:

```bash
python -c "import torch; x = torch.randn(10, device='cuda'); print(torch.__version__, float(x.sum()))"
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
network free. This matters, because a benchmark whose weights silently
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
physically sane amount, order 0.4 eV per atom on a dense metal slab. If a row is
off by tens of eV, the damping and functional pairing for that model is wrong,
and every dispersion number downstream would be meaningless.

Then check the structure builder, which needs no GPU and no checkpoint:

```bash
python scripts/check_structures.py      # expect 10 of 10, exit 0
python -m pytest tests/test_day1.py -q  # expect 16 passed
```

---

## Running

A single reaction, verbose. Start here:

```bash
python invoke.py --single H2_Cu111
```

**Seeded track.** Build the seed structures once, then run:

```bash
python build_seeds.py data_sbh10_si/poscars.txt --out work_seeds/
python scripts/run_seeded.py --seeds work_seeds/ --model uma-s-1p1 --d3 off
python scripts/run_seeded.py --seeds work_seeds/ --only H2_Cu111 --d3 on
```

`build_seeds.py` validates as it parses: metal nearest neighbour distances
against bulk, adsorbate above the surface, breaking bond stretched past
equilibrium, and where SBH17 publishes a geometry, the breaking bond length
against it. H2/Cu(111) and N2/Ru(0001) agree with SBH17 to 0.004 and 0.007 Å,
which is what confirms the transcription is sound.

**Autonomous track**, resumable and budget bounded:

```bash
python scripts/run_grid.py --models uma-s-1p1 --dry-run        # list planned runs
python scripts/run_grid.py --models uma-s-1p1 --d3 on          # run them
python scripts/run_grid.py --models uma-s-1p1 --d3 on --limit 2
python scripts/run_grid.py --models uma-s-1p1 --d3 on --only CH4_Ni100
```

`run_grid.py` writes one JSON per completed run and skips anything already
present **unless that result carries a `run_error`**, in which case it is
re attempted. `--limit` counts reactions that actually ran, not plan entries, so
`--limit 2` on a mostly finished sweep attempts exactly two real reactions
rather than being consumed by instant skips. Failures are logged and do not halt
the batch.

**Cell sensitivity**, agent excluded:

```bash
python scripts/cell_test.py --reaction H2_Cu111 --d3 off
```

**Summary and figure**, no GPU required:

```bash
python scripts/save_seeded_summary.py   # writes SEEDED_SUMMARY.md
python plot_seeded_results.py           # writes seeded_d3_comparison.png
```

Because the two environments are separate, a full cross model grid is run once
per environment and the results merged afterwards.

### Settings the agent cannot vary

Four quantities are pinned outside the agent's control, because a sweep in
which each run silently chose its own is not a comparable dataset.

- **NEB force tolerance is fixed at 0.05 eV/Å** inside `run_neb` and is not a
  tool argument. It was previously agent settable with a default of 0.10, so
  runs could and did converge to different bars while all reporting success.
- **`FORCE_MODEL`** overrides the model key the agent passes, so the grid can
  pin one model per run. `config.DEFAULT_MODEL` is read once at import, so
  setting `MLIP_MODEL` mid process does nothing.
- **`FORCE_D3`** overrides the dispersion flag the agent passes, for the same
  reason. Unset, both behave exactly as before. The enforced value is resolved
  by `calculators.effective_with_d3`, which is also what every tool records into
  the store, so the calculator and the metadata cannot disagree.
- **`config.REQUIRED_CHECKS`** is the single list of checks a run must pass,
  read by both `validation_summary` and `exit_gate`, so the two cannot disagree
  about what validated means.

---

## Architecture

A supervisor picks which worker acts next; workers report back to it.

- **Structure_Agent**: builds the slab, places the adsorbate, constructs the
  dissociated endpoint
- **Simulation_Agent**: relaxes the endpoints and the gas phase reference, runs
  the climbing image NEB, converts the barrier to a gas phase reference, applies
  the zero point correction
- **Validation_Agent**: runs the physical sanity checks

**One deliberate departure from the standard supervisor pattern: the supervisor
cannot end the run.** `exit_gate` in `src/graph.py` reads the validation record
directly and refuses to exit until every check in `config.REQUIRED_CHECKS` has
actually run and passed. A model that talks itself into "looks good to me" is
exactly the failure this project exists to catch, so the guarantee lives in code,
not in a prompt.

The gate previously tested `all(checks.values())`, which is vacuously
satisfiable: a run that executed three checks and passed them looked identical to
one that executed eleven. It now requires the full set **by name**, so a check
that never ran is a failure rather than an absence.

This is not hypothetical. On 2026-09-14 the agent produced a polished final
report for `CH4_Ni111_step`, headline barrier bolded, caveats listed, reading
entirely like a resolved result. The gate refused it, because
`check_saddle_connects` had returned a 0.02 Å separation between the two
displaced and relaxed endpoints, too small to tell forward from backward.

### The checks

Every check is designed to work without knowing what the answer should be.
None of them compares against the reference value.

| Check | Catches |
|---|---|
| `convergence` | energies from non converged optimisations |
| `noise_floor` | results below the model's own resolvable floor |
| `dispersion` | adsorbate drifting away because the model has no vdW term |
| `dispersion_consistent` | a barrier assembled from a mix of D3 on and D3 off steps, which is not a barrier on any single energy surface |
| `gas_reference` | a missing gas phase reference, or a negative well depth meaning the physisorbed state is unbound |
| `fragments` | the wrong bond breaking, or the molecule not dissociating at all |
| `geometry` | atoms driven into the surface; barrier peak at an endpoint |
| `reaction_consistency` | endothermic reaction with a barrier below its reaction energy; chemically impossible magnitudes |
| `endpoints_distinct` | a barrier between two states that turned out to be the same minimum |
| `path_resolved` | a band too coarse to sample the transition state, one image to image step carrying most of the climb |
| `zpe` | a classical barrier reported against a zero point corrected reference |

`endpoints_distinct` is what makes a zero barrier interpretable. Two of the ten
reactions (H₂ on Pt(111) and Ru(0001)) are genuinely non activated, so a
near zero result is correct for those. Checking that the endpoints are
physically different states distinguishes a real non activated reaction from a
failed calculation, without the validator being told which is which.

### Blind evaluation

The agent never sees the reference barrier. No tool reaches `src/benchmark.py`,
and none of the prompts mentions it. Comparison happens in the runner, after the
graph has returned. Any other arrangement measures how well the agent can
curve fit, not how well it can calculate.

Site type is the exception, and deliberately so. Whether a reaction is measured
at a terrace or a step is part of the problem specification, not the answer, so
`run_one` passes it into the task prompt. It previously did not, which meant
`N2_Ru0001_step` and `N2_Ru0001_terrace` sent byte identical instructions while
being scored against references 1.44 eV apart.

The same reasoning is why `SBH10_PREFERRED_SITE`, which encodes Table S1's final
adsorption sites, is **not** wired into the autonomous pipeline. A final state
site taken from the reference calculation is much closer to the answer than a
terrace or step label is. Where site choice needs testing, the blind approach is
to build both candidates and let the model's own energy decide, which is what
`scripts/probe_site_fix.py` does.

**The seeded track is not blind, by construction**, and nothing in it may be
described as if it were. It reads a published transition state. That is the
whole point of stamping `provenance: seeded` on every one of its results and
writing them to a separate directory.

---

## Gas phase referencing

SBH10 barriers are measured relative to a **free molecule**, not to a
physisorbed one. A NEB run between a physisorbed initial state and a
dissociated final state therefore measures against the wrong reference, short
by the physisorption well depth.

The OC20 task is not trained on isolated molecules, so the fix is not to remove
the slab. Instead `build_gas_reference` lifts the molecule about 8 Å clear of
the same slab in the same cell, and the slab contribution cancels in the
difference:

```
well_depth  = E_gasref − E_initial
barrier_gas = barrier_neb − well_depth
```

The sign matters and is easy to get backwards. A positive well depth means the
physisorbed minimum sits *below* the free molecule, so a trajectory starting
from gas is already part way up the hill and the gas referenced barrier must be
smaller than the barrier measured from the physisorbed state.

`compute_gas_referenced_barrier` performs this conversion and reports the well
depth as a diagnostic. For H₂ on Cu the well is shallow and the two conventions
nearly agree; for CH₄ on Ni or Ru, and for N₂ at a Ru step, the correction is
large.

The seeded track uses SBH10's own convention directly: E<sub>b</sub> =
E<sub>TS</sub> − E<sub>asym</sub>, both in the same supercell with the same slab
geometry, the molecule relaxed 8 Å above it. Rebuilding the slab for the
asymptotic state instead of reusing the saddle's own would put a real energy
difference into the reference that has nothing to do with the reaction.

## Zero point correction

**SBH10's tabulated quantity is not a classical barrier.** From the paper,
Section 2.1: transition state energies are the *zero point corrected* difference
between the transition state and the isolated gas phase molecule. Table 2 lists
the correction applied per reaction, and all ten are negative, between 0.03 and
0.14 eV.

A NEB or a refined saddle gives the classical electronic barrier. Scoring that
directly against SBH10 is a convention mismatch of 0.03 to 0.14 eV **in a fixed
direction on every reaction in the set**. That is the same order as the errors
being interpreted, and it biases every computed barrier high, which flatters a
model that undershoots and punishes one that overshoots. Since the dispersion
axis is exactly a question of undershoot against overshoot, omitting this does
not add noise to the conclusion, it inverts it.

`src/zpe.py` computes

```
dZPE          = ZPE(transition state) − ZPE(gas phase molecule)
barrier(zpe)  = barrier(gas referenced, classical) + dZPE
```

Harmonic, over the adsorbate atoms only, slab rigid. Mode bookkeeping is where
this goes wrong silently and is done explicitly: the transition state keeps every
real mode including frustrated translations and rotations and drops exactly one
imaginary mode; the gas molecule drops 5 (linear) or 6 (nonlinear) free modes **by
magnitude**, because at 8 Å a free rotation lands on either side of zero depending
on finite difference noise and filtering by sign keeps some and drops others.

Every one of these raises rather than returning a number: no confirmed first order
saddle, zero or more than one imaginary mode, a residual imaginary mode at the TS,
an imaginary mode in the gas reference outside the free modes, a mismatched
dispersion setting between barrier and Hessian, or |dZPE| greater than 0.5 eV.
There is no clamping and no fallback to zero.

Sanity checks: H₂ gas keeps 1 mode at ZPE about 0.27 eV, CH₄ keeps 9 at about
1.19 eV. Computed dZPE for H₂/Cu(111) is 0.024 eV with dispersion off and 0.075
eV with it on, against the paper's BEEF vdW value of 0.06 eV, all negative. Same
sign, right order of magnitude. Agreement is not exact and is not claimed to be,
because it is a different potential energy surface.

---

## Current status

**Seeded track complete for two models across both dispersion settings, 40
barrier evaluations with no gaps. Autonomous track partially complete.** Orb and
eSEN pending.

### Headline: dispersion overcorrects, on both models

`barrier_at_reference_geometry_eV` is a single point at the published BEEF vdW
saddle with nothing moved. No search, no refinement, no Hessian. It therefore
exists for every reaction regardless of whether anything downstream converged.

| Model | Dispersion | MAE (eV) | RMSE (eV) | Errors below reference |
|---|---|---:|---:|---:|
| UMA S 1.1 | **off** | **0.166** | 0.222 | 4 of 10 |
| UMA S 1.1 | on | 0.445 | 0.502 | **10 of 10** |
| MACE mh 1 | **off** | **0.174** | 0.263 | 7 of 10 |
| MACE mh 1 | on | 0.378 | 0.460 | **10 of 10** |

BEEF vdW, the best functional in the original paper, reports 0.14 eV against the
same references.

Three things make this a real finding rather than a single model quirk:

**D3 off wins 9 of 10 reactions, independently, for each model.** Not an
aggregate effect driven by one or two outliers.

**With D3 on, all ten errors are below the reference, for both models.** Twenty
out of twenty. That is a systematic bias, not scatter.

**Two unrelated models agree closely with D3 off.** UMA at 0.166 eV and MACE at
0.174 eV, an 8 meV difference in aggregate, from different architectures trained
on different corpora. The mean downward shift from turning D3 on is 0.433 eV for
UMA and 0.273 eV for MACE.

The full per reaction table is in `results_seeded_summary.md`, and the figure is
`seeded_d3_comparison.png`.

### Cross model divergence: CH4/Ru(0001)

Worth stating precisely, because it is exactly what the two track architecture
exists to surface.

UMA could not resolve this reaction in either track. The autonomous run found
the dissociated endpoint drifting 3.9 Å back toward recombination on an
independent relaxation. The seeded run produced a persistent second imaginary
mode between 8 and 11 meV across every refinement attempt and both dispersion
settings. A site placement hypothesis was tested on 2026-09-14, comparing fcc
and hcp starting hollows, and ruled out: both relaxed to an identical geometry
with energies agreeing to about 1 µeV.

MACE resolves the same reaction cleanly at the same published geometry: a
confirmed first order saddle, confirmed connectivity, a full zero point
correction, and a barrier of 0.870 eV against the 0.800 eV reference.

The reaction is hard for UMA's potential energy surface specifically. It is not
intrinsically hard.

### Shared weakness: N2/Ru(0001) terrace

The worst reaction for both models and both settings, and it stays worst with
dispersion off: UMA 0.501 eV low, MACE 0.727 eV low. It is also the highest
barrier in the set and the one with the widest experimental spread, 1.3 to 2.27
eV against a 1.84 eV reference. Some of that discrepancy may sit in the
reference rather than in the models, and the honest position is that this one is
not yet separable.

### Autonomous track

Nine reaction and dispersion combinations attempted before an Anthropic credit
exhaustion halted the sweep. Six validated:

| Reaction | D3 | computed | validated |
|---|---|---:|---|
| H2_Cu111 | off | 0.737 | yes |
| H2_Cu111 | on | 0.448 | yes |
| H2_Cu100 | off | 0.687 | yes |
| H2_Pt111 | off | 0.237 | yes |
| H2_Ru0001 | off | 0.272 low | yes |
| N2_Ru0001_terrace | off | 1.405 | yes |
| CH4_Ni100 | off | 0.315 | no |
| CH4_Ru0001 | off | 1.088 | no |
| N2_Ru0001_step | off | 0.274 | no |
| CH4_Ni111_terrace | off | 1.116 | no |
| CH4_Ni111_step | off | 0.488 | no |

`CH4_Ni111_terrace` is worth noting even though it did not validate. Its
autonomous value of 1.116 eV sits 4 meV from the seeded track's independent
measurement of 1.111 eV for the same model and dispersion setting, from a
completely different starting geometry. The gate refused it on convergence
grounds, but the physics agrees with itself.

### Cell sensitivity

The seeded track runs in SBH10's cell (2x2x6, quarter monolayer); the autonomous
track runs 3x3x4 (one ninth). `scripts/cell_test.py` runs the identical scripted
pipeline, agent excluded, at both, holding geometry source and slab flexibility
constant with two mobile layers in each:

| Cell | Coverage | Barrier | Classical | dZPE | Error against 0.63 |
|---|---|---:|---:|---:|---:|
| 3x3x4 | 0.111 ML | 0.744 | 0.764 | 0.019 | +0.114 |
| 2x2x6 | 0.250 ML | 0.711 | 0.737 | 0.027 | +0.081 |

**Gap 0.034 eV.** Both confirm genuine first order saddles with full mode
following connectivity. Cell size is *not* what separates the two tracks; the
existing 3x3x4 results stand and the pipeline is not migrating. The larger
seeded against autonomous difference on the same reaction, 0.657 against 0.744
with dispersion off, is attributable to geometry source and relaxation freedom,
not coverage.

Note also that the experimental references are low coverage quantities, so 3x3
at one ninth of a monolayer is *closer* to the physical limit than 2x2 at a
quarter. Matching SBH10's cell would match their methodology, not the physics
their reference encodes. Their 2x2 choice was a stated compromise to keep hybrid
functionals affordable, a constraint this project does not have.

### What the gate caught

Four seeded reactions produced converged saddles that were refused. All four
refusals were correct, and the diagnosis differs in each case:

- **`CH4_Ni100`**: saddle relaxed to a C–H bond of 1.124 Å against a normal
  1.07 Å, a drift of 0.786 Å from the 1.909 Å seed. It collapsed back to an
  intact molecule. Not a transition state.
- **`CH4_Ni111_step`**: saddle relaxed to 2.268 Å; pushing further apart barely
  moves it and compressing barely recovers. It sits in the **product basin**,
  not between reactant and product.
- **`H2_Pt111`, `H2_Ru0001`**: both references are about zero, genuinely non
  activated. Sella cannot converge onto a saddle that barely exists, and
  displacement based connectivity cannot separate forward from backward on a
  flat surface. An algorithmic limit for this regime, not a defect in the model
  or the saddle.
- **`CH4_Ru0001`, `H2_Cu100`**: genuine second imaginary modes, 10 meV and 23
  meV, stable and reproduced across both dispersion settings and multiple
  refinement attempts. The 23 meV on Cu(100) is over four times the 5 meV noise
  threshold. Hypothesis for Cu(100): the hollow to hollow saddle sits on a four
  fold symmetry axis that UMA's surface treats as a ridge between two lower
  symmetry saddles. Untested.

### Silent bugs found and fixed

In every case the pipeline produced a plausible looking number and raised no
exception. Treat these as the representative failure mode of this whole exercise:
**the errors that matter do not raise exceptions.**

- **CH₄ dissociation broke the wrong bond.** `_orient_for_dissociation` returned
  polyatomics unchanged, so ASE's g2 CH₄ went down edge first, and the
  longest bonded pair rule with a strict `>` comparison deterministically
  selected atom 1, the hydrogen pointing **away** from the metal, since all four
  C–H bonds are 1.0897 Å. The band then had to break a C–H bond with the
  hydrogen nowhere near a metal atom: gas phase homolysis, about 4.5 eV, plus
  strain. That is the 7 eV observed on CH₄/Ru(0001) against a 0.8 eV reference.
  The fix ranks candidates by the terminal atom's **height**, and rotates one
  bond to point at the surface.
- **`FORCE_D3` never reached the stored metadata.** `new_calculator` reassigned
  its own local `with_d3`, which Python passes by value, so the calculator was
  built correctly while every tool stored its own original argument. A sweep run
  under `FORCE_D3=off` produced files named `..._d3off.json` whose every record
  claimed `with_d3=True`. `check_dispersion_consistent` could not catch it
  because all stages were wrong in the same direction. Fixed by making
  `effective_with_d3` the single source of truth for both the calculator and the
  record.
- **`run_grid.py` treated crashed runs as completed**, skipping any reaction
  whose result file existed even when that file recorded an API failure and a
  null barrier. Eleven placeholder results sat looking finished until a manual
  audit found them.
- **Gas phase referencing added the physisorption well depth instead of
  subtracting it**, inflating every bound reaction barrier by twice the well
  depth.
- **`site_type` never reached the agent**, so step and terrace reactions sent
  byte identical prompts while scored against references 1.44 eV apart.
- **`NEBTools.get_barrier()` defaults to a spline fit**, which overshot the
  highest computed image by 0.3 eV on under resolved bands. Now `fit=False`.
- **`mace-mh-1` was requested with head `"oc20"`, which does not exist**; MACE
  fell back to `omat_pbe`, a *bulk crystal* head, with only a warning. Every
  MACE surface number would have been wrong in a way that looked entirely
  normal.
- **`check_structures.py` measured the wrong bond.** It compared `ads_i[0]` to
  `ads_i[1]`, the first two adsorbate atoms by index, which for CH₄ is the
  carbon and a hydrogen that *stays in the CH₃ fragment*, reading 1.09 Å in both
  endpoints by definition. Invisible until the orientation fix changed which
  bond breaks.
- **`_classify_hollow` used the wrong subsurface layer on stepped slabs**,
  comparing a layer against itself and labelling all 34 sites fcc, and returned
  a confident fcc label for four fold hollows where the distinction does not
  exist at all.

---

## Project layout

```text
agentic-surface-catalysis/
├── .gitignore
├── README.md
├── NOTES.md                       # running log of findings and open questions
├── config.py                      # model registry, thresholds, REQUIRED_CHECKS
├── invoke.py                      # entry point, single reaction
├── build_seeds.py                 # SBH10 SI POSCARs to tagged, validated ASE
├── plot_seeded_results.py         # headline figure, no GPU required
├── patch_*.py                     # applied source patches, kept as documentation
├── requirements.txt
├── data/                          # checkpoints and HF cache (gitignored)
├── data_sbh10_si/
│   ├── poscars.txt                # transcribed SBH10 SI transition states
│   └── expected_manifest.json     # validation record for the above
├── work_seeds/                    # generated seed structures (gitignored)
├── scripts/
│   ├── run_grid.py                # resumable sweep, reactions x models x D3
│   ├── run_seeded.py              # seeded track: published TS to barrier
│   ├── cell_test.py               # controlled 3x3x4 against 2x2x6, agent excluded
│   ├── probe_site_fix.py          # blind fcc against hcp site comparison
│   ├── compare_site_geometries.py # free geometry diff of two relaxed endpoints
│   ├── check_structures.py        # geometry checks, no GPU
│   └── save_seeded_summary.py     # writes SEEDED_SUMMARY.md
├── tests/
│   └── test_day1.py               # orientation and ZPE bookkeeping, CPU only
└── src/
    ├── __init__.py
    ├── agent.py                   # builds the three agents
    ├── benchmark.py               # SBH10 references and uncertainties, tool unreachable
    ├── calculators.py             # multi backend registry, dispersion, effective_with_d3
    ├── graph.py                   # supervisor routing and the exit gate
    ├── prompt.py                  # the three agent prompts
    ├── store.py                   # run state and check results
    ├── tools.py                   # structure, simulation and validation tools
    └── zpe.py                     # zero point correction and its validator
```

## What is not finished

- **The autonomous sweep is incomplete.** Six validated results out of twenty
  planned reaction and dispersion combinations. The remainder halted on API
  credit exhaustion, not on a technical failure, and the resume path is fixed.
- **Orb and eSEN have not been swept.** Two of the five registered models remain
  untested. MACE is now done for both dispersion settings, which is what makes
  the headline a cross model result rather than a single model one.
- **Noise floors are still placeholders.** `noise_floor_eV: 0.3` has been split
  into `model_resolution_eV` and `reproducibility_eV`. The second is `None`,
  **not yet measured**, and is itself one of the intended results. Until it
  exists, no claim that any two barriers differ significantly is supported.
- **The step reactions do not resolve their transition states autonomously.**
  `N2_Ru0001_step` runs end to end and dissociates at the edge but fails
  `path_resolved`: at default image counts a single image to image step carries
  up to 89% of the climb, so the reported barrier is a lower bound on a
  transition state that was never sampled.
- **`CH4_Ru0001` is unresolved on UMA, cause narrowed but not found.** The
  dissociated endpoint is not a genuine local minimum: re relaxing `final.traj`
  independently drifts the departed H from 3.96 Å back to 0.03 Å, near complete
  recombination. The fcc against hcp site hypothesis was tested and ruled out.
  MACE resolves the same reaction cleanly, so this is specific to UMA's surface.
- **`check_saddle_connects` fails on stepped geometries.** On `CH4_Ni111_step`
  the displaced and relaxed endpoints differed by 0.02 Å, too little to
  distinguish. The displacement magnitude is probably too small for a C–H bond
  at a step edge. Candidate follow up.
- **Two reactions have no published transition state anywhere.** CH₄/Ni(100) and
  CH₄/Ru(0001) are experiment only in SBH17's Table 2, blank in every geometry
  column. The seeded track still evaluates them from the SBH10 SI POSCARs, but
  there is no independent geometry to cross check against.
- **The Cu(100) symmetry hypothesis is untested.** A small symmetry breaking
  displacement before `refine_saddle` would test whether the 23 meV second mode
  resolves to a genuine nearby first order saddle.
- **No site or orientation sampling in the autonomous track.** One configuration
  per reaction, so the reported barrier is not a minimum over configuration
  space. `probe_site_fix.py` demonstrates the blind approach on one reaction but
  is not wired into the pipeline.
- **No repeat runs.** Every number is a single run with no estimate of its own
  variability.
- **No out of distribution set.** Magnetic metals, oxides, and metal oxide
  interfaces are where universal MLIPs are known to degrade most, and none are
  covered. This is deliberate for now: MLIPs fail *silently* on magnetic
  systems, producing a smooth but wrong potential energy surface, which is the
  worst case for a benchmark whose entire value is trustworthiness.

## Known limitations of the underlying models

- OC20 and RPBE have no dispersion term. This is why D3 was originally included
  in the relaxation loop by default rather than applied afterward as a single
  point correction: the geometry shift under dispersion is most of the effect.
  **On the evidence in this repository, `with_d3=True` is the wrong default for
  surface barriers on both UMA and MACE**, and the registry default should be
  revisited.
- D3 is known to overbind on some metal surfaces. The results here are
  consistent with that and stronger: twenty out of twenty errors below the
  reference with dispersion on, across two independent models.
- UMA's benchmarked MAE against its own reference DFT is roughly 0.009 eV. Any
  computed quantity below that is not resolvable from zero, regardless of how
  tightly the optimiser converged. The equivalent floor for the other models is
  not yet measured.
- Absolute D3 energies on dense metal slabs are large, order 0.4 eV per atom,
  and mostly cancel in a barrier, which is an energy *difference* along a path.
  The quantity of interest is how dispersion changes that difference, not the
  magnitude of the correction.

## Troubleshooting

**`dm-tree` fails to build: `CMake must be installed`.** Install cmake
(`brew install cmake`). If it then fails with "Compatibility with CMake < 3.5
has been removed", set `export CMAKE_POLICY_VERSION_MINIMUM=3.5` before
installing.

**`CUDA capability sm_120 is not compatible with the current PyTorch
installation`.** A Blackwell GPU with a pre Blackwell torch build. Either
request an Ada or Ampere card, or install a CUDA 12.8 or newer torch wheel.
Blackwell support requires CUDA 12.8 and above. See
[GPU architecture matters](#gpu-architecture-matters).

**`RuntimeError: lazy wrapper should be called at most once`.** `torch.det`
initialises its LAPACK backend lazily and that initialisation is not
thread safe. LangGraph runs tools in a thread pool, and UMA calls `torch.det`
on the cell matrix during the forward pass, so the second calculator
constructed inside an agent run raises. Every entry point warms `torch.det` on
the main thread before building the graph. Any new entry point must do the same.

**`AttributeError: module 'config' has no attribute 'DEVICE'`.** `DEVICE`
must be defined in `config.py` before `MODELS`.

**`AttributeError: module 'config' has no attribute 'GRID_RESULTS_DIR'`.**
There is no such attribute. `GRID_RESULTS_DIR` and `SEEDED_RESULTS_DIR` are
**environment variables**, read by their respective scripts at module level with
a `/workspace/...` fallback. `config.py` exposes `WORK_DIR` and `OUTPUT_DIR`
only.

**`TypeError: Object of type bool is not JSON serializable`.** A numpy
comparison returned `numpy.bool_`, which does not subclass Python's `bool` and
has no `json` encoder, unlike `numpy.float64`, which subclasses `float` and
serialises fine. Cast comparisons with `bool()` at the source. `run_seeded.py`
also carries a defensive `default=` encoder so a stray numpy scalar cannot throw
away a finished GPU run at the final write.

**`TypeError: unsupported format string passed to NoneType.__format__`.**
`refine_saddle`'s success message formats the NEB record's `barrier_eV` with
`:.3f`. Any caller that constructs a synthetic NEB record must put a real number
there.

**`ModuleNotFoundError: orb_models.forcefield.calculator`.** Newer
`orb-models` moved this to `orb_models.forcefield.inference.calculator`.
`_build_orb` tries both.

**`NotImplementedError: We do not support periodicity along a subset of
axes`.** Orb requires full 3D periodicity. `ase.build.fcc111(..., vacuum=...)`
returns `pbc=[True, True, False]`. `build_slab` sets `slab.pbc = True`; any
ad hoc script must do the same, or models are being compared under different
boundary conditions.

**`WARNING: Head <name> not found in available heads [...], defaulting to
the last head`.** Do not ignore this. MACE silently falls back to a
different head. Check the printed list and set the exact name in
`config.MODELS`.

**`create_react_agent() got unexpected keyword arguments:
{'state_modifier': ...}`.** LangGraph renamed this argument to `prompt`.

**`KeyError: 'messages'`.** Some LangGraph versions yield node keyed chunks
from `.stream()` rather than full state. `run_worker` in `src/graph.py`
passes `stream_mode="values"` and guards on the key being present.

**`Error code: 400, your credit balance is too low`.** The autonomous track
calls Claude on every supervisor turn. `run_one` catches this *inside* itself
and returns a result with `run_error` set rather than raising, so the sweep
continues and writes placeholder files. `run_grid.py` re attempts any result
carrying `run_error`. The seeded track makes no API calls at all and is
unaffected, which makes it the right thing to run when credit is short.

**A backgrounded run appears frozen, logging nothing.** Python block buffers
stdout when it is redirected to a file. Use `python3 -u`. Check `nvidia-smi` for
real utilisation before assuming a hang.

**`zsh: command not found: #`** or a stuck `quote>` prompt. zsh does not treat
`#` as a comment in interactive input, so an inline comment containing an
apostrophe opens an unterminated quote. Put comments on their own line.

**`zsh: bad assignment`** when setting the API key. Shell variable names
cannot contain hyphens. It is `ANTHROPIC_API_KEY` with underscores, and no
spaces around the `=`.

## Reference

Sharada, S. M.; Bligaard, T.; Luntz, A. C.; Kroes, G.-J.; Nørskov, J. K.
*SBH10: A Benchmark Database of Barrier Heights on Transition Metal
Surfaces.* J. Phys. Chem. C 2017, 121 (36), 19807–19815.

Transition state geometries are taken from that paper's Supporting Information.

Successor set with revised references and published SRP DFT geometries:
Kroes, G.-J. et al. *SBH17: Benchmark Database of Barrier Heights for
Dissociative Chemisorption on Transition Metal Surfaces.*
J. Chem. Theory Comput. 2023, 19 (1), 285–298.
