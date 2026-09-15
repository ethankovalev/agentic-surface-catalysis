# Seeded track: foundation MLIP accuracy at published SBH10 transition states

**provenance: seeded.** Every number here comes from evaluating a model at the
BEEF vdW transition state geometry published in the SBH10 Supporting
Information, with nothing moved. No geometry search is involved. This measures
whether the potential energy surface is correct at a known saddle. It is not
evidence that the engine can find a saddle on its own, which is a separate
question answered only by the autonomous track.

The quantity is `barrier_at_reference_geometry_eV`, a single point evaluation.
It exists for all ten reactions in all eight series regardless of whether saddle
refinement, connectivity or the Hessian succeeded afterwards, which is why there
are no gaps below.

## Headline

| Model | Surfaces in training? | Dispersion | MAE (eV) | RMSE (eV) | Errors below reference |
|---|---|---|---:|---:|---:|
| UMA S 1.1 | yes | **off** | **0.166** | 0.222 | 4 of 10 |
| UMA S 1.1 | yes | on | 0.445 | 0.502 | **10 of 10** |
| MACE mh 1 | yes | **off** | **0.174** | 0.263 | 7 of 10 |
| MACE mh 1 | yes | on | 0.378 | 0.460 | **10 of 10** |
| Orb v3 | no | off | 0.473 | 0.566 | **10 of 10** |
| Orb v3 | no | on | 0.746 | 0.812 | **10 of 10** |
| MACE mpa 0 | no | off | 0.769 | 1.037 | 8 of 10 |
| MACE mpa 0 | no | on | 1.030 | 1.280 | **10 of 10** |

BEEF vdW, the best performing functional in the original SBH10 paper, reports
MAE 0.14 eV against these same references.

Two independent axes separate these eight series, and they are separable because
the design holds one constant while varying the other.

## Finding 1: dispersion makes every model worse

D3 off is the better setting on 9 of 10 reactions for UMA, 9 of 10 for MACE mh 1,
and 10 of 10 for both out of domain models. Across all four models with
dispersion on, **40 out of 40 errors fall below the reference.** That is not
scatter, it is a systematic one directional bias.

Mean shift from turning dispersion on:

| Model | D3 damping | Mean shift (eV) |
|---|---|---:|
| UMA S 1.1 | rpbe | 0.433 downward |
| MACE mh 1 | pbe | 0.273 downward |
| Orb v3 | pbe | 0.273 downward |
| MACE mpa 0 | pbe | 0.273 downward |

**The three pbe damped models give shifts identical to within 1 meV on every
single reaction**, despite MACE applying dispersion through its own internal
`dispersion=True` path and Orb applying it as a summed `torch_dftd` calculator.
Completely different code, different architectures, different training data,
same number. UMA sits apart because it uses rpbe damping, not pbe.

This is what it should look like if the correction depends only on geometry and
damping parameters and not at all on the network's own prediction, which is the
premise of adding D3 as a separate term. It also means the overcorrection cannot
be blamed on any individual model. It is a property of applying these damping
parameters to these systems.

As a side effect the two implementations validate each other: an independent
reimplementation agreeing to 1 meV on ten separate systems is a stronger check
than either could provide alone.

## Finding 2: training domain dominates architecture

| Group | Models | Mean MAE, D3 off | Saddles confirmed |
|---|---|---:|---|
| Surfaces in training | UMA S 1.1, MACE mh 1 | **0.170 eV** | 4 to 6 of 10 |
| Bulk crystals only | Orb v3, MACE mpa 0 | **0.621 eV** | **0 of 10 for both** |

A 3.7x gap in mean absolute error, and a categorical split in whether a first
order saddle can be confirmed at all.

The crucial control here is **MACE mpa 0**. It shares a backend with MACE mh 1,
which means `_build_mace` forces `default_dtype="float64"` for both regardless of
checkpoint. Same numerics, same dispersion mechanism, same code path. The only
difference is that mh 1 saw surfaces in training and mpa 0 saw only bulk
crystals. The out of domain model still confirms zero saddles out of ten, and is
worse in magnitude than Orb despite Orb running at float32.

That rules out numerical precision as the explanation for the out of domain
failures. The first suspicion when Orb returned zero confirmed saddles was that
float32 finite differences were too coarse to resolve Hessian curvature, which
would have been a pipeline artifact rather than a result. A float64 out of domain
model failing identically settles it.

Worst cases are not marginal. MACE mpa 0 places the CH4/Ni(111) step transition
state at 1.755 eV **below** the asymptotic reactant state, against a real
reference of 0.800 eV above it. That is a 2.56 eV error and a physically
impossible sign. Orb places the N2/Ru(0001) step transition state below its
reactants too. These are not models that are somewhat less accurate off domain,
they are models with no learned signal for the process at all.

## Per reaction, dispersion off

| Reaction | Reference | UMA | err | MACE mh 1 | err | Orb v3 | err | MACE mpa 0 | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H2/Cu(111) | 0.630 | 0.713 | +0.083 | 0.664 | +0.034 | 0.240 | 0.390 low | 0.344 | 0.286 low |
| H2/Cu(100) | 0.740 | 0.760 | +0.020 | 0.640 | 0.100 low | 0.256 | 0.484 low | 0.331 | 0.409 low |
| H2/Pt(111) | 0.000 | 0.022 low | 0.022 | 0.127 low | 0.127 | 0.204 low | 0.204 | 0.732 low | 0.732 |
| H2/Ru(0001) | 0.000 | 0.295 low | 0.295 | 0.166 low | 0.166 | 0.565 low | 0.565 | 0.327 low | 0.327 |
| N2/Ru(0001) terrace | 1.840 | 1.339 | 0.501 low | 1.113 | 0.727 low | 0.593 | 1.247 low | 0.820 | 1.020 low |
| N2/Ru(0001) step | 0.400 | 0.453 | +0.053 | 0.316 | 0.084 low | 0.362 low | 0.762 low | 0.419 | +0.019 |
| CH4/Ru(0001) | 0.800 | 1.097 | +0.297 | 1.057 | +0.257 | 0.450 | 0.350 low | 0.840 | +0.040 |
| CH4/Ni(100) | 0.760 | 0.900 | +0.140 | 0.741 | 0.019 low | 0.524 | 0.236 low | 0.998 low | 1.758 low |
| CH4/Ni(111) terrace | 1.010 | 1.184 | +0.174 | 1.065 | +0.055 | 0.851 | 0.159 low | 0.146 | 0.864 low |
| CH4/Ni(111) step | 0.800 | 0.727 | 0.073 low | 0.627 | 0.173 low | 0.465 | 0.335 low | 1.432 low | 2.232 low |

## Per reaction, dispersion on

| Reaction | Reference | UMA | MACE mh 1 | Orb v3 | MACE mpa 0 |
|---|---:|---:|---:|---:|---:|
| H2/Cu(111) | 0.630 | 0.381 | 0.454 | 0.029 | 0.133 |
| H2/Cu(100) | 0.740 | 0.369 | 0.440 | 0.056 | 0.131 |
| H2/Pt(111) | 0.000 | 0.302 low | 0.300 low | 0.378 low | 0.905 low |
| H2/Ru(0001) | 0.000 | 0.549 low | 0.341 low | 0.740 low | 0.502 low |
| N2/Ru(0001) terrace | 1.840 | 0.903 | 0.737 | 0.218 | 0.445 |
| N2/Ru(0001) step | 0.400 | 0.305 | 0.180 | 0.497 low | 0.283 |
| CH4/Ru(0001) | 0.800 | 0.567 | 0.676 | 0.069 | 0.459 |
| CH4/Ni(100) | 0.760 | 0.215 | 0.370 | 0.153 | 1.369 low |
| CH4/Ni(111) terrace | 1.010 | 0.464 | 0.683 | 0.468 | 0.236 low |
| CH4/Ni(111) step | 0.800 | 0.177 | 0.304 | 0.141 | 1.755 low |

## Cross model divergence on CH4/Ru(0001)

UMA could not resolve this reaction in either track. The autonomous run found the
dissociated endpoint drifting 3.9 Å back toward recombination on an independent
relaxation. The seeded run produced a persistent second imaginary mode between 8
and 11 meV across every refinement attempt and both dispersion settings. A site
placement hypothesis was tested and ruled out: fcc and hcp starting hollows
relaxed to an identical geometry with energies agreeing to about 1 µeV.

MACE mh 1 resolves the same reaction cleanly at the same published geometry: a
confirmed first order saddle, confirmed connectivity, a full zero point
correction, and a barrier of 0.870 eV against the 0.800 eV reference.

The reaction is hard for UMA's potential energy surface specifically. It is not
intrinsically hard. Surfacing a statement that precise is what the two track
architecture is for.

## Shared weakness on N2/Ru(0001) terrace

The worst reaction for three of the four models and every dispersion setting for
UMA, MACE mh 1 and Orb. It is also the highest barrier in the set and the one
with the widest spread in the underlying experimental literature, from 1.3 eV
thermal to 2.27 eV from quasi classical trajectory work, against the 1.84 eV
chosen as the reference. Some of that discrepancy may sit in the reference rather
than in the models, and the honest position is that this one is not yet
separable.

## Fully validated barriers

Where the whole chain succeeded, meaning a confirmed first order saddle,
confirmed connectivity and an applied zero point correction:

| Reaction | Reference | UMA off | UMA on | MACE mh 1 off | MACE mh 1 on |
|---|---:|---:|---:|---:|---:|
| H2/Cu(111) | 0.630 | **0.657** | 0.264 | 0.616 | 0.401 |
| H2/Cu(100) | 0.740 | not confirmed | 0.246 | not confirmed | not confirmed |
| N2/Ru(0001) terrace | 1.840 | 1.090 | 0.535 | 1.034 | 0.614 |
| N2/Ru(0001) step | 0.400 | 0.490 | 0.233 | 0.238 | 0.070 |
| CH4/Ru(0001) | 0.800 | not confirmed | not confirmed | **0.870** | 0.480 |
| CH4/Ni(111) terrace | 1.010 | 1.111 | 0.413 | not confirmed | not confirmed |

Orb v3 confirmed one reaction (N2/Ru(0001) terrace, dispersion on) out of twenty
attempts. MACE mpa 0 confirmed none.

H2/Cu(111) with UMA and dispersion off, at 0.657 eV against a 0.630 eV reference,
is the closest single number in the project.

## Caveat that applies to the headline table

These are classical barriers compared against a zero point corrected reference.
SBH10's tabulated values include a zero point correction of between 0.03 and
0.14 eV, always negative. Applying the paper's own Table 2 corrections to the
UMA D3 off column moves its MAE from 0.166 to 0.145 eV, closer still to BEEF
vdW. Those are the paper's zero point values rather than this project's computed
ones, so that number is indicative rather than a result. The full zero point
pipeline runs wherever a first order saddle is confirmed, and those numbers
appear in the validated table above.

## Reproducing

```bash
python build_seeds.py data_sbh10_si/poscars.txt --out work_seeds/

# UMA environment
python scripts/run_seeded.py --seeds work_seeds/ --model uma-s-1p1 --d3 off
python scripts/run_seeded.py --seeds work_seeds/ --model uma-s-1p1 --d3 on

# MACE and Orb environment
python scripts/run_seeded.py --seeds work_seeds/ --model mace-mh-1            --d3 off
python scripts/run_seeded.py --seeds work_seeds/ --model mace-mh-1            --d3 on
python scripts/run_seeded.py --seeds work_seeds/ --model orb-v3-cons-inf-omat --d3 off
python scripts/run_seeded.py --seeds work_seeds/ --model orb-v3-cons-inf-omat --d3 on
python scripts/run_seeded.py --seeds work_seeds/ --model mace-mpa-0           --d3 off
python scripts/run_seeded.py --seeds work_seeds/ --model mace-mpa-0           --d3 on

python plot_seeded_results.py
```

No sweep in this track calls the Anthropic API. UMA and MACE need separate
environments; see the README setup section.
