# Seeded track: foundation MLIP accuracy at published SBH10 transition states

**provenance: seeded.** Every number here comes from evaluating a model at the
BEEF vdW transition state geometry published in the SBH10 Supporting
Information, with nothing moved. No geometry search is involved. This measures
whether the potential energy surface is correct at a known saddle. It is not
evidence that the engine can find a saddle on its own, which is a separate
question answered only by the autonomous track.

The quantity is `barrier_at_reference_geometry_eV`, a single point evaluation.
It exists for all ten reactions in all four series regardless of whether saddle
refinement, connectivity or the Hessian succeeded afterwards, which is why there
are no gaps below.

## Headline

| Model | Dispersion | MAE (eV) | RMSE (eV) | Worst error (eV) | Errors below reference |
|---|---|---:|---:|---:|---:|
| UMA S 1.1 | **off** | **0.166** | 0.222 | 0.501 | 4 of 10 |
| UMA S 1.1 | on | 0.445 | 0.502 | 0.937 | **10 of 10** |
| MACE mh 1 | **off** | **0.174** | 0.263 | 0.727 | 7 of 10 |
| MACE mh 1 | on | 0.378 | 0.460 | 1.103 | **10 of 10** |

BEEF vdW, the best performing functional in the original SBH10 paper, reports
MAE 0.14 eV against these same references.

## Per reaction

| Reaction | Reference | UMA off | err | UMA on | err | MACE off | err | MACE on | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H2/Cu(111) | 0.630 | 0.713 | +0.083 | 0.381 | 0.249 low | 0.664 | +0.034 | 0.454 | 0.176 low |
| H2/Cu(100) | 0.740 | 0.760 | +0.020 | 0.369 | 0.371 low | 0.640 | 0.100 low | 0.440 | 0.300 low |
| H2/Pt(111) | 0.000 | 0.022 low | 0.022 | 0.302 low | 0.302 | 0.127 low | 0.127 | 0.300 low | 0.300 |
| H2/Ru(0001) | 0.000 | 0.295 low | 0.295 | 0.549 low | 0.549 | 0.166 low | 0.166 | 0.341 low | 0.341 |
| N2/Ru(0001) terrace | 1.840 | 1.339 | 0.501 low | 0.903 | 0.937 low | 1.113 | 0.727 low | 0.737 | 1.103 low |
| N2/Ru(0001) step | 0.400 | 0.453 | +0.053 | 0.305 | 0.095 low | 0.316 | 0.084 low | 0.180 | 0.220 low |
| CH4/Ru(0001) | 0.800 | 1.097 | +0.297 | 0.567 | 0.233 low | 1.057 | +0.257 | 0.676 | 0.124 low |
| CH4/Ni(100) | 0.760 | 0.900 | +0.140 | 0.215 | 0.545 low | 0.741 | 0.019 low | 0.370 | 0.390 low |
| CH4/Ni(111) terrace | 1.010 | 1.184 | +0.174 | 0.464 | 0.546 low | 1.065 | +0.055 | 0.683 | 0.327 low |
| CH4/Ni(111) step | 0.800 | 0.727 | 0.073 low | 0.177 | 0.623 low | 0.627 | 0.173 low | 0.304 | 0.496 low |

## What the numbers say

**Dispersion makes both models substantially worse on surface barriers.**
Turning D3 on raises the mean absolute error from 0.166 to 0.445 eV for UMA and
from 0.174 to 0.378 eV for MACE. D3 off is the better setting on 9 of the 10
reactions for each model independently. The mean shift is 0.433 eV downward for
UMA and 0.273 eV downward for MACE.

**With D3 on, every single error is below the reference.** Ten out of ten for
both models. That is not scatter, it is a systematic bias in one direction.

**Two independent models agree closely with D3 off.** UMA at 0.166 eV and MACE
at 0.174 eV differ by 8 meV in aggregate and by 0.118 eV in mean absolute
difference reaction by reaction. Different architectures, different training
corpora, same conclusion. That is the strongest available argument that this is
a property of applying a fitted dispersion correction to metal surfaces rather
than a quirk of one model.

**N2/Ru(0001) terrace is the worst reaction for both models** and stays worst
with dispersion off. It is also the highest barrier in the set and the one with
the widest spread in the underlying experimental literature, from 1.3 eV
thermal to 2.27 eV from quasi classical trajectory work, against the 1.84 eV
chosen as the reference. Some of that 0.5 to 0.7 eV discrepancy may sit in the
reference rather than in the models.

**The two near barrierless reactions behave differently from the rest.**
H2/Pt(111) and H2/Ru(0001) have references of essentially zero, and both models
return small negative values. A negative barrier at a fixed geometry is not
meaningful in the same way as a positive one, so these two contribute to the
MAE without saying much about accuracy.

## Cross model divergence worth recording

CH4/Ru(0001) could not be resolved by UMA in either track. The autonomous run
found the dissociated endpoint drifting back toward recombination, and the
seeded run produced a persistent second imaginary mode between 8 and 11 meV
across every refinement attempt and both dispersion settings. A site placement
hypothesis was tested on 2026-09-14 and ruled out.

MACE resolves the same reaction cleanly at the same published geometry: a
confirmed first order saddle, confirmed connectivity, a full zero point
correction, and a final barrier of 0.870 eV against the 0.800 eV reference, an
error of +0.070 eV.

The reaction is therefore hard for UMA's potential energy surface specifically,
not intrinsically hard. Surfacing a statement that precise is what the two track
architecture is for.

## Fully validated barriers

Where the whole chain succeeded, meaning a confirmed first order saddle,
confirmed connectivity and an applied zero point correction, the final scored
barriers are:

| Reaction | Reference | UMA off | UMA on | MACE off | MACE on |
|---|---:|---:|---:|---:|---:|
| H2/Cu(111) | 0.630 | 0.657 | 0.264 | 0.616 | 0.401 |
| H2/Cu(100) | 0.740 | not confirmed | 0.246 | not confirmed | not confirmed |
| N2/Ru(0001) terrace | 1.840 | 1.090 | 0.535 | 1.034 | 0.614 |
| N2/Ru(0001) step | 0.400 | 0.490 | 0.233 | 0.238 | 0.070 |
| CH4/Ru(0001) | 0.800 | not confirmed | not confirmed | 0.870 | 0.480 |
| CH4/Ni(111) terrace | 1.010 | 1.111 | 0.413 | not confirmed | not confirmed |

H2/Cu(111) with UMA and dispersion off, at 0.657 eV against a 0.630 eV
reference, is the closest single number in the project.

## Caveat that applies to the headline table

These are classical barriers compared against a zero point corrected reference.
SBH10's tabulated values include a zero point correction of between 0.03 and
0.14 eV, always negative. Applying the paper's own Table 2 corrections to the
UMA D3 off column moves its MAE from 0.166 to 0.145 eV, closer still to BEEF
vdW. Those are the paper's zero point values rather than this project's
computed ones, so that number is indicative rather than a result. The full
zero point pipeline runs wherever a first order saddle is confirmed, and those
numbers appear in the validated table above.

## Reproducing

```bash
python build_seeds.py data_sbh10_si/poscars.txt --out work_seeds/
python scripts/run_seeded.py --seeds work_seeds/ --model uma-s-1p1  --d3 off
python scripts/run_seeded.py --seeds work_seeds/ --model uma-s-1p1  --d3 on
python scripts/run_seeded.py --seeds work_seeds/ --model mace-mh-1  --d3 off
python scripts/run_seeded.py --seeds work_seeds/ --model mace-mh-1  --d3 on
python plot_seeded_results.py
```

UMA and MACE need separate environments. See the README setup section.
