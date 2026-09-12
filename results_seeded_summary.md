# Seeded track: UMA barrier at the published SBH10 transition state

**provenance: seeded.** PES accuracy at a known saddle, computed with
zero geometry search - the literature BEEF-vdW transition state is
read directly and evaluated. This measures whether the potential
energy surface is right, not whether the engine can find a
transition state on its own. See scripts/run_seeded.py.

| reaction | ref (eV) | D3-off | err | D3-on | err |
|---|---:|---:|---:|---:|---:|
| CH4_Ni100 | 0.760 | 0.900 | +0.140 | 0.215 | -0.545 |
| CH4_Ni111_step | 0.800 | 0.727 | -0.073 | 0.177 | -0.623 |
| CH4_Ni111_terrace | 1.010 | 1.184 | +0.174 | 0.464 | -0.546 |
| CH4_Ru0001 | 0.800 | 1.097 | +0.297 | 0.567 | -0.233 |
| H2_Cu100 | 0.740 | 0.760 | +0.020 | 0.369 | -0.371 |
| H2_Cu111 | 0.630 | 0.713 | +0.083 | 0.381 | -0.249 |
| H2_Pt111 | 0.000 | -0.022 | -0.022 | -0.302 | -0.302 |
| H2_Ru0001 | 0.000 | -0.295 | -0.295 | -0.549 | -0.549 |
| N2_Ru0001_step | 0.400 | 0.453 | +0.053 | 0.305 | -0.095 |
| N2_Ru0001_terrace | 1.840 | 1.339 | -0.501 | 0.903 | -0.937 |

**MAE, D3-off: 0.166 eV** (10/10 reactions)
**MAE, D3-on: 0.445 eV** (10/10 reactions)

For reference, BEEF-vdW (the best DFT functional in the original
SBH10 paper) reports MAE ~0.12-0.14 eV against the same references.
