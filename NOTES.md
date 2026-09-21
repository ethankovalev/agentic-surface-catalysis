
## N2_Ru0001_step, continued
- fmax was NEVER agent-settable at 0.02 by design - that was the agent choosing to
  tighten a default of 0.10, inconsistently, run to run. Pinned it deliberately today,
  first to 0.02 (mistake - too tight for this model/system, bands stall at ~0.037
  eV/A fmax for hundreds of steps with energy flat to 5 decimal places, reports
  DID NOT CONVERGE indefinitely), then corrected to 0.05.
- Across three runs with fmax=0.02 (n_images 12->16->16, steps up to 1000) the
  barrier stayed rock-solid at 1.956 eV, peak consistently ~59% along the path.
  That number looks real, just never got the "converged" stamp at too-tight a
  tolerance.
- One run at n_images=20 diverged onto a different path (energy dropped to
  -711.3 eV, well below the ~-710.04 the other runs settled at) - NOT YET
  RETESTED with fmax=0.05, worth checking whether that's a different mechanism
  or an artifact.
- NEXT: rerun N2_Ru0001_step with fmax=0.05 pinned, see if it reports converged
  and what barrier it gives. Then still need: CH4_Ni111_step (untested with any
  of today's fixes), scripts/run_grid.py (not started).

## MACE + Orb working combination (.venv-mace on /workspace)
- torch 2.6.0+cu124  (NOT 2.14/cu130 - driver is CUDA 12.4 and cannot run it;
  NOT 2.1.0 - mace-torch calls torch.compiler.is_compiling(), added after 2.1)
- numpy 1.26.4  (numpy 2.x breaks torch's numpy bridge: "Numpy is not available"
  when MACE actually computes, though imports look fine)
- e3nn 0.4.4, mace-torch 0.3.16, orb-models 0.5.5
- Install with: pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  The cu124 index tops out at 2.6.0, which is also the minimum orb-models needs.
- Verified by real calculation, not import: MACE H2O energy -14.048 eV,
  Orb built ConservativeForcefieldRegressor.
- pip warns matscipy wants numpy>=2 and orb wants torch>=2.6 - both are satisfied
  or harmless. Ignore.
- ALWAYS use `python3 -m pip`, never bare `pip` - bare pip resolves to system
  Python and silently installs to the wrong place.
- After cloning on a fresh pod, always: ln -sfn /workspace/agentic-surface-catalysis/data /root/agentic-surface-catalysis/data (checkpoints live on the volume, config.py resolves paths relative to the repo root on local disk)

## N2_Ru0001_step, still unresolved
Two refine_saddle calls, seeded from two different NEB attempts, converged
to two distinct genuine first-order saddles:
  - NEB run 1 peak: gas-referenced barrier -0.878 eV (1 imaginary mode, 16 meV)
  - NEB run 2 peak: gas-referenced barrier +0.536 eV (1 imaginary mode, 7 meV)
Both converged, both pass the one-imaginary-mode check. Neither is confirmed
to connect the actual initial/final states - the connectivity check
(check_saddle_connects) was designed but not yet deployed. Root cause is
almost certainly bad endpoints: NEB profile was incoherent (>2 eV swings
between adjacent images), meaning the final state likely isn't a true
minimum. Do not treat either number as validated. Reference: 0.40 eV.

## MACE invocation, verified against actual code
Override var: MLIP_MODEL (not MACE_MODEL or anything else), read in
config.py:94 as os.environ.get("MLIP_MODEL", "uma-s-1p1").
_build_mace uses mace_mp(model=checkpoint_path, default_dtype="float64",
dispersion=with_d3). Checkpoint path resolves via the data symlink already
in place: data/mace-mh-1/mace-mh-1.model.

First test command (day 3, not yet run):
  deactivate
  source /workspace/agentic-surface-catalysis/.venv-mace/bin/activate
  export MLIP_MODEL=mace-mh-1
  export MLIP_DEVICE=cuda
  export ANTHROPIC_API_KEY=<key>
  nohup python invoke.py --single H2_Cu111 > mace_test1.log 2>&1 &

UNTESTED. .venv-mace has no fairchem-core installed, so this is also an
implicit test of whether src/tools.py's imports crash without it - check
for module-level fairchem imports before running, same way MACE's lazy
import was confirmed today.
Confirmed: no module-level fairchem import in tools.py or calculators.py,
so .venv-mace running an H2_Cu111 reaction should not crash on missing
fairchem-core. Both backends' imports are lazy, inside their build
functions only. Green light to actually try the MACE test command above.

## MACE fails H2_Cu111 twice, cleanly, where UMA succeeds easily
Full validation chain ran correctly and refused two false positives:
  NEB 1: barrier 0.025 eV, saddle confirmed (1 imaginary mode, 7 meV),
         connectivity check FAILED - both IRC directions fall to initial state
  NEB 2: barrier 0.734 eV, saddle confirmed (1 imaginary mode, 87 meV),
         connectivity check FAILED - both IRC directions fall to final state
Two genuine saddles, neither connecting initial to final, barriers 30x apart.
Cap correctly reached after 2 attempts, reaction correctly reported unresolved.
UMA validated this same reaction cleanly on the first attempt (0.486 eV).
This is a real cross-model finding: MACE's NEB interpolation is struggling
on a system UMA handles without issue. Worth investigating whether this is
MACE-specific (float32 vs float64, dispersion handling, IDPP interpolation
sensitivity) or a genuine model-quality difference on this PES.

## Subsurface saddle found on H2_Cu111 (UMA)
refine_saddle returned a genuine first-order saddle (one imaginary mode,
32.6 meV) with one hydrogen at -1.69 A relative to the top metal layer,
i.e. inside the slab. It is a real stationary point for subsurface H
penetration, not for dissociation. check_saddle_connects correctly rejected
it (both IRC directions relaxed to a ~2.8 A basin, neither endpoint), and
this was reproducible across displacements from 0.05 to 0.35 A, so it is
not a displacement-magnitude artefact.
check_geometry now flags adsorbate atoms below MIN_ADSORBATE_HEIGHT (0.3 A)
in initial, final and saddle. closest_contact could not catch it: the buried
H was 1.68 A from its nearest Cu, an ordinary bond length.
Root cause is still upstream - the NEB did not converge and its peak image
was off the dissociation path, so refinement had a bad seed. Same signature
appeared on MACE/H2_Cu111 yesterday.

## H2_Cu111 fully validated - first since the wrap-bug chain of fixes
barrier_gas_eV = 0.449 eV (reference 0.63 eV, error -0.18 eV)
All ten checks pass, including path_resolved and saddle_connectivity for
the first time. Root causes fixed across two days: perpendicular molecule
orientation, periodic-wrap causing an 11.8 A phantom NEB path, endpoint
overshoot past the genuine adjacent-site product, and path_resolved not
deferring to a confirmed connected saddle. Error is consistent in sign and
magnitude with UMA's Cu(100) undershoot (-0.21 eV), supporting the
dispersion-overcorrection hypothesis rather than indicating a new problem.

## D3-off test on H2_Cu111: dispersion hypothesis confirmed, more nuanced than expected
D3 on:  barrier_gas_eV = 0.449 eV  (error -0.18 eV vs 0.63 eV reference)
D3 off: barrier_gas_eV = 0.764 eV  (error +0.13 eV vs 0.63 eV reference)
Both fully validated, all ten checks pass, saddle connectivity confirmed
both directions in both runs. D3 does not just shift the barrier by a
constant - it overcorrects past the true value in the other direction.
Well depth drops from 0.101 eV (D3 on) to 0.004 eV (D3 off), confirming D3
is responsible for the physisorption well, as expected physically.
Working picture: D3 stabilises the transition state (close to the metal)
more than the reactant, lowering the barrier below truth; without D3 the
base UMA functional has its own smaller positive bias. True answer likely
sits between the two settings. Worth repeating on Cu100 and Pt111 once
those validate cleanly, to see if the pattern (undershoot on, overshoot
off) holds across surfaces.

## H2_Cu100, seeded, D3-off: second-order stationary point (2026-09-12)
Sella converges cleanly (True) to a point with 2 imaginary modes: 128 meV
(dominant, likely the true reaction coordinate) and 23 meV (stable, well
above the 5 meV noise threshold - not numerical noise like the
CH4_Ru0001 case). Barrier at the UNMOVED published geometry is 0.760 eV
against a 0.74 eV reference (+0.02 eV) - excellent PES accuracy,
independent of refinement outcome. Hypothesis: Cu(100)'s hollow-hollow
TS sits on a 4-fold symmetry axis that UMA's PES treats as a ridge
between two lower-symmetry saddles, rather than the true minimum-energy
path DFT finds. Deferred: retry with a small (~0.05 A) symmetry-breaking
displacement off the axis before refine_saddle, to test whether this
resolves to a genuine nearby first-order saddle.

## Cell sensitivity test: H2_Cu111, D3-off (2026-09-12)
Scripted (agent-free) pipeline run at 3x3x4 (1/9 ML) and 2x2x6 (1/4 ML),
identical model/dispersion/thresholds otherwise. Both confirm genuine
first-order saddles with full mode-following connectivity.
  3x3x4: 0.744 eV (classical 0.764, dZPE -0.019), vs 0.63 ref: +0.114
  2x2x6: 0.711 eV (classical 0.737, dZPE -0.027), vs 0.63 ref: +0.081
Gap: 0.034 eV, under the 0.05 eV threshold. Cell size is NOT the source
of the seeded-vs-autonomous discrepancy on this reaction. Decision:
autonomous pipeline stays at 3x3x4; no migration. The larger remaining
gap between the seeded track (0.657 eV, published BEEF-vdW geometry) and
this scripted 3x3x4 result (0.744 eV) is attributable to geometry
source/relaxation freedom, not cell size.

## Seeded track complete, all 10 reactions, both D3 settings (2026-09-12)
barrier_at_reference_geometry_eV needs no search/saddle/ZPE - the
headline number. D3-off MAE 0.166 eV, D3-on MAE 0.445 eV, D3-off wins
9/10 reactions. For reference BEEF-vdW (best DFT functional in the
original paper) reports MAE ~0.12-0.14 eV - UMA D3-off is within
striking distance of the reference method's own accuracy, at zero
fitting cost, with zero missing data points.

Full refinement (saddle + ZPE) succeeded on 6/10: H2_Cu111, H2_Cu100
(D3-on only), CH4_Ni111_terrace, N2_Ru0001_terrace, N2_Ru0001_step.
Four refinement non-successes, all correctly caught by the exit gate,
not pipeline bugs:
  - H2_Pt111, H2_Ru0001: both near-barrierless (ref ~0 eV) - connectivity
    test's fixed 60-step relaxation can stall on the flat PES near a
    near-zero barrier. Algorithmic limit of the connectivity check for
    this specific regime, not a defect in UMA or the saddle itself.
  - CH4_Ni100: saddle relaxed to bond length 1.124 A, near-intact
    (normal C-H 1.07 A) - collapsed back toward reactant, correctly
    rejected.
  - CH4_Ni111_step: saddle relaxed to 2.268 A, sits in the product
    basin (displacement further apart barely moves it) - correctly
    rejected.

Full table: results/seeded/SEEDED_SUMMARY.md
Details: results/seeded/seeded_uma-s-1p1_d3{on,off}.json

## fcc/hcp site probe on CH4_Ru0001: negative result (2026-09-14)
Two independent runs both relax hcp and fcc starting endpoints to the
same energy to within ~1 ueV - noise level, not a real difference.
compare_site_geometries.py confirms/refutes whether they reach the same
basin. Conclusion: the fcc/hcp site mismatch found yesterday (code
placed both fragments on hcp against SBH10's FCC,FCC) is real and now
fixed, but is likely NOT the cause of CH4_Ru0001's endpoint instability
- UMA's own PES does not meaningfully distinguish these two sites for
this fragment. The instability's real cause remains open.

Update: compare_site_geometries.py confirms SAME GEOMETRY - both
relaxations reached identical anchor positions (zero lateral delta on
both C and H). The negative result above is confirmed, not just
hedged: fcc and hcp are not separate basins for this fragment on this
PES.

## CH4_Ni111_step autonomous D3-on: computed but not validated (2026-09-14)
Agent report claimed a resolved 0.488 eV (ZPE-corrected) barrier, but
exit_gate correctly refused - check_saddle_connects returned Delta=0.02 A
between displaced-and-relaxed endpoints, too small to distinguish
forward from backward. This cascades: check_convergence and
check_path_resolved both defer to a saddle only when it is BOTH
first-order AND connected, so both stayed failed despite a genuine
1-imaginary-mode (60 meV) saddle being found on the second refine_saddle
attempt. Classical barrier 0.675 eV, dZPE -0.187 eV, reaction energy
+0.484 eV (barrier > reaction energy, physically consistent).

Candidate cause: check_saddle_connects's displacement magnitude may be
too small for this specific C-H bond / step-edge geometry - both push
directions relax back close to the saddle within the check's step
budget rather than reaching clearly separated basins. Not fixed today;
candidate follow-up, not urgent.

Stored as computed_eV=0.4877848102524832, validated=False. Same
treatment as CH4_Ru0001: report the diagnosed cause, do not treat the
number as trustworthy.

## MACE-mh-1 seeded track, D3-off, first cross-model result (2026-09-15)
MAE at reference geometry (10/10, no search needed): MACE 0.174 eV vs
UMA 0.166 eV - nearly tied, both close to BEEF-vdW's 0.14 eV. Supports
the D3-overcorrection finding generalizing beyond UMA specifically.

STANDOUT: CH4_Ru0001 - unresolved in UMA autonomous (endpoint
instability) AND UMA seeded (persistent 8-11 meV second imaginary mode,
multiple attempts) AND ruled out fcc/hcp as the cause yesterday -
resolves CLEANLY on MACE: confirmed first-order saddle, confirmed
connectivity, barrier 0.870 eV vs 0.800 eV reference (+0.070 eV). This
reaction is hard for UMA's PES specifically, not intrinsically hard.

Shared weakness: N2_Ru0001_terrace is the worst reaction for BOTH
models (UMA -0.501, MACE -0.727 at ref geo) - also the highest-barrier
reaction and the one with the widest experimental spread (1.3-2.27 eV)
in the literature, so this may reflect reference uncertainty as much as
model error.

MACE also confirmed H2_Cu111 (0.616 vs 0.630, -0.014) and
N2_Ru0001_step (0.238 vs 0.400, -0.162). Four of ten fully validated;
remainder show the same pattern of failure modes as UMA (saddle found
but ZPE-gated, or no confirmed saddle) - not yet individually diagnosed.

## MACE-mpa-0 seeded track, D3-off: rules out precision as Orb's cause (2026-09-15)
mace-mpa-0 uses the same MACE backend as mace-mh-1, which _build_mace
forces to float64 unconditionally regardless of checkpoint - no
precision confound. Still 0/10 confirmed first-order saddles,
identical to Orb's float32 result, and WORSE in magnitude: MAE 0.769 eV
vs Orb's 0.473 eV. Worst case CH4_Ni111_step at ref geometry: -1.432 eV
against a 0.800 eV reference, a physically nonsensical energy over
2 eV off.

This controls for precision cleanly: same numerics as mace-mh-1
(in-domain, MAE 0.174, 4/10 confirmed), different training domain
(bulk crystals only), and the result is catastrophic rather than
merely worse. Confirms the domain-mismatch finding is real, not a
float32 artifact from Orb specifically.

Four-model MAE at reference geometry:
  UMA (in-domain):        0.166
  MACE-mh-1 (in-domain):  0.174
  Orb (out-of-domain):    0.473
  MACE-mpa-0 (out-of-domain): 0.769

Saddle confirmation: in-domain models 4-6/10, out-of-domain models
0/10 for both, regardless of backend or precision.

## refine_saddle_robust resolves H2_Cu100 (2026-09-19)
The ~23 meV second imaginary mode on H2_Cu100, stable across both
dispersion settings and every previous refine_saddle attempt, is a
symmetry artifact as the README hypothesised. Displacing along the
second imaginary mode and re-refining breaks the ridge.

0.1 A displacement: 3 attempts, still 2 imaginary modes (23 -> 19 meV).
Too small to leave the basin.
0.3 A displacement: 2 attempts, 1 imaginary mode at 125 meV.
Barrier 0.743 eV against a 0.740 eV reference, error +0.003 eV - the
closest single number in the project, previous best being H2_Cu111 at
+0.027 eV.

H2_Cu100 moves from "gate correctly refused" to validated. The saddle
was always there; the original single-attempt refine_saddle had no way
to reach it.

Untested: whether 0.3 A also resolves CH4_Ru0001's ~10 meV second mode,
which is the other documented ridge case.

## refine_saddle_robust resolves CH4_Ru0001 (2026-09-19)
The reaction that resisted every previous approach - unresolved in both
tracks, fcc/hcp site hypothesis tested and ruled out - gives a confirmed
first-order saddle under ridge-descent recovery. 2 attempts, single mode
at 123 meV.

Barrier 1.077 eV against a 0.800 reference, error +0.277 eV. A real
saddle, NOT an accurate one. Compare H2_Cu100 at +0.003 eV. The recovery
locates the saddle; it does not fix the surface.

Displacement threshold, two reactions:
  H2_Cu100   (2nd mode 23 meV): 0.1 A failed, 0.3 A resolved
  CH4_Ru0001 (2nd mode ~10 meV): 0.3 A -> 6.0 meV, still 2 modes;
                                 0.6 A resolved
A stronger second mode needed a SMALLER displacement. Hypothesis, from
n=2 only: a deeper ridge sits in a narrower basin, a shallow one on a
broader flat region needing a bigger push. Not established.

0.6 A is now the committed literal. Worth rechecking H2_Cu100 at 0.6 to
confirm the larger value does not break the case that already worked.

Confirmed: H2_Cu100 re-run at 0.6 A still resolves, 2 attempts, barrier
0.744 eV (error +0.004 against +0.003 at 0.3 A). The 1 meV difference is
optimiser path noise. 0.6 A is therefore safe as a single constant for
both documented ridge cases; no adaptive displacement needed.

## CH4_Ni100: the 0.629 eV saddle is for a different process (2026-09-19)
refine_saddle_robust finds a first-order saddle at 2.387 A with one
imaginary mode at 40.6 meV, but connectivity_by_bond_displacement
returns connects=False: pushing the bond 0.35 A in BOTH directions
relaxes back to ~2.39 A. It is a minimum along the C-H coordinate, so
the 40.6 meV mode points along some other motion entirely.

0.629 eV is therefore not the dissociation barrier. The earlier
"collapse to 1.125 A" reading was wrong - the bond stretched, not
collapsed. Same conclusion, different cause.

Open: refine_saddle_robust reports first_order_saddle=True here because
it gates on mode count alone. Connectivity is computed in the probe but
not used in the accept decision.

## CH4_Ni100: no connecting saddle exists on UMA's surface (2026-09-19)
Four strategies tried, all fail: default Sella, smaller trust radius,
ridge displacement, and bond-constrained refinement. Every attempt that
found a single imaginary mode failed connectivity (IRC and bond
displacement agree). Drift never fell below +0.35 A even with the bond
explicitly pinned.

Best available number remains the unrefined single point at the
published BEEF-vdW geometry: 0.900 eV against 0.760, error +0.140 -
more accurate than any refined result (0.629, error -0.131).

Conclusion: UMA has no first-order saddle near the published TS that
connects reactant to product for this reaction. A statement about the
model, established by the verification layer rather than assumed.

Note: the failure message wrongly attributed attempt 3 to ridge
displacement when it used constraints. Cosmetic, worth fixing.

## Full robust-saddle results, UMA D3-off seeded (2026-09-19)
Connectivity now enforced in the accept decision (IRC primary, bond
displacement fallback). Four previously-refused reactions retested.

RESOLVED, connectivity confirmed:
  H2_Cu100    0.744 eV vs 0.740, error +0.004. Ridge recovery, 2
              attempts. Compressed 0.749 A (intact), stretched 2.584 A.
  CH4_Ru0001  1.077 eV vs 0.800, error +0.277. Ridge recovery, 2
              attempts. Compressed 1.101 A, stretched 2.700 A. The
              earlier result predates enforced connectivity but HOLDS
              UP under it: a genuine saddle UMA places 0.277 eV high.

NOT RESOLVED, and for different reasons:
  CH4_Ni100       Four strategies fail. Every single-mode result fails
                  connectivity. No connecting saddle near the published
                  TS on UMA's surface.
  CH4_Ni111_step  Drift guard rejected at +0.507 A against a 0.5 limit,
                  7 mA over an arbitrary constant. Raised the limit to
                  0.7 as a diagnostic: IRC then rejected it
                  independently, both ends at 2.61 and 2.75 A, i.e.
                  descending to product in BOTH directions. Two
                  criteria agree; the rejection is real. Limit restored.
                  The constraint strategy did hold the bond (-0.097 A
                  drift, attempt 2), so the mechanism works even though
                  this reaction has no saddle to find.

Open: both resolved reactions needed 0.1 A ridge displacement, the
value that FAILED for them in earlier runs at 0.3 and 0.6 A. The
earlier two-point displacement pattern does not survive the other
changes and should not be reported as a finding.

## Correction: the 0.1 A displacement "inconsistency" was a stale label (2026-09-21)
The strategy text in refine_saddle_robust was a hard-coded string reading
"retry displaced 0.1 A". The sed edits changed the value actually used
(0.1 -> 0.3 -> 0.6) but never the label. Every recent run used 0.6 A and
printed 0.1. There was no inconsistency: H2_Cu100 and CH4_Ru0001 resolved
at 0.6 A, as the earlier runs showed. The open question logged on
2026-09-19 is withdrawn.
