
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
