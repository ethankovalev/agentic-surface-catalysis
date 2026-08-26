
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
