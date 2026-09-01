# CLAUDE.md

## Project Identity & Goal
- **Domain:** MLIP barrier benchmarking (MACE, Orb, UMA) against the SBH10 experimental dataset.
- **Goal:** Compute gas-referenced dissociation barriers on transition metal surfaces to evaluate chemical accuracy (0.043 eV).
- **Core Rule:** Blind evaluation. Never access `src/benchmark.py` or reveal experimental reference values to agents.

## Architecture & Logic
- **Agents:** `Structure_Agent`, `Simulation_Agent`, `Validation_Agent`.
- **Exit Gate:** `exit_gate` in `src/graph.py` is an unbypassable code gate enforcing 10 physical checks.
- **Environments:** Two isolated venvs (`.venv` for UMA/FAIRchem, `.venv-mace` for MACE/Orb) due to `e3nn` version pin conflicts.

## Physical & Mathematical Rules
- **Gas Referencing:** `well_depth = E_gasref - E_initial`; `barrier_gas = barrier_neb - well_depth`.
  *(Subtract physisorption well depth; positive well depth reduces the gas-phase barrier)*.
- **NEB:** Pinned force tolerance = 0.05 eV/Å. Barrier extraction MUST use `fit=False` (`NEBTools.get_barrier(fit=False)`).
- **PBC:** Always enforce `slab.pbc = True` on built slabs (Orb requires full 3D periodicity).
- **Dispersion (D3):** Per-model configuration via `torch_dftd`, internal flags, or `none`. Never hardcode global D3 defaults.

## Bug Prevention & Known Edge Cases
- **Threading:** Warm `torch.det` on the main thread before starting LangGraph execution to prevent LAPACK lazy wrapper crashes.
- **MACE Head:** Use `oc20_usemppbe` for surface tasks on `mace-mh-1`. Never use `"oc20"`.
- **LangGraph API:** Use `prompt=...` (not `state_modifier=...`). Handle `stream_mode="values"` and check for `'messages'` key in state updates.
- **Site Type:** Always include `site_type` in prompt specifications for step vs terrace reactions.

## Common Commands
- Single reaction: `python invoke.py --single H2_Cu111`
- Grid sweep: `python scripts/run_grid.py --models uma-s-1p1 --d3 on`
- AST Syntax Check: `python3 -c "import ast; ast.parse(open('src/tools.py').read()); print('Syntax OK')"`
