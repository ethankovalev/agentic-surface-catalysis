# Agentic surface catalysis

An agentic pipeline computing dissociation barriers on transition metal
surfaces, benchmarked blind against SBH10 experimental references.

## Invariants — do not break these

- **The agent must never see a reference barrier.** No tool may reach
  `src/benchmark.py`; no prompt may state a target value or tolerance.
  Scoring happens in the runner after the graph returns. Site type is the one
  deliberate exception: it is part of the question, not the answer.
- **`NEB_FMAX` is pinned at 0.05 eV/A and is not a tool argument.** It was
  agent-settable with a default of 0.10, so runs converged to different bars
  while all reporting success. Same for `SADDLE_FMAX`.
- **`FORCE_MODEL` and `FORCE_D3` override whatever the agent chooses.**
  `config.DEFAULT_MODEL` is read once at import, so setting `MLIP_MODEL`
  mid-process does nothing.
- **Grid results go to `/workspace/...`, not the repo.** The repo is on
  ephemeral local disk. A full sweep was lost that way.

## The failure mode this project exists to catch

Every bug found so far produced a plausible number and raised no exception:

- `site_type` never reached the agent, so step and terrace reactions sent
  identical prompts while scored against references 1.44 eV apart.
- Gas referencing added the physisorption well depth instead of subtracting
  it. Invisible on H2/Cu, 3x wrong on N2/Ru.
- `NEBTools.get_barrier()` defaults to a spline fit that overshot the highest
  computed image by 0.3 eV. Now `fit=False`.
- MACE fell back to a bulk-crystal head with only a warning.
- Dissociated fragments placed by arithmetic, not at binding sites.
- Fragments then snapped to neighbouring sites and recombined.

Assume a new number is wrong until a check says otherwise.

## Workflow — cheapest test first

1. `python scripts/check_structures.py` — all ten reactions, geometry only,
   no GPU, seconds. Every bug so far was a structure bug. Run this first.
2. `python -m pytest tests/ -v` — EMT-based, no GPU, ~5 seconds.
3. `python invoke.py --single <reaction>` — one reaction on GPU, minutes to
   an hour. Only after 1 and 2 pass.
4. `python scripts/run_grid.py --models uma-s-1p1 --d3 on` — full sweep,
   hours. Always under `nohup`.

Never debug on the GPU what can be caught on CPU.

## Environment

- Pod is RunPod, RTX 4090, EU-RO-1, created via `runpodctl` not the web UI.
- Network volume `tminb1ipn7` holds checkpoints and results; local disk does
  not survive pod loss.
- `UMA_CHECKPOINT`, `MLIP_DEVICE`, `ANTHROPIC_API_KEY` must be exported every
  session. They do not persist.
- MACE and UMA **cannot share a virtualenv**: `mace-torch` pins `e3nn==0.4.4`,
  `fairchem-core` needs `>=0.5`. Installing MACE into `.venv` breaks UMA.

## Style

- Keep code simple. Dense array and index manipulation is hard to review.
- Say directly when an approach will not reach the goal. Do not humour it.
- Documentation states what is not finished. Never overclaim a result.
- Guard every patch: match an anchor exactly once, or change nothing.
CLAUDEEOF
git add CLAUDE.md && git commit -m "Add CLAUDE.md: project invariants, failure modes and workflow" && git push
