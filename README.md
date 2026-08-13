# Agentic SBH10 benchmark

An agent that autonomously builds, relaxes, and computes dissociation
barriers on transition metal surfaces, then benchmarks itself against
the SBH10 experimental reference set.

## Why this benchmark

SBH10 (Sharada, Bligaard, Luntz, Kroes, Norskov; *J. Phys. Chem. C*
2017, 121, 19807) is ten dissociation barriers on transition metal
surfaces, referenced to molecular beam scattering, laser-assisted
associative desorption, and thermal experiments.

Two properties make it the right target.

**The references are experimental, not computational.** Agreement means
agreement with reality, not with another calculation.

**Dispersion is the discriminating axis.** In the original study the
dispersion-corrected BEEF-vdW reached 0.14 eV mean error, beating both a
meta-GGA and a screened hybrid functional - the reverse of the typical
gas-phase pattern. The model used here (UMA, `oc20` task) is trained on
RPBE, which has no dispersion term at all. That is a sharp, testable
hypothesis, not a vague gap.

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

export ANTHROPIC_API_KEY=...
export UMA_CHECKPOINT=/path/to/uma-s-1p2.pt   # or place it in data/
export UMA_DEVICE=cuda                        # if you have a GPU
```

The UMA checkpoint is gated: request access at
[huggingface.co/facebook/UMA](https://huggingface.co/facebook/UMA).

## Running

```bash
python invoke.py --single H2_Cu111     # one reaction, verbose - start here
python invoke.py --all                 # the full SBH10 set
python invoke.py --task "..."          # ad-hoc task, not scored
```

Start with `--single`. If the agent gets H2 on Cu(111) badly wrong,
that is worth knowing on day one, not after ten runs.

## Architecture

A supervisor picks which worker acts next; workers report back to it.

- **Structure_Agent** - builds the slab, places the adsorbate,
  constructs the dissociated endpoint
- **Simulation_Agent** - relaxes both endpoints, runs the
  climbing-image NEB
- **Validation_Agent** - runs the physical sanity checks

**One deliberate departure from the standard supervisor pattern: the
supervisor cannot end the run.** `exit_gate` in `src/graph.py` reads the
validation record directly and refuses to exit until every check has
actually run and passed. A model that talks itself into "looks good to
me" is exactly the failure this project exists to catch, so the
guarantee lives in code, not in a prompt.

### The checks

| Check | Catches |
|---|---|
| `convergence` | energies from non-converged optimisations |
| `noise_floor` | results below the model's own ~0.009 eV MAE |
| `dispersion` | adsorbate drifting away because RPBE has no vdW term |
| `geometry` | atoms driven into the surface; barrier peak at an endpoint |
| `magnitude` | barriers outside a physically plausible range |

### Blind evaluation

The agent never sees the reference barrier. No tool reaches
`src/benchmark.py`, and none of the prompts mentions it. Comparison
happens in the runner, after the graph has returned. Any other
arrangement measures how well the agent can curve-fit, not how well it
can calculate.

## Project layout

```
sbh10-agent/
├── config.py            settings: paths, thresholds, model name
├── invoke.py            entry point / CLI
├── requirements.txt
├── data/                put uma-s-1p2.pt here
└── src/
    ├── store.py         run-scoped results store
    ├── calculators.py   UMA + D3 construction
    ├── prompt.py        the three worker system prompts
    ├── agent.py         create_agent helper
    ├── tools.py         all structure/simulation/validation tools
    ├── graph.py         supervisor graph + exit gate
    └── benchmark.py     SBH10 reference table + blind runner
```

## What is not finished

- **`SBH10` reference values are `None`.** Fill them in from Table 1 of
  the paper (`src/benchmark.py`). Nothing can be scored until this is
  done - this is the first thing to fix.
- **Only two of ten reactions are stubbed.** Add the remaining eight
  from the paper.
- **Site placement is approximate.** `place_adsorbate` picks ontop,
  bridge, or hollow from the first few top-layer atoms rather than doing
  a proper geometric site search.
- **Dissociation endpoints are heuristic.** The longest internal bond is
  split and the fragments slid apart. Works for diatomics; will need
  attention for anything larger.
- **No site or orientation sampling.** One configuration per reaction,
  so the reported barrier is not a minimum over configuration space.

## Known limitations of the underlying model

- OC20/RPBE has no dispersion term. This is why D3 is included in the
  relaxation loop by default (`with_d3=True`), not applied afterward as
  a single-point correction - the geometry shift under dispersion is
  most of the effect.
- D3 is known to overbind on some metal surfaces. Agreement with
  experiment should be read as suggestive, not confirmatory, without a
  second independent check.
- UMA's benchmarked MAE against its own reference DFT is roughly
  0.009 eV. Any computed quantity below that is not resolvable from
  zero, regardless of how tightly the optimiser converges.

## Reference

Sharada, S. M.; Bligaard, T.; Luntz, A. C.; Kroes, G.-J.; Norskov, J. K.
*SBH10: A Benchmark Database of Barrier Heights on Transition Metal
Surfaces.* J. Phys. Chem. C 2017, 121 (36), 19807-19815.
