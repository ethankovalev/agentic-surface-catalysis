"""
Test whether cross model disagreement predicts error without using the
reference value.

    python analyse_disagreement.py
    python analyse_disagreement.py --results /path/to/results/seeded

Runs anywhere. No pod, no calculator, no GPU, no API.

WHY THIS EXISTS
---------------
The Nature Catalysis roadmap (Xin, Kitchin, Lopez et al. 2026) asks for a
guardrail layer that blocks high uncertainty or out of distribution
proposals for human verification. Doing that requires a confidence signal
available at run time, when the reference value is by definition unknown.

Deep ensembles are the usual answer and are impractical for foundation
models, where a single pretrained checkpoint is all you have. This script
tests a cheaper substitute that falls out of the benchmark for free:
disagreement between two independently trained IN DOMAIN models.

WHAT IT FINDS, AND WHAT IT DOES NOT
------------------------------------
Positive correlation between in domain disagreement and error, and the
single worst reaction in the set is also the one where the two in domain
models disagree most. With ten reactions that is suggestive and nothing
more; the script prints the sample size and an approximate p value so the
claim cannot be quoted without its caveat.

The useful negative result is firmer: pooling all four models destroys
the signal. Disagreement from a model that was never trained on surfaces
is noise, not information, so an out of domain model must not be averaged
into a confidence estimate.
"""

import argparse
import json
import math
from pathlib import Path

REFERENCE = {
    "H2_Cu111": 0.630, "H2_Cu100": 0.740, "H2_Pt111": 0.000,
    "H2_Ru0001": 0.000, "N2_Ru0001_terrace": 1.840, "N2_Ru0001_step": 0.400,
    "CH4_Ru0001": 0.800, "CH4_Ni100": 0.760, "CH4_Ni111_terrace": 1.010,
    "CH4_Ni111_step": 0.800,
}

IN_DOMAIN = ["uma-s-1p1", "mace-mh-1"]
OUT_OF_DOMAIN = ["orb-v3-cons-inf-omat", "mace-mpa-0"]
KEY = "barrier_at_reference_geometry_eV"


def load(results_dir, model, d3="off"):
    path = Path(results_dir) / f"seeded_{model}_d3{d3}.json"
    if not path.exists():
        raise SystemExit(f"{path} not found. Use --results, or run that sweep.")
    raw = json.loads(path.read_text())
    return {name: float(v[KEY]) for name, v in raw.items()
            if isinstance(v.get(KEY), (int, float))}


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


def approx_p(r, n):
    """Two tailed p for a Pearson r, using the t distribution with n-2 df.

    A normal approximation is tempting and wrong here: at 8 degrees of
    freedom it returns 0.07 where the correct value is 0.11, which is the
    difference between a number someone might call marginally significant
    and one they would not. Integrated numerically to avoid a scipy
    dependency.
    """
    if n < 3 or abs(r) >= 1:
        return float("nan")
    df = n - 2
    t_obs = abs(r) * math.sqrt(df / (1 - r * r))

    log_c = (math.lgamma((df + 1) / 2) - math.lgamma(df / 2)
             - 0.5 * math.log(df * math.pi))
    def pdf(x):
        return math.exp(log_c - (df + 1) / 2 * math.log1p(x * x / df))

    # Simpson's rule over the tail, out to where the density is negligible
    hi, steps = t_obs + 60.0, 20000
    h = (hi - t_obs) / steps
    total = pdf(t_obs) + pdf(hi)
    for i in range(1, steps):
        total += pdf(t_obs + i * h) * (4 if i % 2 else 2)
    return 2 * total * h / 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results",
                    default="/workspace/agentic-surface-catalysis/results/seeded")
    ap.add_argument("--d3", default="off")
    args = ap.parse_args()

    models = {m: load(args.results, m, args.d3)
              for m in IN_DOMAIN + OUT_OF_DOMAIN}
    shared = [r for r in REFERENCE if all(r in v for v in models.values())]
    shared.sort(key=lambda r: abs(models[IN_DOMAIN[0]][r] - models[IN_DOMAIN[1]][r]))

    print(f"seeded track, dispersion {args.d3}, {len(shared)} reactions\n")
    print(f"{'reaction':22s} {'in domain gap':>13s} {'all four gap':>13s} "
          f"{'mean in dom err':>16s}")

    gap_in, gap_all, err = [], [], []
    for r in shared:
        g_in = abs(models[IN_DOMAIN[0]][r] - models[IN_DOMAIN[1]][r])
        vals = [models[m][r] for m in models]
        g_all = max(vals) - min(vals)
        e = sum(abs(models[m][r] - REFERENCE[r]) for m in IN_DOMAIN) / len(IN_DOMAIN)
        gap_in.append(g_in)
        gap_all.append(g_all)
        err.append(e)
        print(f"{r:22s} {g_in:13.3f} {g_all:13.3f} {e:16.3f}")

    n = len(shared)
    r_in = pearson(gap_in, err)
    r_all = pearson(gap_all, err)

    print(f"\nin domain disagreement vs error: r = {r_in:+.3f}  "
          f"(n = {n}, approx p = {approx_p(r_in, n):.2f})")
    print(f"all four disagreement vs error:  r = {r_all:+.3f}  "
          f"(n = {n}, approx p = {approx_p(r_all, n):.2f})")

    worst_err = max(range(n), key=lambda i: err[i])
    worst_gap = max(range(n), key=lambda i: gap_in[i])
    print(f"\nlargest error:            {shared[worst_err]}")
    print(f"largest in domain gap:    {shared[worst_gap]}")
    print("same reaction" if worst_err == worst_gap else "different reactions")

    print(f"\nRead this carefully. n = {n} is too small for the positive")
    print("correlation to be significant. It is a signal worth building an")
    print("escalation trigger around and testing properly, not a validated")
    print("detector. The firmer result is the sign flip: pooling out of")
    print("domain models destroys the signal, so they must not be averaged")
    print("into a confidence estimate.")


if __name__ == "__main__":
    main()
