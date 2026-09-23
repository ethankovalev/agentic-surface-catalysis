"""
A first, honestly-scoped escalation rule: given how much two in-domain
models disagree on a reaction, should the result be accepted or flagged
for human or DFT review?

    python escalate.py

Uses no GPU, no API. Reads only the ten (disagreement, error) pairs
already produced by analyse_disagreement.py on the real pod data.

WHY THIS IS HARDER THAN IT LOOKS
---------------------------------
Finding 3 (README) is r = +0.54 at n = 10, p = 0.11: not significant.
If a threshold is chosen by eye to look good on these same ten
reactions, any report of "it works" is circular - the rule would be
validated against the exact data that produced it.

The honest test is leave-one-out: for each reaction, choose the
threshold using only the OTHER nine, then check whether that
threshold would have flagged the held-out one correctly. This is the
smallest amount of out-of-sample testing possible, and at n = 10 it is
still a small amount - the result below should be read as "does this
survive the weakest possible over-fitting check", not as validation.

THE RULE ITSELF
-----------------
Deliberately the simplest rule that could work: escalate a reaction if
its in-domain disagreement is above the median disagreement of the
other nine reactions. A median split needs no tuned magic number, and
a tuned magic number chosen to fit ten points is the surest way to
overfit ten points.
"""

# the ten real (in_domain_gap_eV, mean_error_eV) pairs, from
# analyse_disagreement.py run against real pod data
DATA = {
    "CH4_Ru0001":          (0.040, 0.277),
    "H2_Cu111":            (0.049, 0.059),
    "CH4_Ni111_step":      (0.100, 0.123),
    "H2_Pt111":            (0.105, 0.074),
    "CH4_Ni111_terrace":   (0.119, 0.115),
    "H2_Cu100":            (0.120, 0.060),
    "H2_Ru0001":           (0.130, 0.230),
    "N2_Ru0001_step":      (0.137, 0.068),
    "CH4_Ni100":           (0.159, 0.079),
    "N2_Ru0001_terrace":   (0.226, 0.614),
}


def median(values):
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2


def leave_one_out():
    names = list(DATA.keys())
    results = []
    for held_out in names:
        others_gap = [DATA[n][0] for n in names if n != held_out]
        others_err = [DATA[n][1] for n in names if n != held_out]

        gap_threshold = median(others_gap)
        # what "high error" means is ALSO computed only from the other
        # nine, so the definition of a reaction worth flagging is not
        # informed by the held-out reaction's own error either
        err_threshold = median(others_err)

        held_gap, held_err = DATA[held_out]
        would_escalate = held_gap > gap_threshold
        was_high_error = held_err > err_threshold

        results.append({
            "reaction": held_out,
            "gap": held_gap,
            "err": held_err,
            "gap_threshold_from_others": gap_threshold,
            "escalated": would_escalate,
            "was_actually_high_error": was_high_error,
            "correct": would_escalate == was_high_error,
        })
    return results


def main():
    results = leave_one_out()

    print(f"{'reaction':22s} {'gap':>7s} {'err':>7s} {'threshold':>10s} "
          f"{'escalated':>10s} {'high err?':>10s} {'correct?':>9s}")
    for r in results:
        print(f"{r['reaction']:22s} {r['gap']:7.3f} {r['err']:7.3f} "
              f"{r['gap_threshold_from_others']:10.3f} "
              f"{str(r['escalated']):>10s} {str(r['was_actually_high_error']):>10s} "
              f"{str(r['correct']):>9s}")

    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    escalated = [r for r in results if r["escalated"]]
    accepted = [r for r in results if not r["escalated"]]

    print(f"\nleave-one-out agreement: {correct}/{n}")
    if escalated:
        print(f"mean error when escalated: "
              f"{sum(r['err'] for r in escalated)/len(escalated):.3f} eV "
              f"({len(escalated)} reactions)")
    if accepted:
        print(f"mean error when accepted:  "
              f"{sum(r['err'] for r in accepted)/len(accepted):.3f} eV "
              f"({len(accepted)} reactions)")

    print("\nRead this carefully. n = 10 means each leave-one-out fold")
    print("drops only one point from a nine-point threshold - a weak")
    print("over-fitting check, not real validation. This tells you the")
    print("rule is not trivially self-fitting, nothing stronger. It is")
    print("an engineering heuristic worth trying, not a validated")
    print("detector, and should not be described as one anywhere this")
    print("goes next: the README, an email, or a future paper.")


if __name__ == "__main__":
    main()
