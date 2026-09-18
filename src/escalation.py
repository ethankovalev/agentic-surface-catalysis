"""
Escalation logic: turn what the pipeline knows at run time into a
recommendation about whether a barrier can be trusted.

    from escalation import assess
    verdict = assess(store_record)

No GPU, no API, no calculator. Pure inspection of a finished run.

DESIGN PRINCIPLE
----------------
This does NOT decide for the user. It reports a level and the reasons
behind it, because a binary accept/escalate gate on the disagreement
signal was tested (escalate.py) and failed: 4/10 under leave-one-out,
worse than chance at n=10. A rule that cannot survive that test has no
business silently discarding or blessing a result.

So the hard blocks below come only from PHYSICS checks, which are
deterministic and do not need a fitted threshold. The disagreement
signal is reported as information, never as a block.

THE FOUR LEVELS
---------------
BLOCKED   a physics check failed. Not a barrier. No number reported.
REVIEW    physics passed, but something measurable looks unusual.
          A human should look before this is used.
CAUTION   physics passed, one soft signal is mildly off.
ACCEPTED  physics passed and nothing measurable looks unusual.

ACCEPTED is not a correctness claim. It means nothing this pipeline
can check has flagged, which is a weaker and more honest statement.
"""

# --- thresholds, each labelled with where it actually came from -------

# Deterministic, from the physics. Not fitted to any dataset.
PHYSICS = {
    "required_imaginary_modes_at_saddle": 1,
    "required_imaginary_modes_at_endpoint": 0,
}

# Measured on this project's own data. Honest provenance in the comment.
MEASURED = {
    # zpe.py: modes below this are numerical noise, not real curvature
    "imaginary_mode_noise_meV": 5.0,
    # UMA's benchmarked MAE against its own reference DFT. Quantities
    # below this are not resolvable from zero for that model.
    "model_resolution_eV": 0.05,
}

# NOT validated. Chosen as round numbers near the observed spread, and
# used only to decide whether to MENTION something, never to block.
# See NOTES.md 2026-09-18: the binary form of this failed leave-one-out.
ADVISORY = {
    # in-domain disagreements in the seeded track ran 0.04 to 0.23 eV
    "disagreement_notable_eV": 0.15,
    # chemical accuracy, for context on whether a gap matters at all
    "chemical_accuracy_eV": 0.043,
}


def assess(run, disagreement_eV=None):
    """Assess one finished run.

    run: the store record for a single reaction, as written by the
        pipeline. Missing keys are treated as "not checked", which is
        itself a reason for REVIEW rather than something to ignore.
    disagreement_eV: absolute barrier difference against another
        IN-DOMAIN model on the same reaction, if one has been run.
        Out-of-domain models must not be passed here: pooling them
        flipped the correlation sign (README, Finding 3).

    Returns a dict with level, blocking reasons, review reasons and
    notes. Never raises on a malformed record: an unreadable run is
    reported as BLOCKED, not crashed on.
    """
    blocking, review, notes = [], [], []

    # --- hard physics blocks -----------------------------------------
    saddle = run.get("saddle") or {}
    if not saddle.get("first_order_saddle"):
        n_imag = saddle.get("imaginary_modes_meV")
        blocking.append(
            f"no confirmed first order saddle (imaginary modes: {n_imag})")

    connectivity = run.get("saddle_connectivity") or {}
    if connectivity.get("connects") is not True:
        blocking.append(
            "saddle not confirmed to connect reactant and product")

    endpoints = run.get("endpoint_modes") or {}
    for which in ("initial", "final"):
        ep = endpoints.get(which) or {}
        if ep.get("is_minimum") is not True:
            blocking.append(f"{which} endpoint is not a genuine minimum")

    zpe = run.get("zpe") or {}
    if not zpe.get("ok"):
        blocking.append("zero point correction did not complete")

    barrier = run.get("barrier_zpe_eV")
    if barrier is None:
        blocking.append("no zero point corrected barrier was produced")
    elif not isinstance(barrier, (int, float)) or isinstance(barrier, bool):
        # A corrupted record must be reported, not crashed on: this
        # function runs over whole sweeps unattended. bool is excluded
        # explicitly because in Python it is a subclass of int, and
        # True would otherwise sail through as the number 1.
        blocking.append(
            f"barrier is not a number ({barrier!r}), the record is corrupt")
        barrier = None

    # --- soft signals, reported not enforced -------------------------
    if saddle.get("first_order_saddle"):
        modes = saddle.get("imaginary_modes_meV") or []
        # a single imaginary mode barely above the noise floor is a weak
        # reaction coordinate: technically a saddle, but a shallow one
        if len(modes) == 1 and modes[0] < 2 * MEASURED["imaginary_mode_noise_meV"]:
            review.append(
                f"reaction coordinate mode is only {modes[0]:.1f} meV, "
                f"close to the {MEASURED['imaginary_mode_noise_meV']} meV "
                "noise floor")

    if barrier is not None and abs(barrier) < MEASURED["model_resolution_eV"]:
        review.append(
            f"barrier {barrier:.3f} eV is below the model's own "
            f"{MEASURED['model_resolution_eV']} eV resolution, so it is "
            "not resolvable from zero")

    reaction_energy = run.get("neb", {}).get("reaction_energy_eV")
    if barrier is not None and reaction_energy is not None:
        if reaction_energy > 0 and barrier < reaction_energy:
            review.append(
                f"barrier {barrier:.3f} eV is below the reaction energy "
                f"{reaction_energy:.3f} eV for an endothermic reaction, "
                "which is not physically possible")

    drift = run.get("r_b_drift_A")
    if drift is not None and abs(drift) > 0.5:
        review.append(
            f"breaking bond drifted {drift:+.3f} A during refinement, so "
            "the saddle may sit in a different basin from the seed")

    # --- the disagreement signal, advisory only ----------------------
    if disagreement_eV is None:
        notes.append(
            "no second in-domain model has been run, so no cross model "
            "disagreement signal is available")
    else:
        if disagreement_eV > ADVISORY["disagreement_notable_eV"]:
            review.append(
                f"in-domain models disagree by {disagreement_eV:.3f} eV, "
                f"above the {ADVISORY['disagreement_notable_eV']} eV seen "
                "across most of the seeded track. NOT a validated "
                "threshold: see NOTES.md 2026-09-18")
        else:
            notes.append(
                f"in-domain models agree to {disagreement_eV:.3f} eV")
        if disagreement_eV > ADVISORY["chemical_accuracy_eV"]:
            notes.append(
                f"that gap exceeds chemical accuracy "
                f"({ADVISORY['chemical_accuracy_eV']} eV), so the two "
                "models would not agree on a rate")

    # --- level -------------------------------------------------------
    if blocking:
        level = "BLOCKED"
    elif len(review) >= 2:
        level = "REVIEW"
    elif review:
        level = "CAUTION"
    else:
        level = "ACCEPTED"

    return {
        "level": level,
        "barrier_eV": None if blocking else barrier,
        "blocking": blocking,
        "review": review,
        "notes": notes,
    }


def format_verdict(reaction, verdict):
    """One readable block per reaction."""
    lines = [f"{reaction}: {verdict['level']}"]
    if verdict["barrier_eV"] is not None:
        lines.append(f"  barrier: {verdict['barrier_eV']:.3f} eV")
    for reason in verdict["blocking"]:
        lines.append(f"  BLOCKED: {reason}")
    for reason in verdict["review"]:
        lines.append(f"  review:  {reason}")
    for note in verdict["notes"]:
        lines.append(f"  note:    {note}")
    return "\n".join(lines)
