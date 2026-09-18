"""
Escalation assessment tests. No GPU, no checkpoint, no API key, seconds.

    python -m pytest tests/test_escalation.py -q

Covers the physics blocks, the advisory-only disagreement signal, and
the separation between them: a run that fails physics must stay blocked
no matter how well the models agree.

Real records from the project are used where possible, so a change that
would have let CH4_Ni111_step through fails here.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.escalation import assess, format_verdict  # noqa: E402


# --- real records from the project ------------------------------------

# The run whose polished agent report exit_gate refused, 2026-09-14.
CH4_NI111_STEP = {
    "saddle": {"first_order_saddle": False,
               "imaginary_modes_meV": [83.88, 10.89]},
    "saddle_connectivity": {"connects": None},
    "endpoint_modes": {"initial": {"is_minimum": True},
                       "final": {"is_minimum": True}},
    "zpe": {"ok": True, "dzpe_eV": -0.186},
    "barrier_zpe_eV": 0.4908,
    "neb": {"reaction_energy_eV": 0.4844},
}

# The project's closest result: UMA seeded, dispersion off.
H2_CU111 = {
    "saddle": {"first_order_saddle": True, "imaginary_modes_meV": [121.0]},
    "saddle_connectivity": {"connects": True},
    "endpoint_modes": {"initial": {"is_minimum": True},
                       "final": {"is_minimum": True}},
    "zpe": {"ok": True, "dzpe_eV": -0.027},
    "barrier_zpe_eV": 0.657,
    "neb": {"reaction_energy_eV": 0.30},
    "r_b_drift_A": -0.116,
}

# Worst in-domain disagreement in the seeded track: 0.226 eV.
N2_RU0001_TERRACE = {
    "saddle": {"first_order_saddle": True, "imaginary_modes_meV": [95.0]},
    "saddle_connectivity": {"connects": True},
    "endpoint_modes": {"initial": {"is_minimum": True},
                       "final": {"is_minimum": True}},
    "zpe": {"ok": True, "dzpe_eV": -0.032},
    "barrier_zpe_eV": 1.090,
    "neb": {"reaction_energy_eV": 0.40},
    "r_b_drift_A": -0.006,
}


# --- physics blocks ---------------------------------------------------

def test_two_imaginary_modes_is_blocked():
    v = assess(CH4_NI111_STEP)
    assert v["level"] == "BLOCKED"
    assert v["barrier_eV"] is None
    assert any("first order saddle" in b for b in v["blocking"])


def test_unconfirmed_connectivity_is_blocked():
    v = assess(CH4_NI111_STEP)
    assert any("connect" in b for b in v["blocking"])


def test_endpoint_that_is_not_a_minimum_is_blocked():
    bad = dict(H2_CU111)
    bad["endpoint_modes"] = {"initial": {"is_minimum": True},
                             "final": {"is_minimum": False}}
    v = assess(bad)
    assert v["level"] == "BLOCKED"
    assert any("final endpoint" in b for b in v["blocking"])


def test_incomplete_zpe_is_blocked():
    bad = dict(H2_CU111)
    bad["zpe"] = {"ok": False}
    assert assess(bad)["level"] == "BLOCKED"


def test_clean_run_is_accepted():
    v = assess(H2_CU111)
    assert v["level"] == "ACCEPTED"
    assert v["barrier_eV"] == pytest.approx(0.657)


# --- the advisory signal is advisory ----------------------------------

def test_perfect_agreement_cannot_rescue_a_blocked_run():
    """The separation that makes this safe to expose to the agent."""
    v = assess(CH4_NI111_STEP, disagreement_eV=0.001)
    assert v["level"] == "BLOCKED"


def test_large_disagreement_flags_but_does_not_block():
    v = assess(N2_RU0001_TERRACE, disagreement_eV=0.226)
    assert v["level"] in ("CAUTION", "REVIEW")
    assert v["barrier_eV"] == pytest.approx(1.090)
    assert any("disagree by" in r for r in v["review"])


def test_unvalidated_threshold_says_so_in_its_own_text():
    """The caveat must travel with the number, not live only in docs."""
    v = assess(N2_RU0001_TERRACE, disagreement_eV=0.226)
    assert any("NOT a validated threshold" in r for r in v["review"])


def test_small_disagreement_is_noted_not_flagged():
    v = assess(H2_CU111, disagreement_eV=0.041)
    assert v["level"] == "ACCEPTED"
    assert any("agree to" in n for n in v["notes"])


def test_absence_of_a_second_model_is_stated():
    v = assess(H2_CU111)
    assert any("no second in-domain model" in n for n in v["notes"])


# --- soft signals -----------------------------------------------------

def test_barrier_below_model_resolution_is_flagged():
    tiny = dict(H2_CU111)
    tiny["barrier_zpe_eV"] = 0.012
    tiny["neb"] = {"reaction_energy_eV": -0.2}
    v = assess(tiny)
    assert v["level"] in ("CAUTION", "REVIEW")
    assert any("resolution" in r for r in v["review"])


def test_endothermic_barrier_below_reaction_energy_is_flagged():
    impossible = dict(H2_CU111)
    impossible["barrier_zpe_eV"] = 0.20
    impossible["neb"] = {"reaction_energy_eV": 0.50}
    v = assess(impossible)
    assert any("not physically possible" in r for r in v["review"])


def test_large_bond_drift_is_flagged():
    drifted = dict(H2_CU111)
    drifted["r_b_drift_A"] = -0.786
    v = assess(drifted)
    assert any("drifted" in r for r in v["review"])


# --- robustness -------------------------------------------------------

@pytest.mark.parametrize("record", [
    {},
    {"saddle": None},
    {"saddle": {}, "zpe": None},
    {"barrier_zpe_eV": "not a number"},
    {"barrier_zpe_eV": True},
])
def test_malformed_records_are_blocked_not_crashed_on(record):
    """assess runs over whole sweeps unattended; it must never raise."""
    v = assess(record)
    assert v["level"] == "BLOCKED"


def test_format_verdict_is_readable():
    text = format_verdict("CH4_Ni111_step", assess(CH4_NI111_STEP))
    assert "CH4_Ni111_step: BLOCKED" in text
