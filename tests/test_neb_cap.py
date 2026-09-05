"""Tests for the run_neb attempt cap.

No GPU, no checkpoint, no API key. Run with:

    python -m pytest tests/test_neb_cap.py -v

The simulation agent decides to rerun run_neb the instant a band fails,
inside the same turn, without consulting the validation agent. Guidance
written into validation_agent_prompt is never read at that moment. Four
prompt revisions failed for that reason. The cap therefore lives in the
tool, and these tests assert it cannot be talked around.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools import MAX_NEB_ATTEMPTS, _neb_attempt_guard  # noqa: E402


def test_first_attempt_allowed():
    assert _neb_attempt_guard(0, peak_exists=False) is None


def test_second_attempt_allowed():
    assert _neb_attempt_guard(1, peak_exists=True) is None


def test_third_attempt_refused():
    """The specific failure: a third band after two unconverged ones."""
    msg = _neb_attempt_guard(2, peak_exists=True)
    assert msg is not None, "third call was allowed; the cap does nothing"
    assert msg.startswith("FAILED"), (
        "refusal must start with FAILED so the agent treats it as a tool "
        "failure rather than a result")


def test_refusal_names_the_alternative():
    """A refusal that does not say what to do instead invites a workaround."""
    msg = _neb_attempt_guard(2, peak_exists=True)
    assert "refine_saddle" in msg, (
        "the refusal must name refine_saddle, or the agent has no route "
        "forward and will try to work around the cap")


def test_refusal_without_a_peak_says_report_it():
    """With no peak image saved, refinement has no starting point.

    The honest outcome is an unresolved reaction, not another band.
    """
    msg = _neb_attempt_guard(2, peak_exists=False)
    assert msg is not None
    assert "unresolved" in msg
    assert "refine_saddle" not in msg, (
        "must not point at refine_saddle when there is no peak.traj for it "
        "to start from")


def test_cap_holds_beyond_the_limit():
    """A fourth or fifth attempt must also be refused, not just the third."""
    for attempts in (3, 4, 10):
        assert _neb_attempt_guard(attempts, peak_exists=True) is not None, (
            f"attempt {attempts + 1} was allowed through")


def test_cap_is_two():
    """If this changes, the prompt text stating 'capped at two' goes stale."""
    assert MAX_NEB_ATTEMPTS == 2, (
        "simulation_agent_prompt tells the agent run_neb is capped at two "
        "calls. Change both together or the prompt lies.")


@pytest.mark.parametrize("attempts", [0, 1])
def test_allowed_attempts_return_none_not_empty_string(attempts):
    """None means proceed. An empty string would be falsy but not None, and
    a caller checking `if refusal is not None` would wrongly refuse."""
    result = _neb_attempt_guard(attempts, peak_exists=True)
    assert result is None and result != ""
