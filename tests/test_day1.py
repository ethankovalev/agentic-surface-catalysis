"""
Day-1 regression tests. No GPU, no checkpoint, no API key, seconds to run.

    python -m pytest tests/test_day1.py -q

Covers the two things that were silently wrong: which bond breaks, and
which vibrational modes count toward a zero-point energy.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from ase.build import molecule

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools import _orient_for_dissociation, breaking_bond  # noqa: E402
from src.zpe import (  # noqa: E402
    IMAG_MIN_meV,
    _is_linear,
    gas_phase_zpe,
    saddle_zpe,
    zpe_from_modes,
)


# --- orientation and bond selection -----------------------------------

@pytest.mark.parametrize("species", ["H2", "N2"])
def test_diatomic_lies_parallel_to_the_surface(species):
    """Both atoms have to reach their own site, so the bond goes flat.

    This is the behaviour the original code already had; the test exists so
    the polyatomic branch cannot quietly break it.
    """
    ads = _orient_for_dissociation(molecule(species))
    assert abs(ads.positions[0, 2] - ads.positions[1, 2]) < 1e-6


def test_ch4_points_one_ch_bond_at_the_surface():
    """The regression. ASE's g2 CH4 is edge-down; it must not stay that way."""
    ads = _orient_for_dissociation(molecule("CH4"))
    anchor, terminal, length = breaking_bond(ads, list(range(len(ads))))

    assert ads[anchor].symbol == "C"
    assert ads[terminal].symbol == "H"
    # the breaking hydrogen sits a full bond length directly below carbon
    assert ads.positions[terminal, 2] == pytest.approx(
        ads.positions[anchor, 2] - length, abs=1e-6)
    # and it is the lowest atom in the molecule, so nothing is between it
    # and the metal
    assert ads.positions[terminal, 2] == pytest.approx(
        ads.positions[:, 2].min(), abs=1e-6)


def test_unoriented_ch4_would_have_broken_the_upward_bond():
    """Documents the bug, so nobody reintroduces it as a simplification.

    Without the rotation the lowest-terminal rule still picks a downward
    hydrogen, but the old longest-bond rule picked atom 1, which is up.
    """
    raw = molecule("CH4")
    assert raw.positions[1, 2] > 0          # ASE's first H points up
    _, terminal, _ = breaking_bond(raw, list(range(len(raw))))
    assert raw.positions[terminal, 2] < 0   # height rule already avoids it


def test_breaking_bond_raises_on_a_dissociated_molecule():
    """No guessing. A molecule with no bonds is an error, not a coin flip."""
    ads = molecule("H2")
    ads.positions[1] += [6.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="bonded"):
        breaking_bond(ads, [0, 1])


def test_breaking_bond_picks_the_heaviest_atom_as_anchor():
    """The fragment that stays intact is the one that binds to the metal."""
    ads = _orient_for_dissociation(molecule("H2O"))
    anchor, terminal, _ = breaking_bond(ads, list(range(len(ads))))
    assert ads[anchor].symbol == "O"
    assert ads[terminal].symbol == "H"


# --- zero-point bookkeeping -------------------------------------------

def _real(*eV):
    return np.array([complex(e, 0.0) for e in eV])


def _imag(meV):
    return complex(0.0, meV / 1000.0)


def test_linearity_detection():
    assert _is_linear(molecule("H2"), [0, 1])
    assert _is_linear(molecule("CO2"), [0, 1, 2])
    assert not _is_linear(molecule("CH4"), [0, 1, 2, 3, 4])
    assert not _is_linear(molecule("H2O"), [0, 1, 2])


def test_zpe_is_half_the_sum():
    assert zpe_from_modes(np.array([0.4, 0.2])) == pytest.approx(0.3)


def test_gas_phase_drops_five_free_modes_for_a_diatomic():
    """H2: six modes, five of them free motion, one real stretch."""
    energies = np.array([_imag(2.0), complex(0.001, 0), complex(0.002, 0),
                         _imag(1.0), complex(0.0005, 0), complex(0.545, 0)])
    zpe, kept, dropped = gas_phase_zpe(energies, linear=True)
    assert len(kept) == 1
    assert len(dropped) == 5
    assert zpe == pytest.approx(0.2725)


def test_gas_phase_drops_six_free_modes_for_a_nonlinear_molecule():
    vibrations = [0.37] * 9
    free = [0.001, 0.002, 0.0005, 0.0011, 0.0003, 0.0015]
    zpe, kept, _ = gas_phase_zpe(_real(*(free + vibrations)), linear=False)
    assert len(kept) == 9
    assert zpe == pytest.approx(0.5 * sum(vibrations))


def test_gas_phase_refuses_a_real_imaginary_mode():
    """A molecule at a saddle is not a gas-phase reference."""
    energies = np.array([_imag(60.0)] + [complex(0.001, 0)] * 4
                        + [complex(0.5, 0)])
    with pytest.raises(ValueError, match="not a minimum"):
        gas_phase_zpe(energies, linear=True)


def test_gas_phase_refuses_too_few_modes():
    with pytest.raises(ValueError, match="leaves nothing"):
        gas_phase_zpe(_real(0.1, 0.1, 0.1), linear=True)


def test_saddle_keeps_every_real_mode_and_drops_the_reaction_coordinate():
    """Frustrated translations and rotations are real modes at a TS.

    Dropping them alongside the imaginary one is the obvious mistake and it
    changes dZPE by tens of meV.
    """
    energies = np.array([_imag(90.0)] + [complex(e, 0) for e in
                                         (0.136, 0.070, 0.055, 0.040, 0.025)])
    zpe, kept, reaction = saddle_zpe(energies)
    assert len(kept) == 5
    assert reaction == pytest.approx(90.0)
    assert zpe == pytest.approx(0.5 * (0.136 + 0.070 + 0.055 + 0.040 + 0.025))


def test_saddle_refuses_a_minimum():
    with pytest.raises(ValueError, match="minimum"):
        saddle_zpe(_real(0.1, 0.2, 0.3))


def test_saddle_refuses_a_higher_order_stationary_point():
    energies = np.array([_imag(90.0), _imag(40.0), complex(0.1, 0)])
    with pytest.raises(ValueError, match="higher-order"):
        saddle_zpe(energies)


def test_noise_level_imaginary_counts_as_real_and_keeps_the_mode_count():
    """A 2 meV imaginary is a finite difference, not a reaction coordinate.

    It must still contribute a mode, or the TS and the gas reference end up
    with mode counts that cannot be compared.
    """
    tiny = IMAG_MIN_meV / 2.0
    energies = np.array([_imag(90.0), _imag(tiny), complex(0.1, 0)])
    zpe, kept, _ = saddle_zpe(energies)
    assert len(kept) == 2
    assert zpe == pytest.approx(0.5 * (tiny / 1000.0 + 0.1))
