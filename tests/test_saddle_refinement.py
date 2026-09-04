"""Tests for Sella saddle refinement.

These use ASE's built-in EMT calculator, so they need no GPU, no model
checkpoint and no API key. Run them on a laptop:

    pip install pytest sella
    python -m pytest tests/test_saddle_refinement.py -v

The physics being tested is not the MLIP's - it is the refinement machinery:
does a coarse band plus refinement land on the same saddle as a fine band, is
the result a genuine first-order saddle, and do the failure modes report
themselves honestly rather than returning a plausible number.
"""
import shutil
import warnings

import numpy as np
import pytest
from ase.build import fcc111, add_adsorbate
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.mep import NEB
from ase.optimize import FIRE
from ase.vibrations import Vibrations

warnings.filterwarnings("ignore")

sella = pytest.importorskip("sella", reason="sella is not installed")
from sella import Sella  # noqa: E402

FMAX = 0.02
IMAG_MIN_meV = 5.0


def _slab(site):
    """Cu adatom on Cu(111) at a named hollow, relaxed. Bottom layers fixed."""
    s = fcc111("Cu", size=(3, 3, 4), vacuum=8.0)
    s.pbc = True
    add_adsorbate(s, "Cu", 2.0, site)
    zmax = s.positions[:, 2].max()
    s.set_constraint(FixAtoms(mask=[a.position[2] < zmax - 3.0 for a in s]))
    s.calc = EMT()
    FIRE(s, logfile=None).run(fmax=FMAX, steps=300)
    return s


def _band(initial, final, n_images):
    """A climbing-image NEB. Returns (images, uphill profile, peak index)."""
    images = [initial.copy()] + [initial.copy() for _ in range(n_images)] + [final.copy()]
    for im in images:
        im.calc = EMT()
    neb = NEB(images, climb=False, method="improvedtangent")
    neb.interpolate(method="idpp")
    FIRE(neb, logfile=None).run(fmax=0.3, steps=200)
    neb.climb = True
    FIRE(neb, logfile=None).run(fmax=0.05, steps=300)
    energies = [im.get_potential_energy() for im in images]
    uphill = [e - energies[0] for e in energies]
    return images, uphill, int(np.argmax(uphill))


def _refine(atoms, template):
    """Refine to a saddle, carrying the template's constraints across."""
    ts = atoms.copy()
    ts.calc = EMT()
    zmax = template.positions[:, 2].max()
    ts.set_constraint(FixAtoms(mask=[a.position[2] < zmax - 3.0 for a in ts]))
    dyn = Sella(ts, order=1, internal=False, logfile=None)
    converged = dyn.run(fmax=FMAX, steps=300)
    return ts, bool(converged), int(dyn.get_number_of_steps())


def _imaginary(atoms, indices, name="_vibtest"):
    shutil.rmtree(name, ignore_errors=True)
    try:
        vib = Vibrations(atoms, indices=indices, name=name)
        vib.run()
        energies = vib.get_energies()
    finally:
        shutil.rmtree(name, ignore_errors=True)
    return [abs(e.imag) * 1000.0 for e in energies
            if np.iscomplex(e) and abs(e.imag) * 1000.0 > IMAG_MIN_meV]


@pytest.fixture(scope="module")
def endpoints():
    return _slab("fcc"), _slab("hcp")


def test_coarse_band_fails_path_resolved(endpoints):
    """The premise: a coarse band under-samples the peak.

    If this ever stops failing, the rest of these tests are testing nothing,
    because refinement would have no problem left to solve.
    """
    initial, final = endpoints
    _, uphill, peak = _band(initial, final, n_images=3)
    steps = [abs(uphill[i + 1] - uphill[i]) for i in range(len(uphill) - 1)]
    assert uphill[peak] > 0, "test system has no barrier; pick a different one"
    assert max(steps) / uphill[peak] > 0.5, (
        "the 3-image band resolved the path better than expected, so it no "
        "longer demonstrates the problem refinement exists to fix")


def test_refinement_from_coarse_matches_fine_band(endpoints):
    """The whole point: refine the peak instead of densifying the band.

    A 3-image band that fails path_resolved, plus refinement, must land on the
    same saddle as an 11-image band. If it does not, adding images is still
    necessary and this feature is not earning its place.
    """
    initial, final = endpoints
    coarse_images, coarse_up, coarse_peak = _band(initial, final, n_images=3)
    fine_images, fine_up, fine_peak = _band(initial, final, n_images=11)

    refined, converged, _ = _refine(coarse_images[coarse_peak], initial)
    assert converged, "refinement from the coarse band did not converge"

    refined_barrier = refined.get_potential_energy() - coarse_images[0].get_potential_energy()
    assert refined_barrier == pytest.approx(fine_up[fine_peak], abs=0.01), (
        f"refined coarse band gave {refined_barrier:.4f} eV but the fine band "
        f"gave {fine_up[fine_peak]:.4f} eV; they should agree")


def test_refined_structure_is_a_first_order_saddle(endpoints):
    """A converged optimisation is not evidence of a transition state.

    Exactly one imaginary mode is what makes the energy a barrier rather than
    just a number, so it is checked rather than assumed.
    """
    initial, final = endpoints
    images, uphill, peak = _band(initial, final, n_images=3)
    refined, converged, _ = _refine(images[peak], initial)
    assert converged

    ads = [len(refined) - 1]          # the adatom
    imag = _imaginary(refined, ads)
    assert len(imag) == 1, (
        f"expected exactly one imaginary mode, found {len(imag)}: "
        f"{[round(m) for m in imag]} meV")


def test_a_minimum_is_not_reported_as_a_saddle(endpoints):
    """The negative control.

    Refining from a relaxed minimum must not produce something the mode check
    calls a transition state. This is the failure that would be invisible
    without the check: a converged optimisation and a plausible energy.
    """
    initial, _ = endpoints
    ads = [len(initial) - 1]
    imag = _imaginary(initial, ads)
    assert len(imag) == 0, (
        f"a relaxed minimum reported {len(imag)} imaginary modes; the mode "
        f"check cannot distinguish minima from saddles")


def test_refinement_is_cheaper_than_densifying(endpoints):
    """Refinement should take few steps, or it is not the cheap option."""
    initial, final = endpoints
    images, uphill, peak = _band(initial, final, n_images=3)
    _, converged, n_steps = _refine(images[peak], initial)
    assert converged
    assert n_steps < 50, (
        f"refinement took {n_steps} steps; the case for it over adding images "
        f"rests on it being cheap")


def test_constraints_survive_refinement(endpoints):
    """Fixed layers must stay fixed, or the barrier is on a different system."""
    initial, final = endpoints
    images, uphill, peak = _band(initial, final, n_images=3)
    before = images[peak].copy()
    refined, converged, _ = _refine(images[peak], initial)
    assert converged

    zmax = initial.positions[:, 2].max()
    fixed = [i for i, a in enumerate(before) if a.position[2] < zmax - 3.0]
    moved = np.abs(refined.positions[fixed] - before.positions[fixed]).max()
    assert moved < 1e-6, f"constrained atoms moved by {moved:.2e} A"
