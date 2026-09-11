"""
Zero-point energy correction to the barrier.

DROP THIS IN AT src/zpe.py, then wire it up (three edits, listed at the
bottom of this docstring).

WHY THIS EXISTS
---------------
SBH10's tabulated quantity is NOT a classical barrier. From the paper,
Section 2.1 and Scheme 1: transition state energies are determined as the
*zero-point corrected* energy difference between the transition state and
the isolated gas phase molecule. The paper's own Table 2 lists the
correction it applied, per reaction, and every one of the ten is negative:
between -0.03 and -0.14 eV.

A NEB or a refined saddle gives the classical electronic barrier. Scoring
that directly against the SBH10 reference is a convention mismatch worth
0.03-0.14 eV, in a fixed direction, on every reaction in the set. That is
the same order as the errors being interpreted, and it biases every
computed barrier HIGH, which flatters a model that undershoots and
punishes one that overshoots. Since the dispersion axis is exactly a
question of undershoot versus overshoot, leaving this out does not add
noise to the conclusion, it inverts it.

WHAT IT COMPUTES
----------------
    dZPE     = ZPE(transition state) - ZPE(gas-phase molecule)
    E_barrier(zpe) = E_barrier(gas-referenced, classical) + dZPE

Both ZPEs are harmonic, over the adsorbate atoms only, with the slab held
rigid. That is the standard surface-science approximation and it is what
reproduces the magnitudes in the paper's Table 2.

Mode bookkeeping, which is where this goes wrong silently if done
casually:

  transition state, N adsorbate atoms -> 3N modes, of which EXACTLY ONE is
      imaginary (the reaction coordinate). It is dropped. Frustrated
      translations and rotations are real modes here and are KEPT - they
      are a genuine part of the TS zero-point energy and they are most of
      why dZPE is not simply minus half the broken stretch.

  gas-phase molecule -> 3N modes, of which 5 (linear) or 6 (nonlinear) are
      free translations and rotations sitting at essentially zero
      frequency, often numerically imaginary. They are dropped by
      magnitude. The remaining 3N-5 / 3N-6 are the real vibrations.

Sanity: H2 gas keeps 1 mode, ZPE ~ 0.27 eV. CH4 gas keeps 9 modes, ZPE ~
1.19 eV. If either comes out far from that, the Hessian is wrong, not the
bookkeeping.

ZERO SILENT FAILURES
--------------------
Every one of these raises a FAILED string rather than returning a number:
  - no confirmed first-order saddle (there is nothing to take a Hessian at)
  - the saddle Hessian has zero, or more than one, imaginary mode
  - a real mode at the TS is imaginary above threshold after the reaction
    mode is removed
  - the gas molecule has an imaginary mode left after the 5/6 free modes
    are dropped
  - |dZPE| > 0.5 eV, which is not a zero-point correction, it is a broken
    Hessian
There is no clamping and no fallback to "0.0 if it did not work".

BLINDNESS
---------
Nothing here reads src/benchmark.py and no reference value appears in this
file. The paper's Table 2 dZPE values are a post-hoc check for the runner,
not an input to the agent. Keep it that way.

WIRING (three edits)
--------------------
1. src/tools.py, at the end of the imports:
       from src.zpe import compute_zpe_correction, check_zpe_applied
   or import this module in whatever list src/graph.py builds its tool
   sets from, next to compute_gas_referenced_barrier.

2. src/graph.py: add `compute_zpe_correction` to the Simulation_Agent tool
   list and `check_zpe_applied` to the Validation_Agent tool list.

3. src/prompt.py, simulation agent prompt, after the sentence telling it
   to call compute_gas_referenced_barrier:
       "Then call compute_zpe_correction. The barrier is not final until
        it has run."
"""

import sys
from pathlib import Path

import numpy as np
from ase.io import read
from ase.vibrations import Vibrations
from langchain_core.tools import tool

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src import store
from src.calculators import new_calculator


# --- thresholds -------------------------------------------------------

# Below this a mode is numerical noise rather than motion. Matched to
# SADDLE_IMAG_MIN_meV in tools.py deliberately: the same Hessian decides
# "is this a first-order saddle" and "which mode do I drop", and two
# different thresholds would let a mode count as imaginary for one
# question and real for the other.
IMAG_MIN_meV = 5.0

# The gas reference is a molecule 8 A above a slab, so its translations and
# rotations are very nearly free and land on either side of zero depending
# on finite-difference noise. Imaginaries up to this size are tolerated
# ONLY among those 5 or 6 modes. Anything imaginary above it, or imaginary
# among the real vibrations, means the molecule is not at a minimum.
GAS_FREE_MODE_MAX_meV = 25.0

# A zero-point correction larger than this is not a zero-point
# correction. The physical range for this benchmark is roughly -0.15 to 0.
MAX_PLAUSIBLE_dZPE_eV = 0.5

# Vibrations displacement. ASE defaults to 0.01 A. MLIP force fields are
# noisier than DFT at that step, and the low-frequency frustrated modes at
# a TS are exactly where that noise shows up as spurious imaginaries.
VIB_DELTA = 0.015


def _adsorbate_indices(atoms):
    """Adsorbate atoms are tagged 2 by _tag() in tools.py."""
    tags = atoms.get_tags()
    return [i for i in range(len(atoms)) if tags[i] == 2]


def _is_linear(atoms, indices) -> bool:
    """True if the adsorbate atoms are collinear.

    Decides whether 5 or 6 free modes are removed from the gas reference.
    Getting it wrong costs one real vibrational mode, which for H2 is the
    entire zero-point energy, so it is not a detail.
    """
    if len(indices) < 3:
        return True
    pos = atoms.positions[indices]
    pos = pos - pos.mean(axis=0)
    sv = np.linalg.svd(pos, compute_uv=False)
    if sv[0] < 1e-9:
        raise ValueError("adsorbate atoms are coincident; geometry is broken")
    return bool(sv[1] < 0.05 * sv[0])


def _mode_energies(atoms, indices, name, model_key, with_d3):
    """Harmonic mode energies in eV, complex where imaginary.

    Slab held rigid: only `indices` are displaced.
    """
    import shutil

    atoms = atoms.copy()
    atoms.calc = new_calculator(model_key, with_d3=with_d3)
    shutil.rmtree(name, ignore_errors=True)
    try:
        vib = Vibrations(atoms, indices=indices, name=name, delta=VIB_DELTA)
        vib.run()
        energies = np.asarray(vib.get_energies())
    finally:
        shutil.rmtree(name, ignore_errors=True)

    expected = 3 * len(indices)
    if len(energies) != expected:
        raise ValueError(
            f"expected {expected} modes for {len(indices)} displaced atoms, "
            f"got {len(energies)}")
    return energies


def _split_modes(energies):
    """(real_eV, imaginary_meV) with noise-level imaginaries folded to real.

    A mode 2 meV below zero is a converged optimiser and a finite
    displacement, not a reaction coordinate. Folding it to +|E| rather than
    discarding it keeps the mode count exact, which is what every check
    below relies on.
    """
    real, imag = [], []
    for e in energies:
        if np.iscomplex(e) and abs(e.imag) > 0:
            mev = abs(e.imag) * 1000.0
            if mev > IMAG_MIN_meV:
                imag.append(mev)
            else:
                real.append(abs(e.imag))
        else:
            real.append(float(np.real(e)))
    return np.asarray(real, dtype=float), sorted(imag, reverse=True)


def zpe_from_modes(real_eV) -> float:
    """Half the sum of the real mode energies."""
    return 0.5 * float(np.sum(real_eV))


def gas_phase_zpe(energies, linear: bool):
    """ZPE of the free molecule, free translations and rotations removed.

    Returns (zpe_eV, kept_modes_eV, dropped_modes_eV).

    The 5 or 6 free modes sit at essentially zero frequency and are removed
    by MAGNITUDE, not by sign - at 8 A above a slab a free rotation lands
    on either side of zero depending on the finite-difference noise, and
    filtering by sign keeps some and drops others.
    """
    n_free = 5 if linear else 6
    if len(energies) <= n_free:
        raise ValueError(
            f"{len(energies)} modes for a {'linear' if linear else 'nonlinear'} "
            f"molecule leaves nothing after removing {n_free} free modes")

    magnitudes = np.array([abs(np.real(e)) + abs(np.imag(e)) for e in energies])
    order = np.argsort(magnitudes)
    dropped_idx = set(int(k) for k in order[:n_free])

    kept, dropped = [], []
    for k, e in enumerate(energies):
        mev = (abs(e.imag) if np.iscomplex(e) else 0.0) * 1000.0
        value = abs(e.imag) if np.iscomplex(e) and abs(e.imag) > 0 else float(np.real(e))

        # An imaginary mode has to be judged before the free modes are
        # dropped, not after. A genuine saddle-point mode is often smaller
        # in magnitude than a real vibration, so it sorts into the bottom
        # 5 or 6 and gets discarded as "free motion" - the molecule is then
        # silently treated as a minimum when it is not. Caught by
        # test_gas_phase_refuses_a_real_imaginary_mode.
        if mev > IMAG_MIN_meV:
            free_rotor = k in dropped_idx and mev <= GAS_FREE_MODE_MAX_meV
            if not free_rotor:
                raise ValueError(
                    f"gas-phase molecule has an imaginary mode of {mev:.1f} meV: "
                    "the reference geometry is not a minimum. Relax gasref.traj "
                    "harder before correcting.")

        if k in dropped_idx:
            dropped.append(value)
        else:
            kept.append(value)

    return zpe_from_modes(np.asarray(kept)), kept, dropped


def saddle_zpe(energies):
    """ZPE of the transition state, reaction coordinate removed.

    Returns (zpe_eV, kept_modes_eV, reaction_mode_meV).
    """
    real, imag = _split_modes(energies)

    if len(imag) == 0:
        raise ValueError(
            "no imaginary mode at the saddle above "
            f"{IMAG_MIN_meV:.0f} meV: this geometry is a minimum, not a "
            "transition state, and has no reaction coordinate to remove")
    if len(imag) > 1:
        raise ValueError(
            f"{len(imag)} imaginary modes at the saddle "
            f"({', '.join(f'{m:.1f}' for m in imag)} meV): this is a "
            "higher-order stationary point. The barrier taken from it is "
            "not a reaction barrier and no zero-point correction makes it one")

    return zpe_from_modes(real), list(real), imag[0]


@tool
def compute_zpe_correction(model_key: str = None, with_d3: bool = True,
                           scope: str = "adsorbate") -> str:
    """Apply the zero-point correction that the benchmark convention requires.

    The barrier from the band or the refined saddle is a classical
    electronic barrier. The quantity measured by these experiments is the
    zero-point corrected barrier: the difference in zero-point energy
    between the transition state and the free molecule shifts it, always
    downward for a dissociation, typically by 0.03 to 0.15 eV.

    Requires a confirmed first-order saddle (run refine_saddle) and a
    relaxed gas reference, and must run AFTER
    compute_gas_referenced_barrier.

    model_key, with_d3: must match what the barrier was computed with. A
        zero-point correction from a different potential energy surface
        than the barrier is not a correction to that barrier.
    scope: only "adsorbate" is supported. Slab modes are not included; they
        very nearly cancel between the two states and including them costs
        a Hessian over a hundred atoms.
    """
    if scope != "adsorbate":
        return (f"FAILED: scope='{scope}' is not supported. Only 'adsorbate' "
                "is implemented; slab modes are assumed to cancel.")

    barrier_gas = store.get("barrier_gas_eV")
    if barrier_gas is None:
        return ("FAILED: no gas-referenced barrier. Call "
                "compute_gas_referenced_barrier first - the zero-point "
                "correction is applied to that number, not to the raw band.")

    saddle = store.get("saddle")
    connectivity = store.get("saddle_connectivity")
    if not (saddle and saddle.get("converged") and saddle.get("first_order_saddle")):
        return ("FAILED: no confirmed first-order saddle. A zero-point "
                "correction needs a Hessian at the transition state, and the "
                "NEB's highest image is not one. Call refine_saddle first.")
    if connectivity is not None and not connectivity.get("connects"):
        return ("FAILED: the refined saddle does not connect the endpoints of "
                "this reaction. Correcting its zero-point energy would make a "
                "wrong barrier look more precise, not more correct.")

    saddle_file = Path(config.WORK_DIR) / "saddle.traj"
    gasref_file = Path(config.WORK_DIR) / "gasref.traj"
    for f in (saddle_file, gasref_file):
        if not f.exists():
            return f"FAILED: {f.name} does not exist."

    # The dispersion setting has to match the barrier's, or the two numbers
    # live on different surfaces. The store knows what the band actually
    # used; trust that over what the agent passed in.
    neb = store.get("neb") or {}
    if "with_d3" in neb and bool(neb["with_d3"]) != bool(with_d3):
        return (f"FAILED: the barrier was computed with d3={neb['with_d3']} "
                f"but this call passes with_d3={with_d3}. A zero-point "
                "correction from a different surface than the barrier is not "
                "a correction to that barrier.")

    model_key = model_key or store.get("model_key") or config.DEFAULT_MODEL

    ts = read(str(saddle_file))
    gas = read(str(gasref_file))

    ts_idx = _adsorbate_indices(ts)
    gas_idx = _adsorbate_indices(gas)
    if not ts_idx or not gas_idx:
        return ("FAILED: no atoms tagged as adsorbate. The trajectories were "
                "written without tags, so the molecule cannot be separated "
                "from the slab.")
    if len(ts_idx) != len(gas_idx):
        return (f"FAILED: {len(ts_idx)} adsorbate atoms at the saddle but "
                f"{len(gas_idx)} in the gas reference. These are not the same "
                "molecule and the difference of their zero-point energies "
                "means nothing.")

    try:
        linear = _is_linear(gas, gas_idx)
        e_ts = _mode_energies(ts, ts_idx, str(Path(config.WORK_DIR) / "vib_ts"),
                              model_key, with_d3)
        e_gas = _mode_energies(gas, gas_idx, str(Path(config.WORK_DIR) / "vib_gas"),
                               model_key, with_d3)
        zpe_ts, ts_modes, reaction_mode_meV = saddle_zpe(e_ts)
        zpe_gas, gas_modes, gas_dropped = gas_phase_zpe(e_gas, linear)
    except Exception as exc:
        store.put("zpe", {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return f"FAILED: {type(exc).__name__}: {exc}"

    dzpe = zpe_ts - zpe_gas

    if abs(dzpe) > MAX_PLAUSIBLE_dZPE_eV:
        store.put("zpe", {"ok": False,
                          "error": f"implausible dZPE {dzpe:+.3f} eV"})
        return (f"FAILED: zero-point correction came out at {dzpe:+.3f} eV "
                f"(ZPE_TS {zpe_ts:.3f}, ZPE_gas {zpe_gas:.3f}). Anything past "
                f"{MAX_PLAUSIBLE_dZPE_eV:.1f} eV is a broken Hessian, not a "
                "zero-point energy. Do not use this barrier.")

    barrier_zpe = barrier_gas + dzpe

    record = {
        "ok": True,
        "dzpe_eV": float(dzpe),
        "zpe_ts_eV": float(zpe_ts),
        "zpe_gas_eV": float(zpe_gas),
        "reaction_mode_meV": float(reaction_mode_meV),
        "n_modes_ts_kept": len(ts_modes),
        "n_modes_gas_kept": len(gas_modes),
        "gas_linear": bool(linear),
        "gas_free_modes_dropped_eV": [float(x) for x in gas_dropped],
        "model_key": model_key,
        "with_d3": bool(with_d3),
        "delta_A": VIB_DELTA,
    }
    store.put("zpe", record)

    # barrier_eV is what gets scored. It now carries the same convention as
    # the reference. The classical number is kept alongside it, not
    # overwritten - the difference between the two conventions is itself a
    # result, and a stored number whose convention is unrecorded is a
    # number nobody can use later.
    store.put("barrier_classical_eV", float(barrier_gas))
    store.put("barrier_zpe_eV", float(barrier_zpe))
    store.put("barrier_eV", float(barrier_zpe))
    store.put("barrier_convention", "zpe_corrected_gas_referenced")

    sign = "lowers" if dzpe < 0 else "raises"
    return (f"Zero-point correction {dzpe:+.3f} eV (ZPE at the transition "
            f"state {zpe_ts:.3f} eV over {len(ts_modes)} real modes, gas-phase "
            f"molecule {zpe_gas:.3f} eV over {len(gas_modes)}). This {sign} "
            f"the barrier from {barrier_gas:.3f} eV to {barrier_zpe:.3f} eV. "
            f"Reaction mode {reaction_mode_meV:.0f} meV imaginary.")


@tool
def check_zpe_applied() -> str:
    """Check the barrier carries the same convention as the benchmark.

    A classical electronic barrier and a zero-point corrected one are
    different quantities. Reporting one where the other is expected is a
    systematic error in a fixed direction on every reaction, which is worse
    than noise because it does not average out.
    """
    record = store.get("zpe")
    barrier = store.get("barrier_eV")
    convention = store.get("barrier_convention")

    if barrier is None:
        passed, detail = False, "no barrier computed"
    elif record is None:
        passed, detail = False, ("compute_zpe_correction was never run, so "
                                 "the barrier is classical and the reference "
                                 "is not")
    elif not record.get("ok"):
        passed, detail = False, f"zero-point correction failed: {record.get('error')}"
    elif convention != "zpe_corrected_gas_referenced":
        passed, detail = False, (f"barrier convention is '{convention}', not "
                                 "zero-point corrected")
    elif record["dzpe_eV"] > 0:
        # Not fatal. Worth surfacing: for dissociation the breaking stretch
        # softens at the transition state, so the correction is normally
        # negative. A positive one means the frustrated modes at the TS are
        # stiffer than the molecule's own vibrations, which happens, but
        # rarely, and is more often a sign of a bad Hessian.
        passed, detail = True, (f"applied, {record['dzpe_eV']:+.3f} eV, but "
                                "POSITIVE - unusual for a dissociation, check "
                                "the saddle geometry")
    else:
        passed, detail = True, (f"applied, {record['dzpe_eV']:+.3f} eV, over "
                                f"{record['n_modes_ts_kept']} TS modes and "
                                f"{record['n_modes_gas_kept']} gas modes")

    store.record_check("zpe", passed, detail)
    return f"zpe: {'PASS' if passed else 'FAIL'} - {detail}"
