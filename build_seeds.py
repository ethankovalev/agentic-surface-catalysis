"""
Convert the SBH10 SI transition-state POSCARs into tagged ASE structures.

    python build_seeds.py poscars.txt --out seeds/

Produces seeds/<reaction>.traj plus seeds/manifest.json.

PROVENANCE - READ THIS BEFORE USING THE OUTPUT
-----------------------------------------------
Every structure written here came from the Supporting Information of
Sharada, Bligaard, Luntz, Kroes & Norskov, "SBH10: A Benchmark Database
of Barrier Heights on Transition Metal Surfaces", and is a BEEF-vdW
transition state geometry computed by those authors.

A barrier obtained by refining one of these is NOT evidence that the
engine can find a transition state. It answers a narrower question: given
a converged saddle, is the MLIP's energy there correct. Every result
derived from these files carries provenance="seeded" and must never be
averaged into, or reported alongside without labelling, results from the
autonomous pipeline.

SLAB MISMATCH - THE IMPORTANT CAVEAT
------------------------------------
These are 2x2 cells with 6 layers (Ni(211) step: 3x3x4, 36 metal atoms).
The autonomous pipeline builds 3x3 cells with 4 layers. That is a
different adsorbate coverage - 1/4 ML here against 1/9 ML there - and a
different amount of metal relaxation, so the two tracks' energies are NOT
directly comparable to each other. Each is separately comparable to the
SBH10 reference, because the reference was computed in the cell used
here.

Consequence: do not "correct" one track onto the other, and do not read a
difference between them as a model error. It is mostly a coverage
difference. The seeded track's job is to isolate PES accuracy at a known
saddle, not to referee the autonomous track's barrier.

Slab flexibility follows the SI exactly: bottom 4 layers frozen, top 2
free, taken from the POSCAR's own selective-dynamics flags rather than
re-derived. Two exceptions the SI notes and this script preserves
automatically because it just reads the flags: CH4/Ni(111) and
CH4/Ni(211) were computed with completely frozen slabs, to match the
frozen-lattice approximation used in the quantum dynamics those
reference values came from.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.data import atomic_numbers, covalent_radii
from ase.io import write


# Bulk nearest-neighbour distances, for validating that the metal lattice
# survived transcription. A typo in a slab coordinate shifts one of these
# well outside tolerance.
BULK_NN = {"Cu": 2.556, "Pt": 2.775, "Ru": 2.706, "Ni": 2.489}

# SBH17 Table 2 reports breaking-bond lengths for the systems that have a
# published SRP-DFT transition state. Those geometries are SRP-DFT, these
# are BEEF-vdW, so they will not agree exactly - a different functional
# puts the saddle somewhere slightly different. Agreement to a couple of
# tenths of an angstrom says the transcription is sound; exact agreement
# was never expected and its absence is not an error.
SBH17_RB = {
    "H2_Cu111": 1.03,
    "H2_Cu100": 1.23,
    "H2_Pt111": 0.769,
    "H2_Ru0001": 0.751,
    "N2_Ru0001_terrace": 1.741,
    "CH4_Ni111_terrace": 1.606,
    "CH4_Ni111_step": 1.632,
}

# Table S1 of the SI: adsorption sites of the fragments in the
# chemisorbed (final) state. Carried through to the manifest because the
# autonomous pipeline's site search should eventually be checked against
# these, and because CH4/Ni(100) carries a footnote worth keeping.
FINAL_SITES = {
    "H2_Cu111": "hollow, hollow",
    "H2_Cu100": "hollow, hollow",
    "H2_Pt111": "fcc, bridge",
    "H2_Ru0001": "hollow, hollow",
    "N2_Ru0001_terrace": "hollow, hollow",
    "N2_Ru0001_step": "hollow, bridge",
    "CH4_Ru0001": "fcc (CH3), fcc (H)",
    "CH4_Ni100": "hollow (CH3), hollow (H)",
    "CH4_Ni111_terrace": "fcc (CH3), hollow (H)",
    "CH4_Ni111_step": "bridge (CH3), hollow (H)",
}

SITE_NOTES = {
    "CH4_Ni100": ("SI footnote: the BEEF-vdW minimum-energy configuration "
                  "actually has CH3 at the bridge site, not the hollow site "
                  "listed in Table S1."),
}

ADSORBATE_ELEMENTS = {"H", "C", "N", "O"}


def parse_poscar_block(lines):
    """Parse one Cartesian POSCAR with selective dynamics."""
    symbols_line = lines[0].split()
    scale = float(lines[1])
    cell = np.array([[float(x) for x in lines[i].split()] for i in (2, 3, 4)])
    cell *= scale
    counts = [int(x) for x in lines[5].split()]

    idx = 6
    selective = lines[idx].strip().lower().startswith("s")
    if selective:
        idx += 1
    mode = lines[idx].strip().lower()
    if not mode.startswith("c"):
        raise ValueError(f"expected Cartesian coordinates, got {mode!r}")
    idx += 1

    if len(symbols_line) != len(counts):
        raise ValueError(
            f"{len(symbols_line)} species but {len(counts)} counts")

    symbols, positions, fixed = [], [], []
    rows = lines[idx:idx + sum(counts)]
    if len(rows) != sum(counts):
        raise ValueError(
            f"expected {sum(counts)} coordinate rows, found {len(rows)}")

    cursor = 0
    for sym, n in zip(symbols_line, counts):
        if sym not in atomic_numbers:
            raise ValueError(f"unknown element {sym!r}")
        for _ in range(n):
            parts = rows[cursor].split()
            positions.append([float(p) for p in parts[:3]])
            if selective:
                flags = parts[3:6]
                if len(flags) != 3:
                    raise ValueError(
                        f"selective dynamics needs 3 flags, got {flags}")
                # Frozen only if all three directions are frozen. A
                # partially constrained atom is not something FixAtoms can
                # express, so refuse rather than silently rounding it.
                if len(set(flags)) != 1:
                    raise ValueError(
                        f"mixed selective-dynamics flags {flags} cannot be "
                        "represented by FixAtoms")
                fixed.append(flags[0].upper() == "F")
            else:
                fixed.append(False)
            symbols.append(sym)
            cursor += 1

    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    atoms.set_constraint(FixAtoms(indices=[i for i, f in enumerate(fixed) if f]))
    return atoms, fixed


def tag_oc20(atoms):
    """Apply the repo's OC20 convention: 0 bulk, 1 surface, 2 adsorbate.

    Adsorbate atoms are identified by element, not by index, because these
    POSCARs are not consistently ordered - files 1-3 list the adsorbate
    first, files 4-10 list the metal first. Assuming a fixed order here
    would tag the wrong half of three structures and the model would see a
    slab with no adsorbate on it.

    Surface metal atoms are the top layer, by z. Everything below is bulk.
    """
    symbols = atoms.get_chemical_symbols()
    ads = [i for i, s in enumerate(symbols) if s in ADSORBATE_ELEMENTS]
    metal = [i for i, s in enumerate(symbols) if s not in ADSORBATE_ELEMENTS]
    if not ads:
        raise ValueError("no adsorbate atoms found")
    if not metal:
        raise ValueError("no metal atoms found")

    metal_z = atoms.positions[metal, 2]
    top = metal_z.max()
    tags = [0] * len(atoms)
    for i in metal:
        if abs(atoms.positions[i, 2] - top) < 0.8:
            tags[i] = 1
    for i in ads:
        tags[i] = 2
    atoms.set_tags(tags)
    return ads, metal


def breaking_bond(atoms, ads):
    """The bond that dissociates: heaviest adsorbate atom to its lowest
    bonded neighbour. Same rule as src/tools.py so the two agree."""
    anchor = max(ads, key=lambda k: covalent_radii[atoms[k].number])
    r_a = covalent_radii[atoms[anchor].number]
    neighbours = []
    for k in ads:
        if k == anchor:
            continue
        # 2.2x, not the 1.3x used for intact molecules. At a transition
        # state the breaking bond is already stretched well past its
        # equilibrium length - CH4/Ni(100) sits at 1.91 A against a 1.09 A
        # normal C-H - so an intact-molecule cutoff excludes exactly the
        # bond being looked for and silently returns a spectator bond
        # instead.
        reach = 2.2 * (r_a + covalent_radii[atoms[k].number])
        if atoms.get_distance(anchor, k, mic=True) < reach:
            neighbours.append(k)
    if not neighbours:
        raise ValueError("adsorbate has no bonded pair")
    # At a transition state the breaking bond is the LONGEST one, since it
    # is already stretched - the opposite of the intact-molecule rule,
    # where degenerate bonds forced a height-based tiebreak.
    terminal = max(neighbours,
                   key=lambda k: atoms.get_distance(anchor, k, mic=True))
    return anchor, terminal, atoms.get_distance(anchor, terminal, mic=True)


def validate(name, atoms, fixed, ads, metal):
    """Physical checks on the parsed structure. Returns list of problems."""
    problems, notes = [], {}

    metal_syms = {atoms[i].symbol for i in metal}
    if len(metal_syms) != 1:
        problems.append(f"expected one metal element, found {metal_syms}")
    metal_sym = metal_syms.pop() if len(metal_syms) == 1 else None

    # Metal lattice intact? A transcription slip in any slab row moves an
    # atom off its lattice site and changes this.
    if metal_sym in BULK_NN:
        d = atoms.get_all_distances(mic=True)
        nn = []
        for i in metal:
            row = [d[i][j] for j in metal if j != i]
            nn.append(min(row))
        nn_min, nn_max = min(nn), max(nn)
        notes["metal_nn_A"] = [round(nn_min, 3), round(nn_max, 3)]
        expected = BULK_NN[metal_sym]
        if not (expected - 0.35 < nn_min and nn_max < expected + 0.45):
            problems.append(
                f"metal nearest-neighbour distances {nn_min:.2f}-{nn_max:.2f} A "
                f"outside the expected band around bulk {metal_sym} "
                f"({expected:.2f} A) - a slab coordinate is likely mistyped")

    # Adsorbate above the top metal layer?
    top_z = max(atoms.positions[i, 2] for i in metal)
    ads_z = [atoms.positions[i, 2] - top_z for i in ads]
    notes["adsorbate_height_A"] = [round(min(ads_z), 3), round(max(ads_z), 3)]
    if min(ads_z) < -0.5:
        problems.append(
            f"an adsorbate atom sits {min(ads_z):.2f} A below the top metal "
            "layer - for a step geometry this can be legitimate, otherwise "
            "the structure is wrong")

    # Breaking bond, cross-checked against SBH17 where available.
    try:
        a, b, rb = breaking_bond(atoms, ads)
        notes["breaking_bond"] = f"{atoms[a].symbol}-{atoms[b].symbol}"
        notes["r_b_A"] = round(rb, 3)
        intact = covalent_radii[atoms[a].number] + covalent_radii[atoms[b].number]
        if rb < intact * 0.9:
            problems.append(
                f"breaking bond {rb:.2f} A is not stretched relative to a "
                f"normal {notes['breaking_bond']} bond ({intact:.2f} A); this "
                "does not look like a transition state")
        if name in SBH17_RB:
            ref = SBH17_RB[name]
            notes["r_b_sbh17_srp"] = ref
            notes["r_b_delta"] = round(rb - ref, 3)
            # Reported, never failed on. These are BEEF-vdW saddles; SBH17
            # Table 2 lists SRP-DFT saddles. For weakly activated systems
            # the two functionals do not merely shift the barrier, they put
            # it in a different place along the reaction coordinate: SBH17
            # labels H2/Pt(111) and H2/Ru(0001) "top (early)" with a nearly
            # intact H2 (r_b 0.77, 0.75), while BEEF-vdW finds a late
            # saddle near 1.0-1.1 A. A large delta here is a physical
            # statement about functional disagreement, not a transcription
            # error, and the internal checks above are what actually catch
            # a mistyped coordinate.
            if abs(rb - ref) > 0.35:
                notes["r_b_note"] = (
                    f"differs from the SBH17 SRP-DFT geometry by "
                    f"{rb - ref:+.2f} A - early vs late saddle, expected for "
                    "a weakly activated system")
    except ValueError as exc:
        problems.append(f"breaking bond: {exc}")

    n_fixed = sum(fixed)
    notes["atoms"] = len(atoms)
    notes["n_adsorbate"] = len(ads)
    notes["n_metal"] = len(metal)
    notes["n_frozen"] = n_fixed
    notes["formula"] = atoms.get_chemical_formula()
    notes["cell_a_b_c_A"] = [round(float(np.linalg.norm(v)), 3)
                             for v in atoms.cell]

    if n_fixed == len(atoms):
        problems.append("every atom is frozen; nothing could be refined")

    return problems, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--out", type=Path, default=Path("seeds"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    text = args.source.read_text()

    blocks, current, name = {}, [], None
    for line in text.splitlines():
        if line.startswith("### "):
            if name:
                blocks[name] = current
            name, current = line[4:].strip(), []
        elif name and line.strip():
            current.append(line)
    if name:
        blocks[name] = current

    manifest, failed = {}, []
    for name, lines in blocks.items():
        try:
            atoms, fixed = parse_poscar_block(lines)
            ads, metal = tag_oc20(atoms)
            problems, notes = validate(name, atoms, fixed, ads, metal)
        except Exception as exc:
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
            failed.append(name)
            continue

        notes["provenance"] = "seeded"
        notes["source"] = ("SBH10 SI, Sharada/Bligaard/Luntz/Kroes/Norskov, "
                           "BEEF-vdW transition state geometry")
        notes["final_state_sites"] = FINAL_SITES.get(name)
        if name in SITE_NOTES:
            notes["site_note"] = SITE_NOTES[name]
        notes["problems"] = problems

        path = args.out / f"{name}.traj"
        write(str(path), atoms)
        manifest[name] = notes

        flag = "FAIL" if problems else " ok "
        print(f"[{flag}] {name:22s} {notes['formula']:14s} "
              f"{notes.get('breaking_bond', '??'):>5s} r_b={notes.get('r_b_A')} "
              f"nn={notes.get('metal_nn_A')} frozen={notes['n_frozen']}/{notes['atoms']}")
        for p in problems:
            print(f"         - {p}")
            failed.append(name)

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)} structures written to {args.out}/")
    if failed:
        print(f"PROBLEMS in: {sorted(set(failed))}")
        return 1
    print("All structures parsed and validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
