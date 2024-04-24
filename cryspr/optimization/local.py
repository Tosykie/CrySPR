"""Local optimization (relaxation) by ML-IAPs through ASE API"""
from packaging import version

import ase
if version.parse(ase.__version__) > version.parse("3.22.1"):
    from ase.constraints import FixSymmetry, FixAtoms
    from ase.filters import FrechetCellFilter as CellFilter
else:
    from ase.spacegroup.symmetrize import FixSymmetry
    from ase.constraints import ExpCellFilter as CellFilter
    from ase.constraints import FixAtoms
    print("Warning: No FrechetCellFilter in ase with version ",
          f"{ase.__version__}, the ExpCellFilter will be used instead.")

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize.optimize import Optimizer
from ase.optimize import FIRE, LBFGS, BFGSLineSearch
from ase.io import read, write
from ase.spacegroup import get_spacegroup
# from pymatgen.core import Structure
# from pymatgen.io.ase import AseAtomsAdaptor
from datetime import datetime

def run_ase_relaxer(
        atoms_in: Atoms,
        calculator: Calculator,
        optimizer: Optimizer = FIRE,
        cell_filter = None,
        fix_symmetry: bool = True,
        fix_fractional: bool = False,
        hydrostatic_strain: bool = False,
        fmax: float = 0.02,
        steps_limit: int = 500,
        logfile: str = "-",
        wdir: str = "./",
) -> Atoms:
    atoms = atoms_in.copy()
    full_formula = atoms.get_chemical_formula(mode="metal")
    reduced_formula = atoms.get_chemical_formula(mode="metal", empirical=True)
    atoms.set_calculator(calculator)
    if fix_fractional:
        atoms.set_constraint([FixAtoms(indices=[atom.index for atom in atoms])])
    spg0 = get_spacegroup(atoms, symprec=1e-5)
    if fix_symmetry:
        atoms.set_constraint([FixSymmetry(atoms)])
    if cell_filter is not None:
        target = cell_filter(atoms, hydrostatic_strain=hydrostatic_strain)
    else:
        target = atoms
    now = datetime.now()
    strnow = now.strftime("%Y-%b-%d %H:%M:%S")
    E0 = atoms.get_potential_energy()
    with open(f"{wdir}/{logfile}", mode='at') as f:
        f.write(
            "\n".join([
                f"Info: Start structure relaxation {strnow}",
                f"Info: Total energy for initial input = {E0:12.5f} eV",
                f"Info: Initial symmetry {spg0.symbol} ({spg0.no})",
                f"Info: Symmetry constraint? {'Yes' if fix_symmetry else 'No'}",
                f"Info: Relax cell? {'Yes' if cell_filter is not None else 'No'}",
                f"Info: Relax atomic postions? {'Yes' if not fix_fractional else 'No'}",
                f"#{'-'*42}#",
                f"\n",
            ]
            )
        )
    opt = optimizer(atoms=target,
                    trajectory=f"{wdir}/{reduced_formula}_{full_formula}_opt.traj",
                    logfile=f"{wdir}/{logfile}",
                    )
    opt.run(fmax=fmax, steps=steps_limit)
    if cell_filter is None:
        write(filename=f'{wdir}/{reduced_formula}_{full_formula}_fix-cell.cif',
              images=atoms,
              format="cif",
              )
    else:
        write(filename=f'{wdir}/{reduced_formula}_{full_formula}_free.cif',
              images=atoms,
              format="cif",
              )
    cell_diff = (atoms.cell.cellpar() / atoms_in.cell.cellpar() - 1.0) * 100
    E1 = atoms.get_potential_energy()
    spg1 = get_spacegroup(atoms, symprec=1e-5)
    now = datetime.now()
    strnow = now.strftime("%Y-%b-%d %H:%M:%S")
    with open(f"{wdir}/{logfile}", mode='at') as f:
        f.write(
            "\n".join([
                f"#{'-' * 42}#",
                f"Info: End structure relaxation {strnow}",
                f"Info: Total energy for final structure = {E1:12.5f} eV",
                f"Info: Final symmetry {spg1.symbol} ({spg1.no})",
                f"Optimized Cell: {atoms.cell.cellpar()}",
                f"Cell diff (%): {cell_diff}",
                f"Scaled positions:\n{atoms.get_scaled_positions()}",
                f"\n",
            ]
            )
        )

    return atoms

