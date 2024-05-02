"""Local optimization (relaxation) by ML-IAPs through ASE API"""
from packaging import version
import os
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
from pymatgen.io.ase import AseAtomsAdaptor
from ..utils.log import now

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
    spg0 = get_spacegroup(atoms, symprec=1e-3)
    if fix_symmetry:
        atoms.set_constraint([FixSymmetry(atoms)])
    if cell_filter is not None:
        target = cell_filter(atoms, hydrostatic_strain=hydrostatic_strain)
    else:
        target = atoms

    E0 = atoms.get_potential_energy()
    logcontent1 = "\n".join([
                f"[{now()}] Info: Start structure relaxation.",
                f"[{now()}] Info: Total energy for initial input = {E0:12.5f} eV",
                f"[{now()}] Info: Initial symmetry {spg0.symbol} ({spg0.no})",
                f"[{now()}] Info: Symmetry constraint? {'Yes' if fix_symmetry else 'No'}",
                f"[{now()}] Info: Relax cell? {'Yes' if cell_filter is not None else 'No'}",
                f"[{now()}] Info: Relax atomic postions? {'Yes' if not fix_fractional else 'No'}",
                f"#{'-'*60}#",
                f"\n",
            ])
    if logfile == "-":
        print(logcontent1)
    else:
        with open(f"{wdir}/{logfile}", mode='at') as f:
            f.write(logcontent1)
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

    logcontent2 = "\n".join([
                f"#{'-' * 60}#",
                f"[{now()}] Info: End structure relaxation.",
                f"[{now()}] Info: Total energy for final structure = {E1:12.5f} eV",
                f"[{now()}] Info: Final symmetry {spg1.symbol} ({spg1.no})",
                f"Optimized Cell: {atoms.cell.cellpar()}",
                f"Cell diff (%): {cell_diff}",
                f"Scaled positions:\n{atoms.get_scaled_positions()}",
                f"\n",
            ]
            )
    if logfile == "-":
        print(logcontent2)
    else:
        with open(f"{wdir}/{logfile}", mode='at') as f:
            f.write(logcontent2)
    return atoms

def stepwise_relax(
        atoms_in: Atoms,
        calculator: Calculator,
        optimizer: Optimizer = FIRE,
        hydrostatic_strain: bool = False,
        fmax: float = 0.02,
        steps_limit: int = 500,
        logfile_prefix: str = "",
        logfile_postfix: str = "",
        wdir: str = "./",
) -> Atoms:
    """
    Do fix-cell relaxation first then cell + atomic postions.
    :param atoms_in: an input ase.Atoms object
    :param calculator: an ase calculator to be used
    :param optimizer: a local optimization algorithm, default FIRE
    :param hydrostatic_strain: if do isometrically cell-scaled relaxation, default True
    :param fmax: the max force per atom (unit as defined by the calculator), default 0.02
    :param steps_limit: the max steps to break the relaxation loop, default 500
    :param logfile_prefix: a prefix of the log file, default ""
    :param logfile_postfix: a postfix of the log file, default ""
    :param wdir: string of working directory, default "./" (current)
    :return: the last ase.Atoms trajectory
    """

    if not os.path.exists(wdir):
        os.makedirs(wdir)
    atoms = atoms_in.copy()
    full_formula = atoms.get_chemical_formula(mode="metal")
    reduced_formula = atoms.get_chemical_formula(mode="metal", empirical=True)
    structure0 = AseAtomsAdaptor.get_structure(atoms)
    structure0.to(filename=f'{wdir}/{reduced_formula}_{full_formula}_0_initial_symmetrized.cif', symprec=1e-3)

    # fix cell relaxation
    logfile1 = "_".join([logfile_prefix,  "fix-cell", logfile_postfix, ]).strip("_") + ".log"
    atoms1 = run_ase_relaxer(
        atoms_in=atoms,
        calculator=calculator,
        optimizer=optimizer,
        fix_symmetry=True,
        cell_filter=None,
        fix_fractional=False,
        hydrostatic_strain=hydrostatic_strain,
        fmax=fmax,
        steps_limit=steps_limit,
        logfile=logfile1,
        wdir=wdir,
    )

    atoms = atoms1.copy()
    structure1 = AseAtomsAdaptor.get_structure(atoms)
    _ = structure1.to(filename=f'{wdir}/{reduced_formula}_{full_formula}_1_fix-cell_symmetrized.cif', symprec=1e-3)

    # relax both cell and atomic positions
    logfile2 = "_".join([logfile_prefix, "cell+positions", logfile_postfix, ]).strip("_") + ".log"
    atoms2 = run_ase_relaxer(
        atoms_in=atoms,
        calculator=calculator,
        optimizer=optimizer,
        fix_symmetry=True,
        cell_filter=CellFilter,
        fix_fractional=False,
        hydrostatic_strain=hydrostatic_strain,
        fmax=fmax,
        steps_limit=steps_limit,
        logfile=logfile2,
        wdir=wdir,
    )
    structure2 = AseAtomsAdaptor.get_structure(atoms2)
    _ = structure2.to(filename=f'{wdir}/{reduced_formula}_{full_formula}_2_cell+pos_symmetrized.cif', symprec=1e-3)

    return atoms2
