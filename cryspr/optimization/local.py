"""Local optimization (relaxation) by ML-IAPs through ASE API"""
import sys

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

from pymatgen.core.composition import Composition
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.lattice import Lattice as PgLattice
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.symmetry.groups import sg_symbol_from_int_number

from pyxtal import pyxtal
from pyxtal.symmetry import Group
from pyxtal.tolerance import Tol_matrix
from pyxtal.lattice import Lattice as PxLattice

from ..utils.log import now
from ..utils.struct import get_crystal_system_from_lattice

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
    atoms.calc = calculator
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
                f"[{now()}] CrySPR Info: Start structure relaxation.",
                f"[{now()}] CrySPR Info: Total energy for initial input = {E0:12.5f} eV",
                f"[{now()}] CrySPR Info: Initial symmetry {spg0.symbol} ({spg0.no})",
                f"[{now()}] CrySPR Info: Symmetry constraint? {'Yes' if fix_symmetry else 'No'}",
                f"[{now()}] CrySPR Info: Relax cell? {'Yes' if cell_filter is not None else 'No'}",
                f"[{now()}] CrySPR Info: Relax atomic postions? {'Yes' if not fix_fractional else 'No'}",
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
        write(filename=f'{wdir}/{reduced_formula}_{full_formula}_cell+pos.cif',
              images=atoms,
              format="cif",
              )
    cell_diff = (atoms.cell.cellpar() / atoms_in.cell.cellpar() - 1.0) * 100
    E1 = atoms.get_potential_energy()
    spg1 = get_spacegroup(atoms, symprec=1e-5)

    logcontent2 = "\n".join([
                f"#{'-' * 60}#",
                f"[{now()}] CrySPR Info: End structure relaxation.",
                f"[{now()}] CrySPR Info: Total energy for final structure = {E1:12.5f} eV",
                f"[{now()}] CrySPR Info: Final symmetry {spg1.symbol} ({spg1.no})",
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

def one_structure_from_one_formula_one_spg(
        full_formula: str,
        space_group_number: int,
        lattice_parameters: list[float, float, float, float, float, float] = None,
        assign_Wyckoff_sites: list[list] = None,
        inter_dist_matx: Tol_matrix = None,
        random_seed = None,
        max_try: int = 20,
        verbose: bool = True,
        wdir="./",
        logfile: str = "-",
        write_cif: bool = False,
        cif_prefix: str = "",
        cif_posfix: str = "",
) -> dict:
    """Relax and get the total energy for single formula with one specific space group"""
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    if lattice_parameters is None:
        inter_dist_matx = Tol_matrix(prototype="atomic", factor=1.25)
    else:
        content = (f"[{now()}] CrySPR Warning: Ignore the default inter-atomic distance matrix,"
                   f" use instead input lattice parameters.\n")
        if verbose:
            print(content)
        if logfile != "-":
            with open(logfile, mode='at') as f:
                f.write(content)
    spg_int_symbol = sg_symbol_from_int_number(space_group_number)
    space_group = SpaceGroup(spg_int_symbol)
    crystal_system = space_group.crystal_system
    ltype = crystal_system

    # Check if the lattice parameters are compatible with the space group
    if lattice_parameters is not None:
        pg_lattice = PgLattice.from_parameters(*lattice_parameters)
        ltype_para = get_crystal_system_from_lattice(pg_lattice)
        if ltype_para.lower() != ltype.lower():
            content = (f"[{now()}] CrySPR Error: Lattice type with parameters {lattice_parameters} ({ltype_para})"
                       f" is incompatible with the space group {ltype}!\n")
            if verbose:
                print(content)
            if logfile != "-":
                with open(logfile, mode='at') as f:
                    f.write(content)
            sys.exit(7)
        else:
            px_lattice = PxLattice.from_para(*lattice_parameters, ltype=ltype)

    # formulae enumeration
    full_composition: Composition = Composition(full_formula)
    full_formula: str = full_composition.formula.replace(" ", "")
    reduced_formula_refined, Z_in_full_formula = full_composition.get_reduced_formula_and_factor()
    ions_and_numbers: dict = full_composition.get_el_amt_dict()
    number_of_ions = [int(f) for f in list(ions_and_numbers.values())]
    species = list(ions_and_numbers.keys())

    pxstrc = pyxtal()
    try:
        pxstrc.from_random(
            dim=3,
            group=space_group_number,
            species=species,
            lattice=None if lattice_parameters is None else px_lattice,
            sites=assign_Wyckoff_sites,
            numIons=number_of_ions,
            tm=inter_dist_matx,
            max_count=max_try,
            seed=random_seed,
        )
        cifname = "_".join([reduced_formula_refined,
                            full_formula,
                            f"{Z_in_full_formula}fu",
                            f"spg{space_group_number}",
                            ]
                           )
        if write_cif:
            ciffile = "_".join([cif_prefix, cifname, cif_posfix]).strip("_") + ".cif"
            pxstrc.to_file(filename= f"{wdir}/{ciffile}")
        strc_ase = pxstrc.to_ase()
        pxstrc_dict = {
            "full_formula": full_formula,
            "pyxtal": pxstrc,
            "ase_Atoms": strc_ase,
        }
        DoF_total = pxstrc.get_dof()
        DoF_lattice = pxstrc.lattice.dof
        DoF_postions = DoF_total - DoF_lattice

        content = "\n".join(
            [
                f"[{now()}] CrySPR Info: Successfully generated structure for {full_formula}",
                f"[{now()}] CrySPR Info: Degree of freedom: total = {DoF_total}, lattice = {DoF_lattice}, postions = {DoF_postions}",
                f"[{now()}] CrySPR Info: pyxtal representation:\n{pxstrc}",
                f"\n"
            ]
        )
        if verbose:
            print(content)
        if logfile != "-":
            with open(logfile, mode='at') as f:
                f.write(content)
        return pxstrc_dict
    except Exception as e:
        content = f"[{now()}] CrySPR Error: Pyxtal exception occurred:\n{e}\n"
        if verbose:
            print(content)
        if logfile != "-":
            with open(logfile, mode='at') as f:
                f.write(content)
        return None
