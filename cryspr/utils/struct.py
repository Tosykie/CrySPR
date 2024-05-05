"""Utilities for generating, manipulating structure object"""

import sys
from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.symmetry.groups import sg_symbol_from_int_number
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal import pyxtal
from pyxtal.lattice import Lattice as PxLattice
from pyxtal.tolerance import Tol_matrix
from pymatgen.core.lattice import Lattice as PgLattice
from .log import now
def get_crystal_system_from_lattice(lattice: PgLattice):
    dummy_postion = [[0, 0, 0]]
    dummy_species = ["H"]
    structure = Structure(lattice, dummy_species, dummy_postion)
    analyzer = SpacegroupAnalyzer(structure)
    crystal_system = analyzer.get_crystal_system()
    return crystal_system

def get_structure_from_pyxtal(
        reduced_formula: str,
        space_group_number: int,
        lattice_parameters: list[float, float, float, float, float, float] = None,
        Z_start: int = 1,
        Z_end: int = 1,
        inter_dist_matx: Tol_matrix = None,
        random_seed = None,
        max_try: int = 20,
        verbose: bool = True,
        logfile: str = "-",
        write_cif: bool = True,
        cif_prefix: str = "",
        cif_posfix: str = "",
) -> dict:
    # initially written by Ruiming (Raymond) Zhu, refined by Wei Nong
    # added compatibility and crystal system checking
    if lattice_parameters is None:
        inter_dist_matx = Tol_matrix(prototype="atomic", factor=1.25)
    else:
        content = (f"[{now()}] CrySPR Warning: Ignore the default inter-atomic distance matrix,"
                   f" use instead input lattice parameters.\n"
                   f"[{now()}] CrySPR Warning: The input lattice parameters might not be compatible with Z range:"
                   f" from {Z_start} to {Z_end}. You should always check ...")
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
    composition_in = Composition(reduced_formula)
    reduced_formula_refined, Z_in_reduced_formula = composition_in.get_reduced_formula_and_factor()
    if Z_in_reduced_formula > 1:
        content = "\n".join(
            [
                f"[{now()}] CrySPR Warning: Input chemical formula {reduced_formula} is not reduced.",
                f"[{now()}] CrySPR Warning: Reduced formula {reduced_formula_refined} will be used instead.",
                f"\n",
            ]
        )
        if verbose:
            print(content)
        if logfile != "-":
            with open(logfile, mode='at') as f:
                f.write(content)

    # dict with Z as the key
    if Z_end < Z_start:
        Z_end = Z_start
    pxstrc_with_Z: dict = {}
    for Z in range(Z_start, Z_end+1):
        full_composition: Composition = composition_in.reduced_composition * Z
        full_formula: str = full_composition.formula.replace(" ", "")
        ions_and_numbers: dict = full_composition.get_el_amt_dict()
        number_of_ions = [int(f) for f in list(ions_and_numbers.values())]
        species = list(ions_and_numbers.keys())

        # pyxtal crystal generation
        pxstrc = pyxtal()
        try:
            pxstrc.from_random(
                dim=3,
                group=space_group_number,
                species=species,
                lattice=None if lattice_parameters is None else px_lattice,
                sites=None,
                numIons=number_of_ions,
                tm=inter_dist_matx,
                max_count=max_try,
                seed=random_seed,
            )
            cifname = "_".join([reduced_formula_refined,
                                full_formula,
                                f"{Z}fu",
                                f"spg{space_group_number}",
                                ]
                               )
            if write_cif:
                ciffile = "_".join([cif_prefix, cifname, cif_posfix]).strip("_") + ".cif"
                pxstrc.to_file(filename= ciffile)
            strc_ase = pxstrc.to_ase()
            pxstrc_with_Z[Z] = {
                "full_formula": full_formula,
                "pyxtal": pxstrc,
                "ase_Atoms": strc_ase,
            }
            DoF_total = pxstrc.get_dof()
            DoF_lattice = pxstrc.lattice.dof
            DoF_postions = DoF_total - DoF_lattice

            content = "\n".join(
                [
                    f"[{now()}] CrySPR Info: Successfully generated structure for {reduced_formula} with Z = {Z}",
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

        except Exception as e:
            content = f"[{now()}] CrySPR Error: Pyxtal exception occurred:\n{e}\n"
            if verbose:
                print(content)
            if logfile != "-":
                with open(logfile, mode='at') as f:
                    f.write(content)
            continue

    return pxstrc_with_Z

#TO-DO: Added more utils
def scale_volume(strc_in: Structure, target_volume: float, verbose: bool = True):
    scaled_structure = strc_in.copy()
    scaled_structure.scale_lattice(target_volume)
    if verbose:
        print(f"[{now()}] CrySPR Info: Volume scaled from {strc_in.volume:.4f} to {scaled_structure.volume:.4f} A^3\n")
    return scaled_structure

