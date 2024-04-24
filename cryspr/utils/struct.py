"""Utilities for generating, manipulating structure object"""

from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.symmetry.groups import sg_symbol_from_int_number
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal import pyxtal
from pyxtal.lattice import Lattice as PxLattice
from pyxtal.tolerance import Tol_matrix
from pymatgen.core.lattice import Lattice as PgLattice

verbose = True
def get_crystal_system_from_lattice(lattice: PgLattice):
    dummy_structure = [[0, 0, 0]]
    dummy_species = ["H"]
    structure = Structure(lattice, dummy_species, dummy_structure)
    analyzer = SpacegroupAnalyzer(structure)
    crystal_system = analyzer.get_crystal_system()
    return crystal_system

def get_structure_from_pyxtal(
        reduced_formula: str,
        space_group_number,
        lattice_parameters: list[float, float, float, float, float, float] = None,
        Z_start: int = 1,
        Z_end: int = 1,
        element_wyckoff_sites: dict[str, str] = None,
        inter_dist_matx: Tol_matrix = Tol_matrix(prototype="atomic", factor=1.25),
) -> dict:
    # initially written by Ruiming (Raymond) Zhu, refined by Wei Nong
    # added compatibility and crystal system checking

    spg_int_symbol = sg_symbol_from_int_number(space_group_number)
    space_group = SpaceGroup(spg_int_symbol)
    crystal_system = space_group.crystal_system
    ltype = crystal_system

    # Check if the lattice parameters are compatible with the space group
    if lattice_parameters is not None:
        pg_lattice = PgLattice.from_parameters(*lattice_parameters)
        ltype_para = get_crystal_system_from_lattice(pg_lattice)
        if ltype_para != ltype:
            if verbose:
                print(f"Error: Input lattice parameters are incompatible with the space group!")
            exit(code=7)
        px_lattice = PxLattice.from_para(*lattice_parameters, ltype=ltype)

    # formulae enumeration
    composition_in = Composition(reduced_formula)
    reduced_formula_refined, Z_in_reduced_formula = composition_in.get_reduced_formula_and_factor()
    if Z_in_reduced_formula > 1:
        if verbose:
            print(f"Warning: Input chemical formula {reduced_formula} is not reduced.\n",
                  f"Warning: Reduced formula {reduced_formula_refined} will be used instead.")

    pxstrc_with_Z: dict = {}
    for Z in range(Z_start, Z_end+1):
        full_composition: Composition = composition_in.reduced_composition * Z
        full_formula: str = full_composition.formula.replace(" ", "")
        ions_and_numbers: dict = full_composition.get_el_amt_dict()
        number_of_ions = [int(f) for f in list(ions_and_numbers.values())]
        species = list(ions_and_numbers.keys())
        if element_wyckoff_sites is not None:
            wyckoff_sites: list = []
            for element in species:
                wyckoff_sites.append(element_wyckoff_sites[element])
        else:
            wyckoff_sites = None

        # pyxtal crystal generation
        pxstrc = pyxtal()
        try:
            pxstrc.from_random(
                dim=3,
                group=space_group_number,
                species=species,
                lattice=None if lattice_parameters is None else px_lattice,
                sites=wyckoff_sites,
                numIons=number_of_ions,
                tm=inter_dist_matx,
                max_count=20,
            )
            cifname = "_".join([reduced_formula_refined,
                                 full_formula,
                                 f"{Z}fu",
                                 ]
                                )
            pxstrc.to_file(filename=cifname + ".cif")
            strc_ase = pxstrc.to_ase()
            pxstrc_with_Z[Z] = {
                "full_formula": full_formula,
                "pyxtal": pxstrc,
                "ase_Atoms": strc_ase,
            }
            DoF_total = pxstrc.get_dof()
            DoF_lattice = pxstrc.lattice.dof
            DoF_postions = DoF_total - DoF_lattice
            if verbose:
                print(
                    f"Info: Successfully generated structure for {reduced_formula} with Z = {Z}\n",
                    f"Info: Degree of freedom: total = {DoF_total}, lattice = {DoF_lattice}, postions = {DoF_postions}\n",
                    f"Info: pyxtal representation\n{pxstrc}",
                )
        except Exception as e:
            print(f"Error: Exception occurred:\n{e}")
            continue

    return pxstrc_with_Z

#TO-DO: Added more utils
def scale_volume(strc_in: Structure, target_volume: float):
    scaled_structure = strc_in.copy()
    scaled_structure.scale_lattice(target_volume)
    if verbose:
        print(f"Info: Volume scaled from {strc_in.volume:.4f} to {scaled_structure.volume:.4f} A^3\n")
    return scaled_structure

