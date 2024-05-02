from typing import Self, Union
from pymatgen.core.composition import Composition
from pymatgen.core import Element
from pyxtal import pyxtal
from pyxtal.symmetry import Group
from pyxtal.tolerance import Tol_matrix
from pyxtal.lattice import Lattice as PxLattice
from pymatgen.core.lattice import Lattice as PgLattice
from .utils.struct import get_crystal_system_from_lattice

class WyckoffGene(pyxtal):
    """
    The Wyckoff postion-based representation of crystal structures by the aid of pyxtal.
    """
    def __init__(self,
                 full_formula: Union[str, Composition],
                 space_group_number: int = 1,
                 element_wyckoff_sites: dict[str, dict] = None,
                 lattice_parameters: list[float, float, float, float, float, float] = None,
                 inter_dist_matx: Tol_matrix = None,
                 max_try: int = 20,
                 random_seed=None,
                 ):
        full_composition: Composition = Composition(full_formula)
        ions_and_numbers: dict = full_composition.get_el_amt_dict()
        number_of_ions = [int(f) for f in list(ions_and_numbers.values())]
        species = list(ions_and_numbers.keys())
        space_group = Group(space_group_number)
        if element_wyckoff_sites is not None:
            wyckoff_sites: list = []
            for element in species:
                wyckoff_sites.append(element_wyckoff_sites[element])
        else:
            wyckoff_sites = None

        # super class call and initialize from input
        super().__init__().from_random(
            dim = 3,
            species=species,
            group=space_group,
            numIons=number_of_ions,
            lattice=None,               #TO-DO: update the lattice?
            sites = wyckoff_sites,
            conventional = True,
            max_count = max_try,
            seed = random_seed,
            tm = inter_dist_matx,
    )
        self.formula = full_formula
        self.elements = species
        self.space_group_number = space_group_number
        self.space_group = space_group
        self.element_wyckoff_letters = element_wyckoff_sites

    def __eq__(self, other: Self):
        """Check if two Wyckoff Genes are equal (if possible?)"""
        pass

