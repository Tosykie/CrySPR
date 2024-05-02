from typing import Self, Union
from pymatgen.core.composition import Composition
from pymatgen.core import Element

class WyckoffGene:
    """
    The Wyckoff postion-based representation of crystal structures
    """
    def __init__(self,
                 compostion: Union[Composition, str] = None,
                 space_group_no: int = 1,
                 element_wyckoff_letters: dict[Union[Element, str], str] = None,
                 wyckoff_coordinates: dict[str, float] = None,
                 ):
        self.compostion = compostion
        self.space_group_no = space_group_no
        self.element_wycoff_letters = element_wyckoff_letters

    def __str__(self):
        wyckoff_string = (f'Wyckoff gene| compostion = {self.compostion}\n'
                          f'            | space group No. = {self.space_group_no}\n'
                          f'            | element\tWycoff site\tcoordinate\n'
                          f'            | ') # TO-DO
        return wyckoff_string

    def __eq__(self, other: Self):
        pass

    def from_file(self, filepath: str = None,):
        pass

    def from_string(self, string: str = None):
        pass

    def check_compatability(self):
        pass

