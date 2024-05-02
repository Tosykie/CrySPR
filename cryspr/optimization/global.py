"""Global optimization"""

from ase import Atoms
from pyxtal.symmetry import Group
from sko.PSO import PSO
from.local import stepwise_relax

def local_energy(atoms_relaxed: Atoms):
    return atoms_relaxed.get_potential_energy()

def initialize_search_space():
    """TO-DO: To get the lower and upper boundaries
    - given a chemical (reduced) formula,
    - set the Z start and Z end
    - set space group: [1, 230]
    - enumerate all possible combinations of Wyckoff postions for each full formula. How to define lb and ub?
    - update the lattice and coordiante through pyxtal.wyckoff_site.atom_site.update(pos=None, reset_wp=False)
    - ...
    """
    pass

def search_by_pso(*args, **kwargs):
    population = 80
    max_pso_step = 200
    pso = PSO(func=local_energy,
              n_dim=len(lb),
              pop=population,
              max_iter=max_pso_step,
              lb=None,
              ub=None,
              w=0.8,
              c1=0.5,
              c2=0.5,
              verbose=True,
              )
    pso.run()
    # print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    return pso