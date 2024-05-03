"""Global optimization"""

from typing import Callable
from ase import Atoms
from pyxtal.symmetry import Group
from sko.PSO import PSO
from.local import stepwise_relax

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

def search_by_pso(objective_func: Callable,
                  n_dim: int, # dimension of particles, which is the number of variables of objective_func
                  lower_boundary: list, # lower boundary of search space
                  upper_boundary: list, # upper boundary of search space
                  population: int = 80, # population size (number of particles)
                  inertia_weight: float = 0.8, # PSO ratio in the population for each generation
                  max_pso_generation: int = 200, # max PSO step
                  acceleration_constant1: float = 0.5,
                  acceleration_constant2: float = 0.5,
                  n_cpus: int = 0, # Number of processes, 0 means use all cpu
                  verbose=True,
                  *args,
                  **kwargs,
                  ) -> PSO:

    pso = PSO(func=objective_func,
              n_dim=n_dim,
              pop=population,
              max_iter=max_pso_generation,
              lb=lower_boundary,
              ub=upper_boundary,
              w=inertia_weight,
              c1=acceleration_constant1,
              c2=acceleration_constant2,
              # n_processes=n_cpus, # only available in dev version of sko
              verbose=verbose,
              *args,
              **kwargs,
              )
    pso.run()
    # print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    return pso