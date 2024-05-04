# CrySPR

CrySPR /ˈkrɪspɚ/ is a Python interface for crystal structure pre-relaxation and prediction using machine-learning interatomic potentials (ML-IAPs). Features include:

- Structure generation (pyxtal) from the input info (e.g., formula, Z, space group, etc.)  and local (optimization) relaxation (ase, ML-IAPs)
- Implement global search task for crystal structure prediction using particle swarm optimization (PSO) foe a given reduced formula (in dev ...)
- More in development

The original old repo [Fast-Universal-CSP-Platform](https://github.com/RaymondZhurm/Fast-Universal-CSP-Platform) @[RaymondZhurm](https://github.com/RaymondZhurm)

## Python dependencies

```
python >= 3.9
ase # https://wiki.fysik.dtu.dk/ase/install.html
pymatgen # https://pymatgen.org/installation.html
pyxtal # https://pyxtal.readthedocs.io/en/latest/Installation.html#installation
torch # https://pytorch.org/get-started/locally/#linux-installation
matgl # https://matgl.ai/#installation
chgnet # https://chgnet.lbl.gov/#installation
mace-torch # https://mace-docs.readthedocs.io/en/latest/guide/installation.html
scikit-opt # https://scikit-opt.github.io/scikit-opt/#/en/README?id=install
```



## Installation

To be updated through pyproject.

At present, just to add the CrySPR project into the system PYTHONPATH.



## Examples

### Ground-state CaTiO3

See [`examples/cryspr_random_predict_CaTiO3.ipynb`](https://github.com/Tosykie/CrySPR/blob/main/examples/cryspr_random_predict_CaTiO3.ipynb)



