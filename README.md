# CrySPR

**Maintainers**: [Wei Nong](https://github.com/Tosykie) [email: nw2y47@outlool.com] ; [Ruiming Zhu](https://github.com/RaymondZhurm) [email: raymond_zhurm@outlook.com]

**CrySPR** /ˈkrɪspɚ/ is a Python interface for crystal structure pre-relaxation and prediction using machine-learning interatomic potentials (ML-IAPs). Features include:

- Implement structure generation from the input info (e.g., formula, Z, space group, etc.)  via `pyxtal` and local structure optimization/relaxation through `ase` calculator using ML-IAPs;
- Implement global search task for crystal structure prediction using 1) random search (done), and 2) particle swarm optimization (PSO) for a given reduced formula (in dev ...);
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

### PyPI distribution

```bash
$ pip install cryspr
```

### Source code (GitHub)

1. Download the repo or git clone.

2. Add the CrySPR project into the system PYTHONPATH either by, e.g., on Linux/Mac OS

```bash
$ export PYTHONPATH=/path/to/CrySPR:$PYTHONPATH
```

or by in the python code

```python
import sys
sys.path.insert(0, '/path/to/CrySPR')
```



## Usage

To be updated.



## Examples

### Ground-state CaTiO3

This example shows the implementation of crystal structure relaxation and prediction from three test space groups (No. 62, 74, 140) through a random prediction mode. The ML-IAP calculator is CHGNet.

Refers to [`examples/cryspr_random_predict_CaTiO3.ipynb`](https://github.com/Tosykie/CrySPR/blob/main/examples/cryspr_random_predict_CaTiO3.ipynb)



