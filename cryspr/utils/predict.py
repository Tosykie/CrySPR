"""Utilities for crystal structure prediction"""
import sys, os
from ..optimization.local import one_structure_from_one_formula_one_spg
from ..optimization.local import stepwise_relax
from ..calculators import *
from .log import now

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize.optimize import Optimizer
from ase.optimize import FIRE, LBFGS, BFGSLineSearch

from pymatgen.core.composition import Composition

from pyxtal.symmetry import Group

def random_predict(
        reduced_formula: str,
        Z_start: int,
        Z_end: int,
        space_group_numbers: list[int] = 0,
        n_trial_each_space_group: int = 3,
        relax_calculator: Calculator = M3GNetCalculator(),
        optimizer: Optimizer = FIRE,
        fmax: float = 0.02,
        verbose: bool = True,
        wdir: str = "./",
        logfile: str = "-",
        relax_logfile_prefix: str = "",
        relax_logfile_postfix: str = "",
        write_cif: bool = True,
        cif_prefix: str = "",
        cif_posfix: str = "",
) -> dict:
    """
    Predict structure for a given reduced chemical formula
    for various space groups.
    """
    content = "\n".join(
        [
            f"[{now()}] CrySPR Info: Input chemical formula = {reduced_formula}.",
            f"Input Z_start = {Z_start}, Z_end = {Z_end}.",
            f"Use ML-IAPs = {relax_calculator.__class__.__name__}",
            f"Use local optimization algorithm = {optimizer.__class__.__name__}",
            f"Use fmax = {fmax}",
            f"\n",
        ]
    )
    if verbose:
        print(content)
    if logfile != "-":
        with open(logfile, mode='at') as f:
            f.write(content)

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

    compatible_spg_with_Z = {}  # {Z: [full_formula, [spgs]]}

    if space_group_numbers == 0:
        spg_list = list(range(1, 231))
    else:
        spg_list = space_group_numbers

    # Enumerate all compatible space groups for each formula
    for Z in range(Z_start, Z_end + 1):
        full_composition: Composition = composition_in.reduced_composition * Z
        full_formula: str = full_composition.formula.replace(" ", "")
        ions_and_numbers: dict = full_composition.get_el_amt_dict()
        species = list(ions_and_numbers.keys())
        number_of_ions = [int(f) for f in list(ions_and_numbers.values())]

        compatible_spg = []
        for spg_num in spg_list:
            spg = Group(group=spg_num)
            compatible_wyckoff_combinations: list = spg.list_wyckoff_combinations(number_of_ions)[0]
            if len(compatible_wyckoff_combinations) > 0:
                compatible_spg.append(spg_num)
                # log
                number_of_solutions = len(compatible_wyckoff_combinations)
                content = "\n".join(
                    [
                        f"[{now()}] CrySPR Info: Formula {full_formula} is compatible with {spg.symbol} ({spg.number})",
                        f"[{now()}] CrySPR Info: by {number_of_solutions} combinations:",
                        f",".join(species),
                        f"\n".join(list(map(str, compatible_wyckoff_combinations))),
                        f"\n",
                    ]
                )
                if verbose:
                    print(content)
                if logfile != "-":
                    with open(logfile, mode='at') as f:
                        f.write(content)
            else:
                content = "\n".join(
                    [
                        f"[{now()}] CrySPR Warning: Formula {full_formula} is NOT compatible with {spg.symbol} ({spg.number})",
                        f"\n",
                    ]
                )
                if verbose:
                    print(content)
                if logfile != "-":
                    with open(logfile, mode='at') as f:
                        f.write(content)
                continue

        if len(compatible_spg) > 0:
            compatible_spg_with_Z[Z] = [full_formula, compatible_spg]
    if len(compatible_spg_with_Z) == 0:
        content = "\n".join(
            [
                f"[{now()}] CrySPR Error: No any compatible combination found "
                f"for formula {reduced_formula} and space groups:\n {spg_list}",
                f"[{now()}] CrySPR Error: Exited with code 7.",
                f"\n",
            ]
        )
        if verbose:
            print(content)
        if logfile != "-":
            with open(logfile, mode='at') as f:
                f.write(content)
        sys.exit(7)

    # Generate n_trial_each_space_group structure for each compatible space group
    reservoir: dict = compatible_spg_with_Z.copy()  # {Z: [full_formula, {spg: Atoms}]}
    for Z in reservoir.keys():
        full_formula_in: str = reservoir[Z][0]
        spgs: list = reservoir[Z][1]
        trial_structure_for_each_formula = {}

        for spg in spgs:
            count = 1
            trial_structure_for_each_spg = []
            while count <= n_trial_each_space_group:
                output = one_structure_from_one_formula_one_spg(
                    full_formula=full_formula_in,
                    space_group_number=spg,
                    random_seed=None,
                    wdir=f"{wdir}/{Z}fu/spg{spg}/trial{count}",
                    logfile=logfile,
                    write_cif=write_cif,
                    cif_prefix=cif_prefix,
                    cif_posfix=cif_posfix,
                )
                # log
                content = "\n".join(
                    [
                        f"[{now()}] CrySPR Info: Done structure generation: trial {count}.",
                        f"\n",
                    ]
                )
                if verbose:
                    print(content)
                if logfile != "-":
                    with open(logfile, mode='at') as f:
                        f.write(content)

                candidate: Atoms = output["ase_Atoms"]
                # relax the candidate
                atoms_relaxd: Atoms = stepwise_relax(
                    atoms_in=candidate,
                    calculator=relax_calculator,
                    optimizer=optimizer,
                    fmax=fmax,
                    wdir=f"{wdir}/{Z}fu/spg{spg}/trial{count}",
                    logfile_prefix=relax_logfile_prefix,
                    logfile_postfix=relax_logfile_postfix,
                )
                trial_structure_for_each_spg.append(atoms_relaxd)

                # log
                content = "\n".join(
                    [
                        f"[{now()}] CrySPR Info: Done structure relaxation: trial {count}.",
                        f"\n",
                    ]
                )
                if verbose:
                    print(content)
                if logfile != "-":
                    with open(logfile, mode='at') as f:
                        f.write(content)
                count += 1
            trial_structure_for_each_formula[spg] = trial_structure_for_each_spg

        # Update the dict
        reservoir[Z][1] = trial_structure_for_each_formula

    content = "\n".join(
        [
            f"[{now()}] CrySPR Info: Done structure generation and relaxation for {reduced_formula}.",
            f"\n",
        ]
    )
    if verbose:
        print(content)
    if logfile != "-":
        with open(logfile, mode='at') as f:
            f.write(content)

    return reservoir