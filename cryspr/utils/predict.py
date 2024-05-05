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

#To-Do: prediction mode: 1) random; 2) sequential enumeration; 3) PSO/BO
def random_predict(
        reduced_formula: str,
        Z_start: int,
        Z_end: int,
        space_group_numbers: list[int] = 0,
        n_trial_each_space_group: int = 1,
        n_trail_sites_combination: int = 1,
        sequentially_enumerate_sites_combination: bool = False,
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
    for various space groups with random lattice parameters
    and/or random assignment of Wyockoff sites.
    """
    content = "\n".join(
        [
            f"[{now()}] CrySPR Info: Input chemical formula = {reduced_formula}.",
            f"Input Z_start = {Z_start}, Z_end = {Z_end}.",
            f"Input space groups = {space_group_numbers}",
            f"Number of trails for each space group = {n_trial_each_space_group}",
            f"Number of trail combinations of Wyckoff sites = {n_trail_sites_combination}",
            f"Sequentially enumerate compatible combinations of Wyckoff sites? {'Yes' if sequentially_enumerate_sites_combination else 'No'}",
            f"Use ML-IAP = {relax_calculator.__class__.__name__}",
            f"Use local optimization algorithm = {optimizer.__name__}",
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

    compatible_spg_with_Z = {}  # {Z: [formula, {spg: sites_combinations}]}, sites_combinations: list

    if space_group_numbers == 0:
        spg_list = list(range(1, 231))
    else:
        spg_list = space_group_numbers

    # Enumerate all compatible space groups for each formula
    if Z_end < Z_start:
        Z_end = Z_start
    for Z in range(Z_start, Z_end + 1):
        full_composition: Composition = composition_in.reduced_composition * Z
        full_formula: str = full_composition.formula.replace(" ", "")
        ions_and_numbers: dict = full_composition.get_el_amt_dict()
        species = list(ions_and_numbers.keys())
        number_of_ions = [int(f) for f in list(ions_and_numbers.values())]

        compatible_spg = []
        compatible_spg_with_sites = {}
        for spg_num in spg_list:
            spg = Group(group=spg_num)
            compatible_wyckoff_combinations: list = spg.list_wyckoff_combinations(number_of_ions)[0]
            if len(compatible_wyckoff_combinations) > 0:
                compatible_spg.append(spg_num)
                compatible_spg_with_sites[spg_num] = compatible_wyckoff_combinations
                # log
                number_of_solutions = len(compatible_wyckoff_combinations)
                content = "\n".join(
                    [
                        f"[{now()}] CrySPR Info: Formula {full_formula} is compatible with {spg.symbol} ({spg.number})",
                        f"[{now()}] CrySPR Info: by {number_of_solutions} combinations:",
                        f",\t".join(species),
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
        #
        if len(compatible_spg_with_sites) > 0:
            compatible_spg_with_Z[Z] = [full_formula, compatible_spg_with_sites]
        else:
            pass

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

    # Write header of summary csv
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    with open(f"{wdir}/summary_{reduced_formula_refined}.csv", mode='wt') as f:
        f.write(
            ",".join(
                [
                    "Z",            "SpaceGroupNumber",       "TrialNumber",
                    "WyckoffSites", "SitesCombinationNumber", "EnergyPerAtom",
                ]
            ) + "\n"
        )
    # Generate n_trial_each_space_group structure for each compatible space group
    reservoir: dict = compatible_spg_with_Z.copy()
    for Z in reservoir.keys():
        full_formula_in: str = reservoir[Z][0]
        compatible_spg_with_sites = reservoir[Z][1]
        spgs: list = list(compatible_spg_with_sites.keys())
        spg_trial_comb_atoms = {}  # {spg: {trial: {WyckoffCombination: Atoms}}}
        for spg in spgs:
            count_spg = 1
            spg_trial_comb_atoms[spg] = {}
            while count_spg <= n_trial_each_space_group:
                count_sites_comb = 1
                compatible_wyckoff_combinations: list = compatible_spg_with_sites[spg]
                n_combinations = len(compatible_wyckoff_combinations)
                spg_trial_comb_atoms[spg][f"trail{count_spg}"] = {}
                # Enumerate the compatible combinations of Wyckoff sites
                while count_sites_comb <= n_trail_sites_combination:
                    destination = f"{wdir}/{Z}fu/spg{spg}/trial{count_spg}/sites_combination{count_sites_comb}"
                    if sequentially_enumerate_sites_combination:
                        Wyckoff_sites = compatible_wyckoff_combinations[count_sites_comb]
                        if count_sites_comb > n_combinations:
                            break
                    else:
                        Wyckoff_sites = None

                    output = one_structure_from_one_formula_one_spg(
                        full_formula=full_formula_in,
                        space_group_number=spg,
                        assign_Wyckoff_sites=Wyckoff_sites,
                        random_seed=None,
                        wdir=destination,
                        logfile=logfile,
                        write_cif=write_cif,
                        cif_prefix=cif_prefix,
                        cif_posfix=cif_posfix,
                    )
                    # log
                    content = "\n".join(
                        [
                            f"[{now()}] CrySPR Info: Done structure generation: Space group No. {spg}, Wyckoff sites: {Wyckoff_sites}",
                            f"[{now()}] CrySPR Info: Trial #{count_spg} | sites combination #{count_sites_comb}",
                            f"[{now()}] CrySPR Info: Starting structure relaxation ...",
                            f"[{now()}] CrySPR Info: Logs and structure files in {destination}",
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
                        wdir=destination,
                        logfile_prefix=relax_logfile_prefix,
                        logfile_postfix=relax_logfile_postfix,
                    )
                    energy_per_atom = atoms_relaxd.get_potential_energy() / len(atoms_relaxd)
                    spg_trial_comb_atoms[spg][f"trail{count_spg}"][f"sites_combination{count_sites_comb}"] = atoms_relaxd


                    # log
                    content = "\n".join(
                        [
                            f"[{now()}] CrySPR Info: Done structure relaxation for "
                            f"trial #{count_spg} | sites combination #{count_sites_comb}",
                            f"#{'-'*60}#",
                            f"\n",
                        ]
                    )
                    if verbose:
                        print(content)
                    if logfile != "-":
                        with open(logfile, mode='at') as f:
                            f.write(content)

                    # Write content of summary csv
                    with open(f"{wdir}/summary_{reduced_formula_refined}.csv", mode='at') as f:
                        f.write(
                            ",".join(
                                [
                                    f"{Z}", f"{spg}", f"{count_spg}",
                                    f"\"{Wyckoff_sites}\"", f"{count_sites_comb}", f"{energy_per_atom}",
                                ]
                            ) + "\n"
                        )
                    count_sites_comb += 1 # count up!

                count_spg += 1 # count up!

            # Update the dict to {Z: {spg: {trial: {WyckoffCombination: Atoms}}}}
            reservoir[Z] = spg_trial_comb_atoms

    content = "\n".join(
        [
            f"[{now()}] CrySPR Info: Finished structure generation and relaxation for {reduced_formula}.",
            f"#{'-'*60}#",
            f"\n",
        ]
    )
    if verbose:
        print(content)
    if logfile != "-":
        with open(logfile, mode='at') as f:
            f.write(content)

    return reservoir