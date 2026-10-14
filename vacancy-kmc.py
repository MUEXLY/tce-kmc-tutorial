from warnings import warn
from pathlib import Path
import json

from ase import Atoms, io
from ase.build import bulk
from tce.calculator import TCECalculator
import numpy as np


RNG: np.random.Generator = np.random.default_rng(seed=0)
LOG_PATH: Path = Path("log.jsonl")
DUMP_DIR: Path = Path("kmc")
NUM_STEPS: int = 1_000
DUMP_EVERY: int = 10
BOLTZMANN_CONSTANT: float = 8.617e-5
TEMPERATURE: float = 1000
RATE_PREFACTOR: float = 1.0e+13


def main():

    beta = 1.0 / (BOLTZMANN_CONSTANT * TEMPERATURE)
    DUMP_DIR.mkdir(exist_ok=True)

    multilattice_basis = Atoms(
        positions=[
            [0.00, 0.00, 0.00],
            [0.50, 0.50, 0.50],
            [0.25, 0.25, 0.25],
            [0.75, 0.25, 0.25],
            [0.25, 0.75, 0.25],
            [0.25, 0.25, 0.75],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
            [0.75, 0.75, 0.75]
        ],
        cell=[1.0, 1.0, 1.0],
        symbols=["W"] * 2 + ["X"] * 8,
        pbc=True
    )
    tungsten_lattice_parameter = np.cbrt(bulk("W", cubic=True).get_volume())

    pure_tungsten = multilattice_basis.repeat((4, 4, 4))
    pure_tungsten.positions *= tungsten_lattice_parameter
    pure_tungsten.cell *= tungsten_lattice_parameter
    bcc_sites_mask, = np.where(pure_tungsten.symbols == "W")

    configuration = pure_tungsten.copy()
    vanadium_fraction = RNG.uniform()
    configuration.symbols[bcc_sites_mask] = RNG.choice(
        ["W", "V"], 
        p=[1.0 - vanadium_fraction, vanadium_fraction], 
        size=len(bcc_sites_mask)
    )
    vacancy_index = RNG.choice(bcc_sites_mask)
    configuration.symbols[vacancy_index] = "X"

    calc = TCECalculator.load("WVX.pkl")
    calc.get_feature_vector(configuration)

    topological_tensors, = calc.topological_tensors.values()
    adjacency_tensors = topological_tensors[2]
    bcc_coord_nums = adjacency_tensors[4, :, :].sum(axis=0).todense()
    assert np.mean(bcc_coord_nums) == 8.0 and np.std(bcc_coord_nums) == 0.0

    with open(LOG_PATH, "w") as log_file:

        t = 0.0
        for step in range(1, NUM_STEPS + 1):

            info_dict = {
                "time": t,
                "vacancy position": configuration.positions[vacancy_index].tolist(),
                "W center": configuration.positions[configuration.symbols == "W"].mean(axis=0).tolist(),
                "V center": configuration.positions[configuration.symbols == "V"].mean(axis=0).tolist()
            }

            print(json.dumps(info_dict), file=log_file)

            if not step % DUMP_EVERY:
                io.write(DUMP_DIR / f"{step:0d}.xyz", configuration, format="extxyz")
                print(f"{step / NUM_STEPS:.2%}", end="\r")
                log_file.flush()

            # build events by permuting vacant index and all of its neighbors
            neighboring_saddle_points, = np.where(adjacency_tensors[0, vacancy_index, :])

            """
            chained tensor product to find the corresponding atomic sites
            i.e., any move looks like along the <111>:
            X X W (initial)
            X W X (transition)
            W X X (final)

            in the initial configuration, we know the two X positions
            the final W position is then the site connected to the saddle via the 1nn bond, and connected to the far left via the 5nn bond
            """

            neighboring_atomic_points = []
            for saddle_point in neighboring_saddle_points:
                neighboring_atomic_point, = np.where(
                    adjacency_tensors[0, vacancy_index, saddle_point] * adjacency_tensors[0, saddle_point, :] * adjacency_tensors[4, vacancy_index, :]
                )
                neighboring_atomic_points.append(neighboring_atomic_point.item())
            
            # each event is then a permutation
            cluster_count_differences = np.zeros((8, calc.feature_vector_size), dtype=float)
            for i, (saddle, atom) in enumerate(zip(neighboring_saddle_points, neighboring_atomic_points, strict=True)):

                transition = configuration.copy()
                transition.symbols[[saddle, atom]] = transition.symbols[[atom, saddle]]
                cluster_count_differences[i, :] = calc.get_feature_vector_difference(initial=configuration, final=transition)

            barriers = calc.models["energy"].predict(cluster_count_differences)

            if np.any(barriers < 0):
                warn("Some negative barriers! Maybe overfitted model")
            rates = RATE_PREFACTOR * np.exp(-beta * barriers)
            probabilites = rates / rates.sum()

            atom_to_swap = RNG.choice(neighboring_atomic_points, p=probabilites)
            configuration.symbols[[vacancy_index, atom_to_swap]] = configuration.symbols[[atom_to_swap, vacancy_index]]
            vacancy_index = atom_to_swap

            t += RNG.exponential(scale=1.0 / rates.sum())


if __name__ == "__main__":

    main()
