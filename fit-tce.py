from pathlib import Path

from ase import io
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
from tce.calculator import TCECalculator


def main():

    pairs = []

    # vacancy hops first!!

    for p in Path("vacancy-hops").iterdir():

        initial, saddle, final = io.read(p / "neb-tce-space.xyz", format="extxyz", index=slice(None))

        neb_trajectory = io.read(p / "neb.xyz", format="extxyz", index=slice(None))
        energies = np.fromiter(
            (image.get_potential_energy() for image in neb_trajectory),
            dtype=float
        )

        initial.calc = SinglePointCalculator(initial, energy=energies[0])
        saddle.calc = SinglePointCalculator(saddle, energy=energies.max())
        final.calc = SinglePointCalculator(final, energy=energies[-1])

        pairs.append((initial, saddle))
        pairs.append((final, saddle))

    # from these pairs, we can delta-train a tce model
    lattice_parameter = np.cbrt(bulk("W", cubic=True).get_volume())
    # cutoffs and many body features from table 1 in https://doi.org/10.1016/j.commatsci.2026.115142
    calc = TCECalculator(
        neighbor_cutoffs=lattice_parameter * np.array([
            0.43, 0.50, 0.71, 0.83, 0.87, 1.0, 1.09, 1.12
        ]),
        many_body_features=[
            (0, 0, 1),
            (0, 0, 2),
            (1, 1, 2),
            (2, 2, 2)
        ],
        species=["W", "V", "X"]
    ).difference_train(pairs)

    for initial, saddle in pairs:

        actual_energy_diff = saddle.get_potential_energy() - initial.get_potential_energy()

        initial_cluster_counts = calc.get_feature_vector(initial)
        saddle_cluster_counts = calc.get_feature_vector(saddle)

        predicted_energy_diff = calc.models["energy"].predict(saddle_cluster_counts - initial_cluster_counts)
        print(actual_energy_diff, predicted_energy_diff)


if __name__ == "__main__":

    main()
