from pathlib import Path

from ase import io
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from tce.calculator import TCECalculator
import matplotlib.pyplot as plt


def main():

    pairs = []

    # vacancy hops first!!

    for p in Path("vacancy-hops").iterdir():
        try:
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
        except FileNotFoundError:
            continue

    pairs_train, pairs_test = train_test_split(pairs, test_size=0.2, random_state=0)

    # from these pairs, we can delta-train a tce model
    # cutoffs and many body features from table 1 in https://doi.org/10.1016/j.commatsci.2026.115142
    lattice_parameter = np.cbrt(bulk("W", cubic=True).get_volume())
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
        species=["W", "V", "X"],
        models={
            "energy": Pipeline([
                ("scale", StandardScaler(with_mean=False)),
                ("fit", RidgeCV(
                    fit_intercept=False,
                    alphas=np.logspace(-3, 3, 100),
                    store_cv_results=True
                ))
            ])
        }
    ).difference_train(pairs_train)

    actual = []
    predicted = []
    for initial, saddle in pairs_train:

        actual_energy_diff = saddle.get_potential_energy() - initial.get_potential_energy()

        initial_cluster_counts = calc.get_feature_vector(initial)
        saddle_cluster_counts = calc.get_feature_vector(saddle)
        diff = saddle_cluster_counts - initial_cluster_counts
        predicted_energy_diff = calc.models["energy"].predict(diff.reshape(1, -1))

        actual.append(actual_energy_diff)
        predicted.append(predicted_energy_diff)
    plt.scatter(actual, predicted, label="train", edgecolor="black", zorder=7)

    actual = []
    predicted = []
    for initial, saddle in pairs_test:
    
        actual_energy_diff = saddle.get_potential_energy() - initial.get_potential_energy()

        initial_cluster_counts = calc.get_feature_vector(initial)
        saddle_cluster_counts = calc.get_feature_vector(saddle)
        diff = saddle_cluster_counts - initial_cluster_counts
        predicted_energy_diff = calc.models["energy"].predict(diff.reshape(1, -1))

        actual.append(actual_energy_diff)
        predicted.append(predicted_energy_diff)
    plt.scatter(actual, predicted, label="test", edgecolor="black", zorder=8)

    xlim = plt.gca().get_xlim()
    plt.plot(xlim, xlim, linestyle="--", color="black", zorder=6)

    plt.xlabel(r"actual $\Delta E$ (eV)")
    plt.ylabel(r"predicted $\Delta E$ (eV)")
    plt.legend()
    plt.grid()
    plt.savefig("plots/parity.pdf", bbox_inches="tight")

    # clear the cache that is created in the calculator
    calc.topological_tensors = {}
    calc.save("WVX.pkl")


if __name__ == "__main__":

    main()
