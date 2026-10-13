# KMC with TCE as a driver

Below is a tutorial for using `ase`, `tce-lib`, and `scikit-learn` in tandem to run KMC for vacancy hopping

## environment setup

I recommend you use `uv` for the environment setup here. See [here](https://docs.astral.sh/uv/getting-started/installation/) for how to set this up. Then, simply clone and sync:

```
git clone tce-kmc-tutorial
cd tce-kmc-tutorial
uv sync
```

## running vacancy hops

You can then run the vacancy hops using the W-V [embedded atom method](https://en.wikipedia.org/wiki/Embedded_atom_model) (EAM) potential included in the repository, wrapped with a `slurm` script:

```
sbatch vacancy-neb.slurm
```

This script generates a bunch of vacancy hops in TCE space (following our paper [here](https://www.sciencedirect.com/science/article/pii/S0927025626006610)), and computes the reaction coordinate diagram using that EAM potential.

This script is pretty modular for different bcc systems. You can change the calculator in these lines:

```py
from ase.calculators.eam import EAM

def get_calculator():

    return EAM(potential="WV.eam.fs")
```

to whatever else `ase` supports, which has documentation [here](https://ase.gitlab.io/ase/ase/calculators/calculators.html). There are plenty of third-party libraries that would work here too, e.g. `mace-torch`, with `ase`-specific documentation [here](https://mace-docs.readthedocs.io/en/latest/guide/ase.html).

**CAUTION**: This step is honestly quite slow, and will be even slower if you tighten the convergence criteria and/or increase the number of NEB images. I'm sure it could be parallelized (each NEB run is independent of any others), but this is likely to be a pain for an expensive potential. I've found that `ase`'s implementation of NEB is practically much slower than, say, LAMMPS's (which is not super surprising), but using `ase` is the only way to keep this modular. It might be of interest to copy this workflow into another application separately.

## fitting a model

You can then $\Delta$-train a TCE model by running the fitting script:

```
uv run fit-tce.py
```

The important part of the script here is the portion defining the calculator:

```py
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
```

You can define more many-body features here (including 4-body features by specifying 6 edges), or change the model that you use to fit the data. See [here](https://muexly.github.io/tce-lib/tce.html#custom-training-advanced) for documentation on how to do this using `tce-lib`. It's likely that the default model will overfit, so I strongly recommend using some regularization technique, e.g. `RidgeCV`, `LassoCV`, or `ElasticNetCV` from `scikit-learn`.

## running kmc

TODO