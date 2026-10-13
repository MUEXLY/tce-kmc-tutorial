"""
This script runs the NEBs necessary to fit a TCE model
Each NEB will generate a directory that looks like:
vacancy-hops/
    */
        neb-tce-space.xyz
        neb.xyz
        relax-final.traj
        relax-initial.traj
        rxn.pdf

neb-tce-space.xyz is the hop represented in cluster space
neb.xyz is the hop represented in real space
relax-*.traj are the relaxation trajectories before we run the NEB (i.e. we need to relax the endpoints)
rxn.pdf is a reaction coordinate diagram
"""

from pathlib import Path

from ase import Atoms, io
from ase.calculators.eam import EAM
from ase.build import bulk
from ase.geometry import find_mic
from ase.filters import UnitCellFilter
from ase.optimize import BFGS
from ase.mep import NEB
import numpy as np
import matplotlib.pyplot as plt


# just so everything is deterministic
RNG: np.random.Generator = np.random.default_rng(seed=0)

# some basic settings. you probably want tighter convergence criteria here
# ase's optimization criteria is the maximum force in the configuration being below fmax
# units are eV/A
NUM_HOPS: int = 100
NUM_NEB_IMAGES: int = 10
RELAX_FMAX_TOL: float = 0.05
NEB_FMAX_TOL: float = 0.1

# these are specific to bcc, hop directions are <111> and the <111> spacing is sqrt(3)/2
HOP_DIRECTIONS: list[tuple[int, int, int]] = [
    (+1, +1, +1),
    (-1, +1, +1),
    (+1, -1, +1),
    (+1, +1, -1),
    (-1, -1, +1),
    (-1, +1, -1),
    (+1, -1, -1),
    (-1, -1, -1)
]
INTERPLANAR_SPACING: float = 0.5 * np.sqrt(3.0)


# define the calculator you want to use. for a more complex interatomic potential, probably hook up to LAMMPS
# or use a third party library that hooks into ASE, e.g. mace-torch
def get_calculator():

    return EAM(potential="WV.eam.fs")


def closest_atom(atoms, position):
    """Return (index, distance) of the atom nearest to position, accounting for PBC."""
    displacements = atoms.positions - np.asarray(position)
    _, distances = find_mic(
        displacements, atoms.cell, pbc=atoms.pbc
    )
    index = np.argmin(distances)
    return index


def main():

    # basis for the bcc multilattice with ghost X atoms at (1/2, 1/2, 1/2) + (\pm 1/4, \pm 1/4, \pm 1/4) lattice points
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

    # create bulk tungsten from the basis described above
    tungsten_lattice_parameter = np.cbrt(bulk("W", cubic=True).get_volume())

    pure_tungsten = multilattice_basis.repeat((4, 4, 4))
    pure_tungsten.positions *= tungsten_lattice_parameter
    pure_tungsten.cell *= tungsten_lattice_parameter
    bcc_sites_mask, = np.where(pure_tungsten.symbols == "W")

    for hop_index in range(NUM_HOPS):

        # we want to store all the simulation results for the given hop in that directory
        hop_directory = Path("vacancy-hops") / f"{hop_index:0d}"
        hop_directory.mkdir(exist_ok=True, parents=True)

        # randomly alloy
        # you can make an ordered sample here too
        configuration = pure_tungsten.copy()
        vanadium_fraction = RNG.uniform()
        configuration.symbols[bcc_sites_mask] = RNG.choice(
            ["W", "V"], 
            p=[1.0 - vanadium_fraction, vanadium_fraction], 
            size=len(bcc_sites_mask)
        )

        # pick a random vacancy and a random <111> direction to hop
        vacancy_index = RNG.choice(bcc_sites_mask)
        configuration.symbols[vacancy_index] = "X"

        hop_direction = RNG.choice(HOP_DIRECTIONS)
        hop_direction_array = np.array(hop_direction, dtype=float)
        hop_direction_array /= np.linalg.norm(hop_direction_array)

        """
        this is kind of the meat of the code

        we want to pick a neighbor for the vacancy to hop into
        we can then permute the positions to create this new configuration

        along the <111>, the initial configuration looks like (in cluster space):
        X X W (initial)
        X W X (transition)
        W X X (final)

        we picked the X index on the far left already, now we need to find the two sites to its right
        then, once we find these sites, we can permute
        """

        hopping_atom_index = closest_atom(
            configuration,
            position=configuration.positions[vacancy_index] + \
                INTERPLANAR_SPACING * tungsten_lattice_parameter * hop_direction_array
        )
        assert configuration.symbols[hopping_atom_index] != "X"
        saddle_index = closest_atom(
            configuration,
            position=configuration.positions[vacancy_index] + \
                0.5 * INTERPLANAR_SPACING * tungsten_lattice_parameter * hop_direction_array
        )
        assert configuration.symbols[saddle_index] == "X"

        saddle = configuration.copy()
        saddle.positions[[hopping_atom_index, saddle_index]] = saddle.positions[[hopping_atom_index, vacancy_index]]
        assert saddle.symbols[hopping_atom_index] != "X" and saddle.symbols[vacancy_index] == "X"

        final = configuration.copy()
        final.positions[[hopping_atom_index, vacancy_index]] = saddle.positions[[vacancy_index, hopping_atom_index]]
        assert final.symbols[hopping_atom_index] != "X" and final.symbols[vacancy_index] == "X"

        # now we can write the trajectory in cluster space
        io.write(hop_directory / "neb-tce-space.xyz", [configuration, saddle, final], format="extxyz")

        """
        in real space, this is a bit more annoying
        we perform the hop instead by just moving the hopping atom into the old vacant position
        """

        initial_relaxable = configuration.copy()
        vacancy_position = initial_relaxable.positions[vacancy_index]
        hopping_atom_position = initial_relaxable.positions[hopping_atom_index]

        initial_relaxable = initial_relaxable[[i for i, atom in enumerate(configuration) if atom.symbol != "X"]]
        # this reindexes, so we need to find the index of the hopping atom again
        hopping_atom_index = closest_atom(initial_relaxable, hopping_atom_position)

        # now, relax!
        # the relaxation will be saved to a .traj file which is not readable with OVITO basic
        # you can convert it with ASE, see https://docs.ase-lib.org/ase/io/io.html#module-ase.io
        initial_relaxable.calc = get_calculator()
        BFGS(
            UnitCellFilter(initial_relaxable), 
            trajectory=str(hop_directory / "relax-init.traj")
        ).run(fmax=RELAX_FMAX_TOL)

        final_relaxable = initial_relaxable.copy()
        final_relaxable.calc = get_calculator()

        """
        CAUTION!! this might not be robust. The relaxation above changes the vacancy's position
        and the assumption is that the new vacancy position is close enough to the old one's, such that the relaxation finds it

        maybe you'd want to use something more robust here, e.g. polyhedral template matching to find the vacancy's center
        """
        final_relaxable.positions[hopping_atom_index] = vacancy_position

        # don't want to modify volume for the final relaxation, so no filter
        BFGS(
            final_relaxable,
            trajectory=str(hop_directory / "relax-final.traj")
        ).run(fmax=RELAX_FMAX_TOL)

        # now run neb between these two endpoints
        images = [initial_relaxable]
        images += [initial_relaxable.copy() for _ in range(NUM_NEB_IMAGES - 2)]
        images += [final_relaxable]
        neb = NEB(images)
        neb.interpolate(mic=True)

        for image in images[1:-1]:
            image.calc = get_calculator()

        optimizer = BFGS(neb)
        optimizer.run(fmax=NEB_FMAX_TOL)

        io.write(hop_directory / "neb.xyz", images, format="extxyz")

        energies = np.fromiter(
            (image.get_potential_energy() for image in images),
            dtype=float
        )
        forward_barrier = energies.max() - energies[0]
        backward_barrier = energies.max() - energies[1]

        rxn_progress = np.linspace(0.0, 100.0, len(energies))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(rxn_progress, energies - energies[0])
        ax.grid()
        ax.set_xlabel("rxn progress (%)")
        ax.set_ylabel("energy (eV)")
        ax.set_title(f"forward = {forward_barrier:.2f} eV, backward = {backward_barrier:.2f} eV")
        fig.savefig(hop_directory / "rxn.pdf")
        plt.close(fig)


if __name__ == "__main__":

    main()
