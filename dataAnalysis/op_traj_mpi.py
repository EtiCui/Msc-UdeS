#!/usr/bin/python
# -*- coding: utf-8 -*-
""" This script will a LAMMPS trajectory in parallel with MPI4PY and calculate the nematic,SmA order parameters.

Usage:
#change the arguments in the open_trajectory function
mpirun -np NOMBER_OF_PROCS op_traj_mpi.py

Requirement:
python2.7
numpy
mpi4py
pandas
nematic_sma_OP.py

"""
import numpy as np
from glob import glob
from mpi4py import MPI
from nematic_sma_OP import PO

rank = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()
first_frame = -1000
last_frame = -1


def open_trajectory(nprocs, rank, first_frame, last_frame, wrap=True, visualize=True, ini_layer_spacing=35., 
                    gb_type=3, gb_ends_type=2, atoms_per_monomer=23, number_of_monomer=800, number_of_chains=100):
    """
    This function will open a LAMMPS trajectory in parallel to calculate the SmA and nematic order parameters.
    Each frames are considered independent, and the final results are transmitted to the processor with rank=0

    Args:
    ----
    nprocs(int): number of processor (read from mpirun command)
    rank(int): rank of the process
    first_frame(int): the first frame of the trajectory
    last_frame(int): the last frame of the trajectory
    wrap(bool): True if the coordinates are to be wrapped
    visualize(bool): True if the gz12 in function of z12 graph for the SmA OP is desired(3X slower)
    ini_layer_spacing(float): SmA layer spacing to optimize
    gb_type(int): atom type of the ellipsoid
    gb_ends_type(int): atom type of the pseudo-atoms at the end of the ellipsoids
    atoms_per_monomer(int): atoms per monomer in the chains
    number_of_monomer(int): total number of monomer in the system
    number_of_chains(int): number of polymer chains in the system

    Returns:
    ----
    nematic_OP.out(text file): a file with the timestep and the calculated nematic OP
    sma_OP.out(text file): a file with the timestep, the SmA OP and the optimized layer spacing
    """
    # create a list of all the files in the trajectory
    complete_trajectory = glob("*dump*")
    # sort the list
    complete_trajectory.sort(key=lambda f: int(filter(str.isdigit, f)))

    # consider only the desired frames
    desired_trajectory = complete_trajectory[first_frame:last_frame]

    # Divide the trajectory by the number of rank
    fragment_trajectory = np.array_split(desired_trajectory, nprocs)

    # select a fragment of the trajectory for each rank
    for trajectory in np.nditer(fragment_trajectory[rank][:], flags=['external_loop']):
        steps_nematic_OP = []
        steps_sma_OP_distance = []
        for dump in trajectory:
            po = PO(dump, wrap, visualize, ini_layer_spacing, gb_type, gb_ends_type,
                    atoms_per_monomer, number_of_monomer, number_of_chains)
            # nematic
            step, nematic_OP, director = po.nematic()
            steps_nematic_OP.append([step, nematic_OP])
            # sma
            step, sma_OP, distance = po.sma()
            steps_sma_OP_distance.append([step, sma_OP, distance])
        print("Rank: ", rank, " has finished")

    MPI.COMM_WORLD.barrier()
    # the processor with rank=0 gather the calculated OP
    steps_nematic_OP = MPI.COMM_WORLD.gather(steps_nematic_OP, root=0)
    MPI.COMM_WORLD.barrier()
    steps_sma_OP_distance = MPI.COMM_WORLD.gather(
        steps_sma_OP_distance, root=0)
    MPI.COMM_WORLD.barrier()

# rank=0 processor writes the output
    if rank == 0:
        steps_nematic_OP = np.concatenate(steps_nematic_OP)

        steps_nematic_OP = steps_nematic_OP[steps_nematic_OP[:, 0].argsort()]
        np.savetxt('nematic_OP.out', steps_nematic_OP)

        steps_sma_OP_distance = np.concatenate(steps_sma_OP_distance)
        steps_sma_OP_distance = steps_sma_OP_distance[steps_sma_OP_distance[:, 0].argsort(
        )]
        np.savetxt('sma_OP.out', steps_sma_OP_distance)


open_trajectory(nprocs, rank, first_frame, last_frame)
