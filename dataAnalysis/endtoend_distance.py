#!/usr/bin/python
"""
These function can be used to calculate the average end to end distances of a backbone from a lammmps output.

Usage:
# dump_dataframe must be in pythonpath or working directory
from endtoend_distance import rf
rf,rf_std = rf(first_frame=-1000, last_frame=-1, trajectory_step=10,atoms_per_polymer=184, number_of_chains=100)

Requirement:
numpy
pandas
dump_dataframe.py
scipy

Limitations:
Coordinates must be unwrapped (ex:xu,yu,zu)
Each dump must be a file

TODO:
Function to read a trajectory from a single file
"""

from dump_dataframe import read_dump
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from glob import glob

def endtoend(filename, atoms_per_polymer, number_of_chains):
    """
    Function to calculate the end to end distances of each polymer chains from a dump.

    Args:
    ----
    filename(string): Filename of the dump
    atoms_per_polymer(int): The number of particles/atoms in a single chains
    number_of_chains(int): Number of chains in the system

    Returns:
    ----
    endtoend_dists(array): Numpy array with the end-to-end distance for each chains
    """
    # Read the dump, coordinates must be unwrapped
    dump = read_dump(filename, wrap=False)
    # Select only the useful columns
    wanted_columns = ["xu", "yu", "zu"]
    rf_df = dump["atom_df"][wanted_columns]
    # Create an empty array which will contains the distances
    endtoend_dists = np.zeros(number_of_chains)

    i = 0
    while i < number_of_chains:
        # Calculate the distance between the fist and the last atoms in the
        # backbone
        endtoend_dists[i] = pdist(
            rf_df.loc[[1 + atoms_per_polymer * i, atoms_per_polymer + atoms_per_polymer * i]])
        i += 1
    return endtoend_dists


def rf(first_frame=-1000, last_frame=-1, trajectory_step=10,atoms_per_polymer=184, number_of_chains=100):
    """
    Function to calculate the Rf of a lammps trajectory.

    Args:
    ----
    first_frame(int): The first frame desired in the trajectory
    last_frame(int): The frame to stop
    trajectory_step(int): calculate only for each # of files
    atoms_per_polymer(int): The number of atoms in the polymer chain
    number_of_chains(int): The number of chains in the system

    Returns:
    ----
    Rfmean(float): The average end to end distances in the trajectory
    Rfstd(float): The standard deviation of the Rf
    """
    # List of all the dump in the trajectory
    complete_trajectory = glob("*dump*")

    # sort the list according to the number in the filename
    complete_trajectory.sort(key=lambda f: int(filter(str.isdigit, f)))

    # consider only the desired frames
    desired_trajectory = complete_trajectory[first_frame:last_frame:trajectory_step]

    #create a empty numpy array to contains the end to end distances for each chain (columns)
    #for each step (time)
    rf = np.zeros((len(desired_trajectory),number_of_chains))
    i=0
    # for each file in the trajectory
    for f in desired_trajectory:
        #calculate the end to end distances for each chain
        rf[i] = endtoend(f, atoms_per_polymer, number_of_chains)
        i+=1
    #return the mean average distances with its standard deviation
    return rf.mean(),rf.std()
