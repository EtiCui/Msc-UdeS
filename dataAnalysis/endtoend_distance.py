#!/usr/bin/python
"""
This function will read a lammps output to compute the end to end distances (Rf if averaged) of the backbone.

Usage:
# dump_dataframe must be in pythonpath or working directory
from endtoend_distance import rf
distances = rf(filename,atoms_per_polyer,number_of_chains)

Requirement:
numpy
pandas
dump_dataframe.py
scipy

Limitations:
Coordinates must be unwrapped (ex:xu,yu,zu)

TODO:
Function to read a trajectory from a single file
"""

from dump_dataframe import read_dump
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

def rf(filename,atoms_per_polymer=184,number_of_chains=100):
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
    #Read the dump, coordinates must be unwrapped
    dump = read_dump(filename, wrap=False)
    #Select only the useful columns
    wanted_columns = ["xu","yu","zu"]
    rf_df = dump["atom_df"][wanted_columns]
    #Create an empty array which will contains the distances
    endtoend_dists=np.zeros(number_of_chains)

    i=0
    while i < number_of_chains:
        #Calculate the distance betwenn the fist and the last atoms in the backbone
        endtoend_dists[i]=pdist(rf_df.loc[[1+atoms_per_polymer*i,atoms_per_polymer+atoms_per_polymer*i]])
        i+=1
    return endtoend_dists
rf()
