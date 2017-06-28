#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:11:22 2016

@author: etienne
This function will read a LAMMPS compressed dump (ex:dump.step.gz) and return the step, cell dimensions (x,y,z),
number of atoms and atoms informations as a dictionnary. The atoms information is a pandas dataframe with the
atom attributes as columns and id as index

Usage:
# This script must be in PYTHONPATH or working directory
from dump_dataframe import read_thermo
dump = read_dump(fname="dump.500.gz")
step = dump["Step"]
nb_atoms = dump["nb_atoms"]
dimensions = dump["dimensions"]
atom_df = dump["atom_df"]

Requirement:
pandas

Limitations:
Must be only one step per dump
id must an element of the dump

TODO:
Adding a function if there are multiple steps per dump
"""

import gzip
import pandas as pd


def read_dump(fname):
    """Function to read dump from LAMMPS
    Input: filename of dump
    Output: Dictionnary with {"Step":timestep,"nb_atoms":nb_atoms,"dimensions":dimensions,"atom_df":atom_df}
    where dimensions is a list of the cell dimensions (x,y,z) and atom_df a pandas dataframe with the atoms
    attributes as columns
    """
    f = gzip.open(fname, "rt")
    get_atoms = False
    atoms_data = []
    for line in f:
        # get timestep
        if line.startswith("ITEM: TIMESTEP"):
            timestep = int(next(f))
        # get number of atoms
        if line.startswith("ITEM: NUMBER OF ATOMS"):
            nb_atoms = int(next(f))
        # get box dimension
        if line.startswith("ITEM: BOX BOUNDS pp pp pp"):
            # x dimension
            x_bound = [float(x) for x in next(f).split(" ")]
            x_dimension = x_bound[1] - x_bound[0]
            # y dimension
            y_bound = [float(x) for x in next(f).split(" ")]
            y_dimension = y_bound[1] - y_bound[0]
            # z dimension
            z_bound = [float(x) for x in next(f).split(" ")]
            z_dimension = z_bound[1] - z_bound[0]
            dimensions = [x_dimension, y_dimension, z_dimension]
        if line.startswith("ITEM: ATOMS"):
            # ignore ITEM: ATOMS
            columns_name = line.split()[2:]
            get_atoms = True
            line = next(f)
        # get atoms information
        if get_atoms == True:
            atoms_data.append(line.split())

    # convert to a pandas data frame
    atom_df = pd.DataFrame(atoms_data, columns=columns_name)
    atom_df = atom_df.apply(pd.to_numeric)
    atom_df = atom_df.set_index(["id"])    
    return {"step": timestep, "nb_atoms": nb_atoms, "dimensions": dimensions, "atom_df": atom_df}
