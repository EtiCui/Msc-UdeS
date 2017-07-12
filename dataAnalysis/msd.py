#!/usr/bin/python
""" Functions to calculate the mean-square displacement from a LAMMPS trajectory

Usage:
#Must be in pythonpath or working directory
from msd import msd
msd_df = msd(atom_type,first_frame,last_frame)

Requirement:
python2
numpy
dump_dataframe.py
pandas

TODO:
Parallelisation
Add a function for a trajectory in a single file
"""
from dump_dataframe import read_dump
import numpy as np
import pandas as pd
from glob import glob


def msd(atom_type=3, first_frame=-1000, last_frame=-1):
    """ Function to calculate the mean-square displacement(in each direction and the total msd)
        of a trajectory. Reads all the dump to create an array with the time evolution of
        the positions for each particles of an atom_type

    Args:
    ----
    atom_type(int): The atom type of the desired atoms to calculate the msd_df
    first_frame(int): The first frame to start the msd
    last_frame(int): The last frame for the msd

    Returns:
    ----
    msd(dataframe): An dataframe with the time as index, msd x,msd y,msd z and total as columns

    """
    # List of all the dump in the trajectory
    complete_trajectory = glob("*dump*")

    # sort the list according to the number in the filename
    complete_trajectory.sort(key=lambda f: int(filter(str.isdigit, f)))

    # consider only the desired frames
    desired_trajectory = complete_trajectory[first_frame:last_frame]

    # Initialize the lists for the positions and timestep
    x = []
    y = []
    z = []
    timesteps = []

    for step in desired_trajectory:
        # read the dump for each steps
        dump = read_dump(step, wrap=False)
        timestep = dump["step"]
        atom_df = dump["atom_df"]

        # select only the usefull columns
        msd_col_list = ["type", "xu", "yu", "zu"]
        msd_df = atom_df[msd_col_list]

        # choose only the wanted atom_type
        msd_df = msd_df[msd_df["type"] == atom_type]
        # drop the now useless type column
        msd_df = msd_df.drop(["type"], axis=1)

        # append each values to the list
        timesteps.append(timestep)
        x.append(msd_df.xu.values.tolist())
        y.append(msd_df.yu.values.tolist())
        z.append(msd_df.zu.values.tolist())

    # Convert list to arrays and transpose them, so the lines will be each particles
    # and the columns the steps
    timesteps = np.array(timesteps).T
    x = np.array(x).T
    y = np.array(y).T
    z = np.array(z).T

    msd = []
    n = 1
    while n < len(desired_trajectory):
        # calculate the delta_t
        delta_t = timesteps[n] - timesteps[0]

        # calculate (x(t+n)-x(t))**2 and the mean over all the particles and
        # the same delta_t
        x_diff = x[:, n:] - x[:, :-n]
        msd_x = np.mean(x_diff**2)

        y_diff = y[:, n:] - y[:, :-n]
        msd_y = np.mean(y_diff**2)

        z_diff = z[:, n:] - z[:, :-n]
        msd_z = np.mean(z_diff**2)

        msd.append([delta_t, msd_x, msd_y, msd_z, msd_x + msd_y + msd_z])
        n += 1
    msd = np.array(msd)
    msd_df = pd.DataFrame(msd[:, 1:], index=msd[:, 0],
                          columns=["x", "y", "z", "total"])
    msd_df.index.name = "temps"
    return msd_df
