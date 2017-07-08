#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 09:55:40 2017

@author: etienne
This script will open a log.lammps output from LAMMPS and output a pandas dataframe

Usage:
# This script must be in PYTHONPATH or working directory
from thermo_dataframe import read_thermo
thermo_df = open_thermo(fname="log.lammps")

Requirement:
numpy
pandas

Limitations:
The thermo_style must start with Step
The last line of the thermo output must be "Loop time"

TODO
Multiple dataframe if there are multiple runs in the log.lammps
"""
import numpy as np
import pandas as pd


def read_thermo(fname="log.lammps"):
    """Function to read log.lammps
    Input: log.lammps file
    Output: a pandas dataframe with step as index and the thermo_style as columns"""
    # Opens the file
    f = open(fname, "r")
    get_thermo = False
    thermo_data = []
    for line in f:
        if line.startswith("Step"):
            get_thermo = True
            # Use this line for the column name
            columns_name = line.split()
            line = next(f)
        if line.startswith("Loop time") or line.startswith("WARNING"):
            get_thermo = False
        if get_thermo == True:
            thermo_data.append(line.split())

    # convert to a pandas data frame
    thermo_df = pd.DataFrame(thermo_data, columns=columns_name)
    thermo_df = thermo_df.apply(pd.to_numeric)
    thermo_df = thermo_df.set_index(["Step"])
    return thermo_df
