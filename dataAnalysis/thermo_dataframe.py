#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 09:55:40 2017

@author: etienne
This function will open a log.lammps output and returns a pandas dataframe

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

def open_thermo(fname="log.lammps"):
    """Function to read the log.lammps
    Returns: a pandas dataframe with step as index and the thermo_style as columns"""
    #Opens the file
    f = open(fname, "r")
    get_thermo = False
    thermo_data = []
    for line in f:
        if line.startswith("Step"):
            get_thermo = True
            # Use this line for the column name
            columns_name = line.split()
            line = next(f)
        if line.startswith("Loop time"):
            get_thermo=False
        if get_thermo == True:
            thermo_data.append(line.split())

    # convert to a pandas data frame
    thermo_df = pd.DataFrame(thermo_data, columns=columns_name)
    thermo_df = thermo_df.apply(pd.to_numeric)
    thermo_df = thermo_df.set_index(["Step"])
    return thermo_df
