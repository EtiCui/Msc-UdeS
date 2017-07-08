#!/usr/bin/python
"""
These function will read LAMMPS output (dihedral.#step.out) of the dihedral angles (column1:#, column2:angle value),
to plot an histogram of the angle's frequencies.
Works in parallel with mpi4py, altough it's very fast without parallelisation.
A


Usage:
#change first and last desired step in the script
python dihedral_histogram.py

Requirement:
python2.7
numpy
matplotlib
mpi4py
pandas

TODO
Read a trajectory in a single file
"""
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from glob import glob
import pandas as pd

rank = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()
#first desired step
first=-50
#last desired step
last=-1

def open_dihedral_file(fname):
    """Function to load a dihedral output from lammps (name: dihedral.STEP.out).

    Args:
    ----
    fname(string): filename of the output (ex:dihedral.timestep.out)
                   The output must countains the second column as the dihedral angle

    Returns:
    ----
    dihedral_data(array): an array of all the dihedral angle
    """
    f = open(fname, "r")
    get_atoms = False
    dihedral_data = []
    for line in f:
        if line.startswith("ITEM: ENTRIES"):
            get_atoms = True
            line = next(f)
        # get the index and dihedral
        if get_atoms == True:
            number, dihedral_angle = (int(line.split()[0]),
                                      float(line.split()[1]))

            dihedral_data.append(dihedral_angle)
    #only returns the angle
    return dihedral_data

def get_histogram_dihedral(fname=None,dihedral_data=None,histogram_bins=np.arange(-180,181)):
    """ This function creates a numpy array with the frequencies of occurence for each angle

    Args:
    ----
    fname(string): filename (optional)
    dihedral_data(array): array of all the dihedral angles (optional if fname is given)
    histogram_bins(array): bins for the histogram(default -180 to 180 with increment of 1)

    Returns:
    ----
    histogram_bins(array):bins used for the histogram_bins
    dihedral_histogram(array): Occurence of each angle
    """
    if dihedral_data == None:
        dihedral_data = open_dihedral_file(fname)
    #create a histogram for the dihedral angle
    dihedral_histogram,bins = np.histogram(dihedral_data,histogram_bins)
    return histogram_bins[:-1],dihedral_histogram



def open_multiple_dihedral_file():
    """ This function will open a trajecteory in parallel to create a global histogram
    Returns:
    ----
    For rank 0 only (None for other ranks)
    dihedral_df(dataframe): a dataframe with the bins as index and the angle frequencies as the column
    trans_ratio(float) : trans dihedral ratio
    gauche_ratio(float): gauche dihedral ratio
    gauche_minus_ratio(float): gauche minus ratio
    output_dihedral.out(file): The first column is the bins and the second the frequencies
    """
    rank = MPI.COMM_WORLD.Get_rank()
    nprocs = MPI.COMM_WORLD.Get_size()
    # create a list of all the dihedrals filename
    complete_dihedral = glob("dihedral*")
    # sort the list
    complete_dihedral.sort(key=lambda f: int(filter(str.isdigit, f)))
    # consider only the desired file
    last5ns_dihedral = complete_dihedral[first:last]
    fragment_dihedral = np.array_split(last5ns_dihedral, nprocs)

    for dihedral_files in np.nditer(fragment_dihedral[rank][:], flags=['external_loop']):
        dihedral_histogram_rank = np.zeros(shape=(1),dtype=np.int)
        intrachain_histogram_rank = np.zeros(shape=(1),dtype=np.int)
        for dihedral_file in dihedral_files:
            histogram_bins,dihedral_histogram = get_histogram_dihedral(dihedral_file)
            #reshape the array to have the same size
            if len(dihedral_histogram_rank) < len(dihedral_histogram):
                dihedral_histogram_rank.resize(dihedral_histogram.shape)
            dihedral_histogram_rank = dihedral_histogram_rank + dihedral_histogram
        #the first processor will gather all the arays

    MPI.COMM_WORLD.barrier()
    intrachain_histogram_total = np.zeros(intrachain_histogram_rank.shape,dtype=np.int)
    dihedral_histogram_total = np.zeros(dihedral_histogram_rank.shape,dtype=np.int)

    MPI.COMM_WORLD.Reduce(intrachain_histogram_rank,intrachain_histogram_total,op = MPI.SUM,root = 0)
    MPI.COMM_WORLD.Reduce(dihedral_histogram_rank,dihedral_histogram_total,op = MPI.SUM,root = 0)
    #The first rank calculate the ratios and the output
    if rank == 0:
        dihedral_df = pd.DataFrame({"bins":histogram_bins,"dihedral_angle":dihedral_histogram_total})
        dihedral_df = dihedral_df.set_index(["bins"])
        #file
        dihedral_df.to_csv("output_dihedral.out",sep = " ")
        #ratio of the dihedral configuration
        trans_ratio = float((dihedral_df.loc[:-120].sum()+dihedral_df.loc[120:].sum())/dihedral_df.sum())
        gauche_minus_ratio = float(dihedral_df.loc[-120:0].sum()/dihedral_df.sum())
        gauche_ratio = float(dihedral_df.loc[0:120].sum()/dihedral_df.sum())
        print("Ratio of trans,gauche+ and gauche-: ",trans_ratio ,gauche_ratio,gauche_minus_ratio)
        return dihedral_df, trans_ratio,gauche_ratio,gauche_minus_ratio
    else:
        return None,None,None,None

def visualize(dihedral_df):
    """Function to visualize the histogram

    Args:
    dihedral_df(dataframe): dataframe with the bins and frequencies

    Returns:
    a matplotlib histogram of the dihedral angles
    """
    plt.bar(dihedral_df.index,dihedral_df.dihedral_angle)
    plt.xlabel(r"$\phi(^o)$")
    plt.ylabel(r"Nombre d'angles de torsion")
    plt.show()


dihedral_df, trans_ratio,gauche_ratio,gauche_minus_ratio = open_multiple_dihedral_file()


if rank == 0:
    visualize(dihedral_df)
