#!/usr/bin/python
"""
Functions to compute the cluster size distribution from a LAMMPS output. The average
number of hairpin defects per chain can also be calculated.

Usage:
#must be in pythonpath or working directory
from cluster_analysis import cluster
cluster = cluster(fname="cluster.15000000.out.gz")
print(cluster.occurrence())
print(cluster.histogram())
print(cluster.hairpin())

Requirement:
python2
numpy
pandas
dump_dataframe.py

Limitations:
The calculated average number of hairpins defects per chain is invalid if particles
from different layers share a cluster_id
"""
import numpy as np
import pandas as pd
from dump_dataframe import read_dump


class cluster():
    def __init__(self, fname, monomerid=True, monomer_in_chains=8, number_of_chains=100):
        """
        Initialisation of parameters

        Args:
        ----
        fname("string"): filename of the cluster output from lammps
        monomerid(boolean): True to change the mol_id to reprensent a chain instead of a momoner
        monomer_in_chains(int): Number of monomer in the polymer chains
        number_of_chains(int): Number of chains in the polymer

        Attributes:
        ----
        Same as Args +
        cluster_array(array): the occurence of each cluster size when occurrence is called
        average_hairpins(float) average number of hairpins defect per chain when the function hairpin_defects is called
        cluster_histogram(array): The distrubition of cluster size when histogram is called
        bins(array): the bins of the cluster size distribution when histogram is called
        cluster_df(DataFrame): Pandas dataframe with the atom id as index, mol id and cluster id as columns
        """
        # initalization of the parameters
        self.monomerid = monomerid
        self.monomer_in_chains = monomer_in_chains
        self.number_of_chains = number_of_chains
        self.average_hairpins = None
        self.cluster_array = None
        self.cluster_histogram = None
        self.bins = None
        # reading the file
        self.cluster_df = read_dump(fname)["atom_df"]
        self.cluster_df.columns = ["mol", "cluster_id"]
        # to hav the mol_id corresponding to a polymer chain if monomerid is
        # true
        if self.monomerid == True:
            for i in range(self.number_of_chains):
                self.cluster_df["mol"][(self.cluster_df.mol > i * self.monomer_in_chains) & (
                    self.cluster_df.mol <= self.monomer_in_chains + i * self.monomer_in_chains)] = i + 1

    def occurrence(self):
        """
        Function to calculate the occurrence of each cluster

        Return:
        ----
        cluster_array(array): Occurence of each cluster_id
        """
        # count the occurrence of each cluster
        self.cluster_array = np.array(
            self.cluster_df["cluster_id"].value_counts())
        return self.cluster_array

    def hairpin(self):
        """
        Function to calculate the average number of hairpin defects in a chain, by comparing
        if the next index has the same mol and cluster_id.

        Limitations: Invalid if different layers shares cluster_id

        Return:
        average_hairpins(float): average number of hairpin defects
        """
        number_of_hairpins = 0
        for i in range(len(self.cluster_df.index) - 1):
            # compare the mol and cluster_id of the index and the next
            if self.cluster_df.iloc[i].equals(self.cluster_df.iloc[i + 1]):
                # count the number of occurence
                number_of_hairpins += 1
        # Calculate the average per chains
        self.average_hairpins = float(
            number_of_hairpins) / float(self.number_of_chains)
        return self.average_hairpins

    def histogram(self):
        """
        Function to create a distribution of the clusters size

        Return:
        ----
        cluster_histogram(array): numpy array with the occurence for each size
        bins(array): the corresponding cluster size
        """
        # histogram for the cluster size. max size +2 to include the largest
        # cluster
        self.cluster_histogram, self.bins = np.histogram(
            self.cluster_array, bins=np.arange(1, self.cluster_array.max() + 3))
        # return the histogram and the bins without the last value to have
        # arrays with the same shape
        self.bins = self.bins[:-1]
        return self.cluster_histogram
