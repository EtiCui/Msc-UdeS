#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:11:22 2016

@author: etienne
Functions to calculate the nematic order parameter from ordering tensor (see arXiv:1409.3542v2) and
the smectic A order parameter (see The Journal of chemical physics 138(20), 204901 (2013).)
from a LAMMPS output with Gay-Berne particles (with pseudo-atoms at the ends) for a liqud crystalline polymer.

Usage:
#must be in pythonpath or working directory
from nematic_sma_OP import PO
po = PO(fname=dump.gz, wrap=True, visualize=True, ini_layer_spacing=35., gb_type=3, gb_ends_type=2, atoms_per_monomer=23,
        number_of_monomer=800, number_of_chains=100))
po.nematic()
po.sma()

Requires:
pandas
mpi4py
numpy
scipy
dump_dataframe
matplotlib

TODO:
Orientation from quaternion
multiple ellipsoid type
reading atomatically atoms per monomer and number of molecules
"""

import numpy as np
import gzip
import pandas as pd
import itertools
from scipy.optimize import curve_fit
from dump_dataframe import read_dump
import matplotlib.pyplot as plt


class PO():
    def __init__(self, fname, wrap, visualize, ini_layer_spacing=35., gb_type=3, gb_ends_type=2,
                atoms_per_monomer=23, number_of_monomer=800, number_of_chains=100):
        """Initialization of the parameters and reading the dump

        Args:
        -----
        fname(str): filename (must be a .gz)
        wrap (bool): True if the coordinates are to be wrapped in the box
        visualize (bool): True if a graph is desired for the layers in the SmA script
                   WARNING: ABOUT 3 TIMES SLOWER for 300dpi figure
        ini_layer_spacing (float): initial layer spacing to optimize with least-square
        gb_type (int): Atom type of the GB particles
        gb_ends_type (int): int of the atom type of the pseudo-atoms at the ends of GB particles
        atoms_per_monomer(int): Number of atoms per monomer
        number_of_monomer(int): Number of monomer in the system
        number_of_chains(Int): Number of chains in the system

        Attributes:
        ----
        The args
        step(int): Timestep of the dump
        nb_atoms(int): Number of atoms in the system
        x_dimension(float): lenght of the box in x
        y_dimension(float): lenght of the box in y
        z_dimension(float): lenght of the box in the z dimension
        atom_df (dataframe): dataframe for the dump see dump_dataframe
        """
        # atom type for the ellipsoid
        self.gb_type = gb_type
        # atom type for the ends of the ellipsoid
        self.gb_ends_type = gb_ends_type
        # atoms per monomer in the chain
        self.atoms_per_monomer = atoms_per_monomer
        # number of monomer
        self.number_of_monomer = number_of_monomer
        # number of chains
        self.number_of_chains = number_of_chains
        # approximate SmA spacing between the layers
        self.ini_layer_spacing = ini_layer_spacing
        # Boolean for visualization
        self.visualize = visualize
        # Reading the dump
        dump = read_dump(fname, wrap)
        # Defininition the parameters obtained from the dump
        self.step = dump["step"]
        self.nb_atoms = dump["nb_atoms"]
        dimensions = dump["dimensions"]
        self.x_dimension = dimensions[0]
        self.y_dimension = dimensions[1]
        self.z_dimension = dimensions[2]
        self.atom_df = dump["atom_df"]

    def nematic(self):
        """This function will calculate the nematic order parameter with -2*middle_eigenvalues

        Returns:
        ----
        step(int): timestep of the dump
        nematic_OP(float): the nematic order parameter
        director(array): [x,y,z] components of the director
        """
        # choose only the usefull columns
        nematic_col_list = ["type", "x", "y", "z"]
        nematic_df = self.atom_df[nematic_col_list]

        # Choose only the atoms with the atom type corresponding to the
        # pseudo-atoms of the ellipsoid
        ellipsoid_ends = nematic_df[nematic_df["type"]
                                    == self.gb_ends_type]
        # drop the now useless type column
        ellipsoid_ends = ellipsoid_ends.drop(["type"], axis=1)
        # iterating over all the monomer for the outer product
        qtot = 0.
        for i in range(0, self.number_of_monomer):
            # create all the ellipsoids_vector
            ellipsoid_ends_per_chain = ellipsoid_ends.loc[self.atoms_per_monomer *
                                                          i:self.atoms_per_monomer + self.atoms_per_monomer * i]
            # vector components while considering PBC
            x_component = ellipsoid_ends_per_chain["x"].iloc[1] - \
                ellipsoid_ends_per_chain["x"].iloc[0]
            x_component = np.where(
                x_component > 0.5 * self.x_dimension, x_component - self.x_dimension, x_component)
            x_component = np.where(
                x_component < -0.5 * self.x_dimension, x_component + self.x_dimension, x_component)

            y_component = ellipsoid_ends_per_chain["y"].iloc[1] - \
                ellipsoid_ends_per_chain["y"].iloc[0]
            y_component = np.where(
                y_component > 0.5 * self.y_dimension, y_component - self.y_dimension, y_component)
            y_component = np.where(
                y_component < -0.5 * self.y_dimension, y_component + self.y_dimension, y_component)

            z_component = ellipsoid_ends_per_chain["z"].iloc[1] - \
                ellipsoid_ends_per_chain["z"].iloc[0]
            z_component = np.where(
                z_component > 0.5 * self.z_dimension, z_component - self.z_dimension, z_component)
            z_component = np.where(
                z_component < -0.5 * self.z_dimension, z_component + self.z_dimension, z_component)

            # ellipsoid vector
            ellipsoid_vector = np.array(
                [x_component, y_component, z_component])

            # normalize the vector
            ellipsoid_vector = ellipsoid_vector / \
                np.linalg.norm(ellipsoid_vector)

            # compute the outer product
            outer_product = 3. * \
                np.outer(ellipsoid_vector, ellipsoid_vector) - np.identity(3)
            qtot = qtot + outer_product

        # Order matrix
        Q = qtot / 2. / self.number_of_monomer

        # calculate the eigenvector,eigenvalues and sort them
        eig_vals, eig_vecs = np.linalg.eig(Q)
        idx = eig_vals.argsort()[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]
        # The middle eigenvalue multiplied by -2 is the order parameter
        # the eigeinvector corresponding to the largest eigenvalue is the
        # director
        self.nematic_OP = -eig_vals[1] * 2.

        self.director = eig_vecs[:, 0]
        return self.step, self.nematic_OP, self.director

    def fit_sma(self, gz12, distance):
        """ Non linear least-square fitting for the SmA order parameter
        Parameters:
        ----
        gz12 (array): array of the gz12 values
        distance(array): array for the corresponding distance

        Returns:
        ----
        step(int): The timestep of the dump
        sma_OP(float): the smectic A order parameter
        layer_spacing(float): the optimized layer spacing
        matplotlib graph in png of gz12 in function of z12
        """

        # function to fit, 6 terms in the cosinus sum are considered
        def func(x, p1, p2, p3, p4, p5, p6, p7):
            return 1 + 2 * p1 * p1 * np.cos(2 * 3.141592 * x / p2) + 2 * p3 * p3 * np.cos(2 * 2 * 3.141592 * x / p2) \
            + 2 * p4 * p4 * np.cos(2 * 3 * 3.141592 * x / p2) + 2 * p5 * p5 * np.cos(2 * 4 * 3.141592 * x / p2) \
            + 2 * p6 * p6 * np.cos(2 * 5 * 2 * 3.141592 * x / p2) + 2 * p7 * p7 * np.cos(2 * 6 * 3 * 3.141592 * x / p2)

        popt, pcov = curve_fit(func, distance, gz12, p0=(
            1.0, self.ini_layer_spacing, 1.0, 1.0, 1.0, 1.0, 1.0))

        #function to plot the function with fitted parameters
        if self.visualize == True:
            plt.plot(distance, func(distance, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5],
                                    popt[6]), label=r"$1+2\sum_{n=1}^6 (\tau_n)^2 cos( \frac{2\pi n z_{12}}{d})$")
            plt.legend(loc="upper right")
            plt.savefig("gz12_" + str(self.step) + ".png", dpi=300)

        self.sma_OP, self.layer_spacing, self.t2, self.t3, self.t4, self.t5, self.t6 = popt
        return self.step, self.sma_OP, self.layer_spacing

    def sma(self):
        """This function will calculate the SmA order parameter by least-square fitting
        the two-particle density correlation function. Will return -1 if it did not converge
        WARNING the layer direction will be the direction of the box dimensions which is
        not always true, visualizing the results  is thus important

        Returns:
        ----
        If the least-square converged:
        step(int): The timestep of the dump
        sma_OP(float): the smectic A order parameter
        layer_spacing(float): the optimized layer spacing
        matplotlib graph in png of gz12 in function of z12

        If least-square did not converged:
        step(int):the timestep of the dump
        sma_OP(int):-1
        layer_spacing(int):-1
        matplotlib graph in png of gz12 in function of z12
        """

        sma_col_list = ["type", "x", "y", "z"]
        sma_df = self.atom_df[sma_col_list]
        # Catch the exception if the fitting did not converge
        try:
            # dataframe for the center of mass of the ellipsoid
            ellipsoid_cm = sma_df[sma_df["type"] == self.gb_type]
            # Droping the now useless type column
            ellipsoid_cm = ellipsoid_cm.drop(["type"], axis=1)

            # The bounds will be the largest box dimension
            bounds = max(self.x_dimension, self.y_dimension, self.z_dimension)
            if bounds == self.z_dimension:
                self.direction = "z"
            if bounds == self.x_dimension:
                self.direction = "x"
            if bounds == self.y_dimension:
                self.direction = "y"
            # dataframe for the component of the cm in the layer direction
            ellipsoid_cm_layer = pd.DataFrame()
            ellipsoid_cm_layer["z"] = ellipsoid_cm[self.direction]

            # empty array to create all the difference between cm
            z12 = np.empty([0])
            combi = np.array(
                list(itertools.combinations(ellipsoid_cm_layer.z, 2)))
            z12 = np.absolute(combi[:, 0] - combi[:, 1])

            # For the PBC
            z12 = np.absolute(np.where(
                z12 > 0.5 * bounds, z12 - bounds, z12))

            # create the histogram for the frequencies at each distance in bins
            # of 1
            neighbor_histogram = np.histogram(
                z12, bins=np.arange(0, bounds / 2.))
            # normalize to obtain gz12
            gz12 = neighbor_histogram[0] / np.average(neighbor_histogram[0])
            # ignore the last value for the array to have the same size
            distance = neighbor_histogram[1][:-1]
            if self.visualize == True:
                plt.clf()
                plt.plot(distance, gz12)
                plt.xlabel(r"$z_{12}$")
                plt.ylabel(r"$g(z_{12})$")
            return self.fit_sma(gz12, distance)
        except RuntimeError:
            # return -1, -1 if it did not converge for the PO and the layer
            # spacing
            if self.visualize ==True:                
                plt.savefig("gz12_" + str(self.step) + ".png", dpi=300)
            return self.step, -1, -1
