#!/usr/bin/python
# -*- coding:utf-8-*-
"""
Created on Wed Apr 12 09:38:44 2017

@author: etienne cuierrier

This script will do a least-square fitting of Gay-Berne parameters from ab initio results
The rotationally averaged results from the 4 orthogonal configurations are required
(average_X.txt,average_ll.txt,average_z.txt,average_T.txt) with the first columns as the
intermolecular distance in angstrom and the second the energy in kj/mol.

****IMPORTANT****: In this script, the T configuration well depth had to be fitted independently,
 since it the least-square algorithm could not converge. For the graph, it uses the same
  l and d as the other configurations

Usage:
from gayberne_fit import gayberne_fit
fit_gb = gayberne_fit()
# Fitting the parameters
fit_gb.fit()
#To caculate the GB potential with LAMMPS
fit_gb.lammps_df()
# To visualize with the desired methods
fit_gb.visualize(methods=["MP2", "average", "GB", "MD"])

Requires:
numpy
matplotlib
pandas
scipy
lammps as a python library
gayberne.py (see theory folder)

References:
Berardi, Roberto, C. Fava, and Claudio Zannoni. "A generalized Gay-Berne intermolecular
potential for biaxial particles." Chemical physics letters 236.4-5 (1995): 462-468.
The fitting procedure is modified from  https://mail.scipy.org/pipermail/scipy-user/2013-April/034406.html

TODO:
Reduce the number of err_function to 1
Simpler/shorter move_molecule functions
Calculate the average from the rotated molecules
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import glob
import pandas as pd
import scipy.optimize
from collections import OrderedDict
from lammps import lammps
from gayberne import gb


class gayberne_fit():

    def __init__(self, temperature=800, initial_global=[20, 4.69, 19.91, 12, 1, 5.22],
                 initial_T=[10, 4, 15, 5.44], cutoff=30, initial_mu=-0.5):
        """
        Reads the files and initialization of the parameters
        """
        #%% loads all the files with .txt extension to create a data frame
        # Create a list with all files with glob
        self.fnames = glob.glob("*.txt")
        #  Pandas dataframe  with the filename as column name. In the
        # .txt files, x is the distance in angstrom and y the energy in jk/mol
        self.df = pd.concat([pd.read_csv(fname, names=[fname.split(".")[0] + "_distance", fname.split(".")[0] + "_energy"],
                                         sep=' |\t', engine='python') for fname in self.fnames], axis=1)  # The separator in the files are space
        # temperature for the boltzmann weight for the fit
        self.temp = temperature
        # initial global parameter to optimize for the curve fit
        self.initial_global = initial_global
        self.initial_T = initial_T
        # configuration with the color for the graph: red, green, blue, yellow
        self.colors = {"ll": "r", "X": "g", "z": "b", "T": "y"}
        # marker with the size if there are different type of graph (ex:
        # average results, all the results, lammps...
        self.marker = OrderedDict([("o", 1), ("*", 6), ("-", 1), ("x", 3)])
        # cutoff for lammps interaction
        self.cutoff = cutoff
        # Estimate for mu wihich will be solved numerically
        self.initial_mu = initial_mu

    #%% Functions for the orthogonal configuration of the Gay-Berne potential
    def configuration(self, x, p, config):
        """
        The Gay-Berne potential for the orthogonal configuration

        Parameters
        ----
        x: array of distances
        p: list of parameters for the gay-berne Potentiel
        config: string of the desired orthogonal configuration

        Returns
        ---
        The calculated potentiel
        """
        if config == "ll":
            es, d, l, c = p
            return 4 * es * ((c / (x - d + c))**12 - (c / (x - d + c))**6)

        if config == "X":
            eo, d, l, c = p
            return 4 * eo * ((c / (x - d + c))**12 - (c / (x - d + c))**6)

        if config == "z":
            ee, d, l, c = p
            return 4 * ee * ((c / (x - l + c))**12 - (c / (x - l + c))**6)

        if config == "T":
            et, d, l, c = p
            return 4 * et * ((c / (x - np.sqrt((d**2 + l**2) / 2) + c))**12 - (c / (x - np.sqrt((d**2 + l**2) / 2) + c))**6)

    #%% individual error function to optimize by least-square fitting with boltzman weight
    def err_ll(self, p, x, y):
        return (self.configuration(x, p, "ll") - y) * np.exp(-y / 8.31 * 1000 / self.temp) / np.sum(-np.array(self.df["average_ll_energy"].dropna()) / 8.31 * 1000 / self.temp)

    def err_X(self, p, x, y):
        return (self.configuration(x, p, "X") - y) * np.exp(-y / 8.31 * 1000 / self.temp) / np.sum(-np.array(self.df["average_X_energy"].dropna()) / 8.31 * 1000 / self.temp)

    def err_z(self, p, x, y):
        return (self.configuration(x, p, "z") - y) * np.exp(-y / 8.31 * 1000 / self.temp) / np.sum(-np.array(self.df["average_z_energy"].dropna()) / 8.31 * 1000 / self.temp)

    def err_T(self, p, x, y):
        return (self.configuration(x, p, "T") - y) * np.exp(-y / 8.31 * 1000 / self.temp) / np.sum(-np.array(self.df["average_T_energy"].dropna()) / 8.31 * 1000 / self.temp)

    #%% global error function
    def err_global(self, p, x1, x2, x3, y1, y2, y3):
        """
        Global error function to optimize by least-square fitting. T configuration is commented due to convergence problem.

        Parameters
        ----
        p: list of the gay_berne parameters to optimize [epsilon_ll,d,l,epsilon_X,epsilon_z,sigma_c]
        x1,x2,x3: array of distances for the configurations
        y1,y2,y3: arrays of energies for the configurations

        Returns
        ----
        The concatenated error for each configuration
        """
        # Shared and independent parameter for each configuration :
        # epsilon_ll, d, l, epsilon_X,epsilon_z, sigma_c
        ll_parameter = p[0], p[1], p[2], p[5]
        X_parameter = p[3], p[1], p[2], p[5]
        z_parameter = p[4], p[1], p[2], p[5]
    #   T_parameter = p[6], p[1], p[2], p[5]

        err_ll = self.err_ll(ll_parameter, x1, y1)
        err_X = self.err_X(X_parameter, x2, y2)
        err_z = self.err_z(z_parameter, x3, y3)
    #    err_T = err_T(p4,x4,y4)
        return np.concatenate((err_ll, err_X, err_z))
    #%% Function to do the least-square fitting

    def fit(self):
        """
        Least-square fitting of the Gay-Berne potential

        Returns:
        ----
        Print of the optimized Gay-Berne parameters
        """
        best_global, ier = scipy.optimize.leastsq(self.err_global, self.initial_global,
                                                  args=(np.array(self.df["average_ll_distance"].dropna()),
                                                        np.array(
                                                      self.df["average_X_distance"].dropna()),
                                                      np.array(
                                                      self.df["average_z_distance"].dropna()),
                                                      np.array(
                                                      self.df["average_ll_energy"].dropna()),
                                                      np.array(
                                                      self.df["average_X_energy"].dropna()),
                                                      np.array(self.df["average_z_energy"].dropna())))

        best_T, ier = p_best, ier = scipy.optimize.leastsq(self.err_T, self.initial_T,
                                                           args=(np.array(self.df["average_T_distance"].dropna()),
                                                                 np.array(self.df["average_T_energy"].dropna())))
        # Optimized Gay-Berne parameters
        self.ll_depth = best_global[0]
        self.X_depth = best_global[3]
        self.z_depth = best_global[4]
        self.T_depth = best_T[0]
        self.width = best_global[1]
        self.length = best_global[2]
        self.sigma = best_global[5]

        # Nu parameter in gay-berne potential
        logbase = (self.width**2 + self.length**2) / \
            (2 * self.length * self.width)
        self.nu = math.log(self.ll_depth / self.X_depth, logbase)

        # Epsilon_z in gay-berne
        self.epsilon_z = self.z_depth / \
            (self.width / logbase)**self.nu

        # Function to optimize the mu parameter in gay-berne potential
        def mu_equation(mu):
            return -self.T_depth + (2. / ((1 / self.X_depth)**(1 / mu) + (1 / self.epsilon_z)**(1 / mu)))**mu
        self.mu = scipy.optimize.fsolve(mu_equation, self.initial_mu)[0]

        print("Global fit results")
        print("ll-by-ll well depth: ", self.ll_depth, "X well depth: ", self.X_depth, "z to z well depth: ", self.z_depth,
              "T well depth:", self.T_depth, "epsilon z: ", self.epsilon_z,
              "width d: ", self.width, "length l: ", self.length, "sigma: ", self.sigma, "nu: ", self.nu, "mu: ", self.mu)

    # Assign each parameter to the corresponding orthogonal configuration
    def configuration_parameter(self, config):
        """
        Assignation of the parameters to each orthogonal configuration

        Parameters:
        ----
        config: string of the configuration

        Returns:
        list of the parameters
        """
        if config == "ll":
            return [self.ll_depth, self.width, self.length, self.sigma]
        if config == "X":
            return [self.X_depth, self.width, self.length, self.sigma]
        if config == "z":
            return [self.z_depth, self.width, self.length, self.sigma]
        if config == "T":
            return [self.T_depth, self.width, self.length, self.sigma]

    # Lammps results from the optimised parameters
    def lammps_df(self):
        """
        Function to calculate the Gay-Berne potentiel with lammps
        """
        gb_ini = gb(self.width, self.length, self.sigma, self.X_depth,
                    self.epsilon_z, self.nu, self.mu, self.cutoff, x_range=1000)
        lammps_df = gb_ini.lammps_gb()
        self.df = pd.concat([self.df, lammps_df], axis=1)

    def visualize(self, methods=["MP2", "average", "GB", "MD"]):
        """ Function to visualize with matplotlib the ab initio , the fitted curve lammps"""
        # add to dataframe the GB potential with the defined parameters, it
        # will use the lammps intermolecular distance as x since the increment
        # is small
        for configuration in self.colors:
            self.df["GB_" + str(configuration) + "_distance"] = self.df[
                "MD_" + str(configuration) + "_distance"]
            self.df["GB_" + str(configuration) + "_energy"] = self.configuration(self.df["GB_" + str(
                configuration) + "_distance"], self.configuration_parameter(str(configuration)), str(configuration))
        i = 0

        for method in methods:

            for configuration in self.colors:
                # Remove values with a energy superior to 20 kJ/mol for the
                # graph
                df_graph = self.df.where(
                    self.df[str(method) + "_" + str(configuration) + "_energy"] < 20)
                plt.plot(df_graph[str(method) + "_" + str(configuration) + "_distance"].dropna() / 10., df_graph[str(method) + "_" + str(configuration) + "_energy"].dropna(),
                         self.marker.keys()[i], label=method + " " + str(configuration), color=self.colors[configuration], ms=self.marker.values()[i])
                plt.ylabel("Potentiel (kJ/mol)")
                plt.xlabel("Distance (nm)")
            i += 1
        plt.legend(loc="lower right", ncol=2)
        plt.show()
