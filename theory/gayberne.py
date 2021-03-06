#!/usr/bin/python2
"""This script will call LAMMPS to calculate and visualize (with matplotlib) the Gay-Berne potential
for the side-by-side (ll), end-to-end(z),tee(T) and cross(X) configurations for a uniaxial ellipsoid.

Note:
If the default intermolecular distances for a configuration are too far from it's minimum,
some tuning might be required in the move_molecule function

Usage:
#the script must be in python path or working directory
from gayberne import gb
gb = gb(width=1, length=3, sigma=1, X_depth=3, z_depth=1, nu=1, mu=2,x_range=150)
lammps_df = gb.lammps_gb()
gb.visualize(lammps_df=lammps_df)

Requirement:
numpy
matplotlib
pandas
Python interface to LAMMPS (http://lammps.sandia.gov/doc/Section_python.html)
LAMMPS compiled with asphere and molecule package
system.data file (the defined parameters will be overwritten)

TODO:
The system.data file should not be required
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
from lammps import lammps
import matplotlib.lines as mlines


class gb():
    def __init__(self, width, length, sigma, X_depth, z_depth, nu, mu, cutoff=30,x_range=150):
        """ Definition of the parameters and initialization"""
        self.df = pd.DataFrame()
        # configuration with the color for the graph: red, green, blue, yellow
        self.colors = {"ll": "r", "X": "g", "z": "b", "T": "y"}
        # parameters definition
        self.width = width
        self.length = length
        self.sigma = sigma
        self.X_depth = X_depth
        self.z_depth = z_depth
        self.nu = nu
        self.mu = mu
        self.cutoff = cutoff
        # range for the graph
        self.x_range = x_range

    def lammps_gb(self):
        """ This function will call lammps to calculate the Gay-Berne potential for a uniaxial ellipsoid
        Returns:
        Pandas dataframe with the intermolecular distance and energy for the four configurations
        """
        # no output to screen or file
        lmp = lammps(cmdargs=["-sc", "none", "-log", "none"])
        lmp.command("units lj")
        lmp.command("boundary p p p")
        lmp.command("atom_style hybrid molecular ellipsoid")
        # define the gay berne pair_style
        lmp.command("pair_style gayberne 1.0 %f %f %f" %
                    (self.nu, self.mu, self.cutoff))
        lmp.command("read_data system.data")
        # define the ellipsoid's shape
        lmp.command("set type 1 shape %f %f %f" %
                    (self.width, self.width, self.length))
        # define initial orientation for the atoms
        lmp.command("set atom 1 quat 1 0 0 0")
        lmp.command("set atom 2 quat 1 0 0 0")
        # Define the gay-berne pair_coeff
        lmp.command("pair_coeff 1 1 1.0 %f %f %f %f %f %f %f" % (self.sigma, self.X_depth, self.X_depth,
                                                                 self.z_depth, self.X_depth,
                                                                 self.X_depth, self.z_depth))

        # function to move the first ellipsoid. direction: 0 is x, 1 is y and 2
        # is z.
        def move_molecule(configuration, direction):
            """ Function to translate the molecule in the desired direction"""
            i = 0
            pot_tot = []
            position = []
            i = 0
            while i < self.x_range:
                xc = lmp.gather_atoms("x", 1, 3)
                lmp.command("run 0")
                pot = lmp.extract_compute("thermo_pe", 0, 0)
                pot_tot.append(pot)
                position.append(xc[direction])
                xc[direction] = xc[direction] + 0.01
                lmp.scatter_atoms("x", 1, 3, xc)
                i = i + 1
            position = pd.DataFrame(
                {"MD_"+ str(configuration) + "_distance": position})
            pot_tot = pd.DataFrame(
                {"MD_" + str(configuration) + "_energy": pot_tot})
            self.df = pd.concat([self.df, position, pot_tot], axis=1)

        for configuration in self.colors:
            # reinitialze the position of the first atom
            lmp.command("set atom 1 x 0 y 0 z 0")
            if configuration == "ll" or "z":
                # No rotatio for ll and end configuration
                lmp.command("set atom 1 quat 1 0 0 0")
                if configuration == "ll":
                    # move in x direction for side-by-side
                    direction = 0
                    xc = lmp.gather_atoms("x", 1, 3)
                    # Can be changed if the values are too far from the minimum
                    xc[direction] = xc[direction] + self.width/1.5
                    xc = lmp.scatter_atoms("x", 1, 3, xc)
                    move_molecule(configuration, direction)
                if configuration == "z":
                    # move in the z direction for end-to-end
                    direction = 2
                    xc = lmp.gather_atoms("x", 1, 3)
                    # Can be changed if the values are too far from the minimum
                    xc[direction] = xc[direction] + self.length/1.1
                    xc = lmp.scatter_atoms("x", 1, 3, xc)
                    move_molecule(configuration, direction)

            if configuration == "T" or "X":
                # The first ellipsoid is rotated by 90 degree in the x axis
                lmp.command("set atom 1 quat 1 0  0 90")
                if configuration == "X":
                    # the first ellipsoid is translated in the x direction
                    direction = 0
                    xc = lmp.gather_atoms("x", 1, 3)
                    # Can be changed if the values are too far from the minimum
                    xc[direction] = xc[direction] + self.width/1.5
                    xc = lmp.scatter_atoms("x", 1, 3, xc)
                    move_molecule(configuration, direction)
                if configuration == "T":
                    # Translation in the y axis
                    direction = 1
                    xc = lmp.gather_atoms("x", 1, 3)
                    xc[direction] = xc[direction] + self.length - \
                        self.width  # Can be changed if the values are too far from the minimum
                    xc = lmp.scatter_atoms("x", 1, 3, xc)
                    move_molecule(configuration, direction)
        # close lammps
        print("LAMMPS done")
        return self.df
        lmp.close()

    def visualize(self, lammps_df=None):
        """ Function to visualize with matplotlib the gay-berne potential calculated from lammps"""
        if type(lammps_df).__name__ == "NoneType":
            lammps_df = self.lammps_gb()

        for configuration in self.colors:
            # Remove values with a energy superior to 5 for the graph
	    df_graph = lammps_df.where(lammps_df["MD_"+str(configuration) + "_energy"] < 5)
            plt.plot(df_graph["MD_"+str(configuration) + "_distance"].dropna(),
            df_graph["MD_"+str(configuration) + "_energy"].dropna(), label=str(configuration),
            color=self.colors[configuration], ms=5, markevery=0.02)
            plt.ylabel(r"Potentiel ($\epsilon_0$)")
            plt.xlabel(r"Distance ($\sigma_0$)")

        plt.legend()
        plt.tight_layout()

        plt.show()
#gb = gb(width=1, length=3, sigma=1, X_depth=3, z_depth=1, nu=1, mu=2,x_range=150)
#lammps_df = gb.lammps_gb()
#gb.visualize(lammps_df=lammps_df)
