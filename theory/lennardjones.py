""" Script to visualize the Lennard-jones 12-6 potential

Usage:
#the script must be in pythonpath or working directory
from lennardjones import visualize_lj
visualize_lj(e=[3,4],s=[4,6],x=[3,10,0.05],ylim=[-6.5,4.5])

Limitations:
Even if there is only one epsilon or sigma, they must be in a list, ex: e=[3] and not e=3
"""

import numpy as np
import matplotlib.pyplot as plt


def visualize_lj(e, s, x, ylim):
    """Function to visualize the Lennardjones-potential

    Parameters:
    ----------
    e: list of the epsilon values ex: [e1,e2]
    s: list of sigma values ex: [s1,s2]
    x: list for the range of x values and step ex: [x_0,x_final,step]
    ylim: list for the y limits of the plotted axis ex:[lower_y,upper_y]

    Returns:
    --------
    Matplolib graph of the potential
    """
    x = np.arange(x[0], x[1], x[2])
    # lennardjones-potential

    def lennardjones(x, e, c):
        """Function to compute the Lennard-Jones 12-6 potential"""
        return 4 * e * ((c / x)**12 - (c / x)**6)
    # Vectorize the function with numpy so it will return an array
    vlj = np.vectorize(lennardjones)
    # For loops for each epsilon and sigma values
    for i in e:
        for j in s:
            # calculate lennard jones
            e_lj = vlj(x, i, j)
            # plot with the sigma and epsilon as label
            plt.plot(x, e_lj, label=(
                r"$\epsilon =$ %.2f $\sigma =$ %.2f" % (i, j)), linewidth=1)

    plt.ylabel(r"$v^{LJ}$ (potential)")
    plt.xlabel(r"r (distance)")
    plt.ylim(ylim[0], ylim[1])
    plt.legend()
    plt.show()
