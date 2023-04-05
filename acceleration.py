"""
NAME
    acceleration
DESCRIPTION
    a collection of functions that calculates the accelerations of each particle
FUNCTIONS:
    numpy_pairwise
"""

import numpy as np


def numpy_pairwise(force, m, q):
    """
    To implement RK4 and other algorithms easily, we want an acceleration
    function that only has position as a variable

    Creates a function to calculate the accelerations of particles from their positions.
    Compares each particle to each other so O(n2), however it is very optimized due to only
    using numpy functions

    Arguments
    ---------
        force: function
            the function that calculates the force between two particles
        m: the mass of the particles

    Returns
    -------
        _numpy_pairwise: function
            the function that calculates the accelerations of the particles from only
            their positions
    """

    def _numpy_pairwise(s):
        """
        Calculates the acceleration of each particle from their positions

        Arguments
        ---------
            s: np.ndarray
                the positions of the particles

        Returns
        -------
            a: np.ndarray
                the accelerations of each particle
        """
        n = s.shape[0]  # Number of particles
        a = np.zeros_like(s)
        for i in range(1, n):
            r = np.roll(s, i, axis=0) - s  # Vectors between pairs of particles
            q2 = np.roll(q, i)  # Charges of attracting particles
            d = np.linalg.norm(r, axis=1)  # Distances between pairs of particles
            # Direction between particles
            r_norm = np.divide(r, d[:, np.newaxis], where=(d[:, np.newaxis] != 0))
            a += r_norm * (force(d, q, q2) / m)[:, np.newaxis]
        return a

    return _numpy_pairwise
