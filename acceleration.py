"""
NAME
    acceleration
DESCRIPTION
    a collection of functions that calculates the accelerations of each
    particle
FUNCTIONS:
    numpy_pairwise
"""

import numpy as np
from typing import Callable, Any

from numpy import ndarray, dtype
from numpy._typing._generic_alias import ScalarType
from numpy.typing import NDArray


def numpy_pairwise(force: Callable[[NDArray, NDArray, NDArray], NDArray],
                   m: NDArray, q: NDArray) -> Callable[[NDArray], NDArray]:
    """
    To implement RK4 and other algorithms easily, we want an acceleration
    function that only has position as a variable

    Creates a function to calculate the accelerations of particles from
    their positions. Compares each particle to each other so O(n2), however
    it is very optimized due to only using numpy functions

    Arguments
    ---------
        force: function
            the function that calculates the force between two particles
        m: NDArray
            the mass of the particles
        q: NDArray
            the charge of the particles

    Returns
    -------
     _numpy_pairwise: function
        the function that calculates the accelerations of the particles from
        only their positions
    """

    def _numpy_pairwise(s: NDArray[float]) -> NDArray:
        """
        Calculates the acceleration of each particle from their positions

        Arguments
        ---------
            s: NDArray
                the positions of the particles

        Returns
        -------
            a: NDArray
                the accelerations of each particle
        """
        n = s.shape[0]  # Number of particles
        a = np.zeros_like(s)
        for i in range(1, n):
            r = np.roll(s, i, axis=0) - s  # Vectors between pairs of particles
            q2 = np.roll(q, i)  # Charges of attracting particles
            d = np.linalg.norm(r,
                               axis=1)  # Distances between pairs of particles
            # Direction between particles
            r_norm = np.divide(r, d[:, np.newaxis],
                               where=(d[:, np.newaxis] != 0))
            a += r_norm * (force(d, q, q2) / m)[:, np.newaxis]
        return a

    return _numpy_pairwise


def pairwise(force: Callable[[NDArray, NDArray, NDArray], NDArray],
             m: NDArray, q: NDArray) -> Callable[[NDArray], NDArray]:
    def _pairwise(s):
        """
        Calculates the acceleration of each particle from their positions

        Arguments
        ---------
            s: NDArray
                the positions of the particles

        Returns
        -------
            a: NDArray
                the accelerations of each particle
        """
        a = np.zeros_like(s)
        for i, s_i in enumerate(s[:-1]):
            for j, s_j in enumerate(s[i + 1:], start=i + 1):
                r = s_i - s_j  # Vectors between pairs of particles
                q_i, q_j = q[i], q[j]  # Charges of attracting particles
                m_i, m_j = m[i], m[j]  # Masses of attracting particles
                d = np.linalg.norm(r)  # Distances between pairs of particles
                # Direction between particles
                r_norm = np.divide(r, d, where=(d != 0))

                a[i] -= r_norm * (force(d, q_i, q_j) / m_i)
                a[j] += r_norm * (force(d, q_i, q_j) / m_j)

        return a

    return _pairwise
