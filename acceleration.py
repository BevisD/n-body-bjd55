'''
NAME
    acceleration
DESCRIPTION
    a collection of functions that calculates the accelerations of each particle
FUNCTIONS:
    numpy_pairwise
'''

import numpy as np

def numpy_pairwise(force):
    '''
    Creates a function to calculate the accelerations of particles from their positions.
    Compares each particle to each other so O(n2), however it is very optimized due to only
    using numpy functions

    Arguments
    ---------
        force: function
            the function that calculates the force between two particles

    Returns
    -------
        _numpy_pairwise: function
            the function that calculates the accelerations of the particles from only
            their positions
    '''
    def _numpy_pairwise(s):
        '''
        Calculates the acceleration of each particle from their positons

        Arguments
        ---------
            s: np.ndarray
                the positions of the particles

        Returns
        -------
            a: np.ndarray
                the accelerations of each particle
        '''
        n = s.shape[0]
        a = np.zeros_like(s)
        for i in range(1, n):
            r = np.roll(s, i, axis=0) - s
            d = np.linalg.norm(r, axis=1)[:, np.newaxis]
            r_norm = np.divide(r, d, where=(d != 0))
            a += r_norm * force(r)
        return a
    return _numpy_pairwise
