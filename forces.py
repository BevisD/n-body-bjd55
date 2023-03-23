'''
NAME
    forces
DESCRIPTION
    a collection of functions that represent different types of forces, and
    their potentials
FUNCTIONS
    gravitational_force
'''


def gravitational_force(G, softening):
    '''
    Creates a force function that only requires the distance between two particles

    Arguments
    ---------
        G: float
            gravitational constant
        softening: float
            the factor that prevents the divergence of the force at small distances

    Returns
    -------
        _gravitational_force: function
            the force function that is only a function of distance
    '''
    def _gravitational_force(r):
        '''
        Calculates the gravitational force for a given distance and softening

        Arguments
        ---------
            r: np.ndarray
                the distance between particles

        Returns
        -------
            force: float
                the gravitational force between each particle
        '''
        force = G / (r ** 2 + softening ** 2)
        return force
    return _gravitational_force
