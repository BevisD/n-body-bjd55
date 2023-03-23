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
        return G / (r ** 2 + softening ** 2)
    return _gravitational_force
