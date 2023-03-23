def euler(s, v, dt, acc_func):
    '''
    Calculates the state after the next time-step using Euler integration

    Arguments
    ---------
        s: np.ndarray
            Positions of particles
        v: np.ndarray
            Velocities of particles
        dt: float
            The time-step increment amount
        acc_func: function
            The function used to calculate the accelerations of particles

    Returns
    -------
        s: np.ndarray
            The updated positions of the particles
        v: np.ndarray
            The updated velocities of the particles

    '''
    a = acc_func(s)
    v += a * dt
    s += v * dt
    return s, v
