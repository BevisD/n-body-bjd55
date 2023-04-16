"""
NAME
    integration
DESCRIPTION
    a collection of functions that numerically integrate the positions and
    velocities of the particles
FUNCTIONS
    euler
    runge_kutta_4
"""

from numpy.typing import NDArray
from typing import Callable


def euler(s: NDArray, v: NDArray, dt: float,
          acc_func: Callable[[NDArray], NDArray]) \
        -> tuple[NDArray, NDArray]:
    """
    Calculates the state after the next time-step using Euler integration

    Arguments
    ---------
        s: NDArray
            Positions of particles
        v: NDArray
            Velocities of particles
        dt: float
            The time-step increment amount
        acc_func: function
            The function used to calculate the accelerations of particles

    Returns
    -------
        s: NDArray
            The updated positions of the particles
        v: NDArray
            The updated velocities of the particles

    """
    a = acc_func(s)
    v += a * dt
    s += v * dt
    return s, v


def runge_kutta_4(s: NDArray, v: NDArray, dt: float,
                  acc_func: Callable[[NDArray], NDArray]) \
        -> tuple[NDArray, NDArray]:
    """
    Calculates the state after the next time-step using RK4 integration

    Arguments
    ---------
        s: NDArray
            Positions of particles
        v: NDArray
            Velocities of particles
        dt: float
            The time-step increment amount
        acc_func: function
            The function used to calculate the accelerations of particles

    Returns
    -------
        s: NDArray
            The updated positions of the particles
        v: NDArray
            The updated velocities of the particles
    """
    k1_x = v
    k1_v = acc_func(s)

    k2_x = v + k1_v * dt * 0.5
    k2_v = acc_func(s + k1_x * dt * 0.5)

    k3_x = v + k2_v * dt * 0.5
    k3_v = acc_func(s + k2_x * dt * 0.5)

    k4_x = v + k3_v * dt
    k4_v = acc_func(s + k3_x * dt)

    s += (k1_x + 2 * k2_x + 2 * k3_x + k4_x) * dt / 6.0
    v += (k1_v + 2 * k2_v + 2 * k3_v + k4_v) * dt / 6.0
    return s, v
