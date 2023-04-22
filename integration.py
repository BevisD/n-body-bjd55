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
from particle import Particle
from algorithms import Algorithm
import numpy as np

__all__ = ["euler", "runge_kutta_4"]


def euler(particles: list[Particle], algorithm: Algorithm, dt: float) -> None:
    """
    Numerically integrates and updates the positions and velocities of the
    particles, using euler integration

    Arguments
    ---------
        particles: list[Particle]
            the particles to update positions and velocities for
        algorithm: Algorithm
            the algorithm class containing the acceleration function
        dt: float
            the time step of numerical integration

    Returns
    -------
        None
    """
    s = np.array([p.centre for p in particles])
    v = np.array([p.velocity for p in particles])

    a = algorithm.calculate_accelerations(particles)
    v += a * dt
    s += v * dt

    for particle, pos, vel in zip(particles, s, v):
        particle.centre = pos
        particle.velocity = vel
    return


def runge_kutta_4(particles: list[Particle], algorithm: Algorithm, dt: float) \
        -> None:
    """
    Numerically integrates and updates the positions and velocities of the
    particles, using RK4 integration

    Arguments
    ---------
       particles: list[Particle]
           the particles to update positions and velocities for
       algorithm: Algorithm
           the algorithm class containing the acceleration function
       dt: float
           the time step of numerical integration

    Returns
    -------
        None
    """

    s = np.array([p.centre for p in particles])
    v = np.array([p.velocity for p in particles])

    k1_x = v
    k1_v = algorithm.calculate_accelerations(particles)

    k2_x = v + k1_v * dt * 0.5
    k2_v = algorithm.calculate_accelerations(particles + k1_x * dt * 0.5)

    k3_x = v + k2_v * dt * 0.5
    k3_v = algorithm.calculate_accelerations(particles + k2_x * dt * 0.5)

    k4_x = v + k3_v * dt
    k4_v = algorithm.calculate_accelerations(particles + k3_x * dt)

    s += (k1_x + 2 * k2_x + 2 * k3_x + k4_x) * dt / 6.0
    v += (k1_v + 2 * k2_v + 2 * k3_v + k4_v) * dt / 6.0

    for centre, velocity, particle in zip(s, v, particles):
        particle.centre = centre
        particle.velocity = velocity
    return
