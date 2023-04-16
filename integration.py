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


def euler(particles: list[Particle], algorithm: Algorithm, dt: float) -> None:
    s = np.array([p.centre for p in particles])
    v = np.array([p.velocity for p in particles])

    a = algorithm.calculate_accelerations(particles)
    v += a * dt
    s += v * dt

    for particle, pos, vel in zip(particles, s, v):
        particle.centre = pos
        particle.velocity = vel


def runge_kutta_4(s: NDArray, v: NDArray, dt: float,
                  acc_func: Callable[[NDArray], NDArray]) \
        -> tuple[NDArray, NDArray]:

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
