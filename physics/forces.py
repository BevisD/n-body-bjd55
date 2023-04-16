"""
NAME
    forces
DESCRIPTION
    a collection of functions that represent different types of forces, and
    their potentials
CLASSES
    Force
    InverseSquare
"""
import numpy as np
from abc import ABCMeta, abstractmethod
from numpy.typing import NDArray
from typing import Union
from particle import Particle


class Force(metaclass=ABCMeta):

    @abstractmethod
    def calculate_force(self, particle1: Particle, particle2: Particle):
        pass

    @abstractmethod
    def calculate_potential(self, particle1: Particle, particle2: Particle):
        pass


class InverseSquare(Force):

    def __init__(self, K: float, softening: float) -> None:
        self.K = K
        self.SOFTENING = softening

    def calculate_force(self, particle1: Particle, particle2: Particle) -> \
            Union[float, NDArray]:
        r = particle2.centre - particle1.centre
        d = abs(r)
        force = self.K * particle1.charge * particle2.charge * r / (
                    d * (d ** 2 + self.SOFTENING ** 2))
        return force

    def calculate_potential(self, particle1: Particle, particle2: Particle) ->\
            Union[float, NDArray]:
        potential = 0
        # potential = q1 * q2 * self.K * (
        #             np.pi / 2 - np.arctan(r / self.SOFTENING)) / self.SOFTENING
        return potential
