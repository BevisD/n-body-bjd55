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


class Force(metaclass=ABCMeta):
    """
    A class to store the force and potential functions for a specific force type

    Methods
    -------
        calculate_force()
            calculates the force between two particles
        calculate_potential()
            calculates the potential between two particles
    """
    @abstractmethod
    def calculate_force(self, *args):
        pass

    @abstractmethod
    def calculate_potential(self, *args):
        pass


class InverseSquare(Force):
    """
        Holds the force and potential functions for an inverse square relationship

        ATTRIBUTES
        ----------
        K: float
            the constant of proportionality for the force e.g. G for gravitation
        SOFTENING: float
            the softening distance to prevent divergent forces
    """
    def __init__(self, K: float, softening: float) -> None:
        self.K = K
        self.SOFTENING = softening

    def calculate_force(self, r: Union[float,NDArray], q1: Union[float,NDArray],
                        q2: Union[float,NDArray]) -> Union[float, NDArray]:
        """
        Calculates the gravitational force between two particles

        Arguments
        ---------
            r: float | NDArray
                the distance between particles
            q1: float | NDArray
                the charge of the current particle, a mass if gravitation
            q2: float | NDArray
                the charge of the other particle, a mass if gravitation

        Returns
        -------
            force: float | NDArray
                the force between each particle
        """
        force = q1 * q2 * self.K / (r ** 2 + self.SOFTENING ** 2)
        return force

    def calculate_potential(self, r: Union[float,NDArray], q1: Union[float,NDArray],
                            q2: Union[float,NDArray]) -> Union[float,NDArray]:
        """
            Calculates the potential between two particles

            Arguments
            ---------
                r: float | NDArray
                    the distance between particles
                q1: float | NDArray
                    the charge of the current particle, a mass if gravitation
                q2: float | NDArray
                    the charge of the other particle, a mass if gravitation

            Returns
            -------
                potential: float | NDArray
                    the potential between the particles
        """
        potential = q1 * q2 * self.K * (np.pi / 2 - np.arctan(r / self.SOFTENING)) / self.SOFTENING
        return potential
