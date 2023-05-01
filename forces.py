"""
NAME
    forces
DESCRIPTION
    a collection of functions that represent different types of forces, and
    their potentials
CLASSES
    Force
    InverseSquare
    Inverse
"""
import numpy as np
from abc import ABCMeta, abstractmethod
from numpy.typing import NDArray
from typing import Union
from particle import Particle

__all__ = ["Force", "InverseSquare", "Inverse"]


class Force(metaclass=ABCMeta):
    """
    An abstract class that holds the required functions needed for a force
    object

    Methods
    -------
        calculate_force()
            calculates the force between two particles, returns a complex force
        calculate_potential()
            calculates the potential between two particles
    """
    @abstractmethod
    def calculate_force(self, particle1: Particle, particle2: Particle) -> \
            complex:
        """
        Calculates the force on particle1 due to the field of particle2

        Arguments
        ---------
            particle1: Particle
                the particle experiencing the force
            particle2: Particle
                the particle causing the force

        Returns
        -------
            force: complex
                the force on particle1, caused by particle2

        """
        pass

    @abstractmethod
    def calculate_potential(self, particle1: Particle, particle2: Particle) -> \
            complex:
        """
        Calculates the potential between two particles

        Arguments
        ---------
            particle1: Particle
                a particle object, order does not matter
            particle2: Particle
                a particle object, order does not matter

        Returns
        -------
            potential: float
                the potential between the two particles
        """
        pass


class InverseSquare(Force):
    """
    A Force class that represents a force with an inverse-square law
    relationship with separation

    Attributes
    ----------
        K: float
            the scaling factor for the force
        SOFTENING: float
            | the softening distance for the force to prevent divergence of
             force
            | for a value of s:
            | force = 1/(r**2+s**2)

    Methods
    -------
        calculate_force()
            calculates the force between two particles, returns a complex force
        calculate_potential()
            calculates the potential between two particles
    """
    def __init__(self, K: float, softening: float = 0.0) -> None:
        self.K = K
        self.SOFTENING = softening

    def calculate_force(self, particle1: Particle, particle2: Particle) -> \
            complex:
        r = particle2.centre - particle1.centre
        d = abs(r)
        force = -self.K * particle1.charge * particle2.charge * r / (
                    d * (d ** 2 + self.SOFTENING ** 2))
        return force

    def calculate_potential(self, particle1: Particle, particle2: Particle) ->\
            float:
        potential = 0
        z = particle1.centre - particle2.centre
        q1 = particle1.charge
        q2 = particle2.charge
        s = self.SOFTENING
        if s == 0:
            potential = -q1*q2 / (4*np.pi*abs(z))
        else:
            potential = q1*q1 / (8*s) * (
                2/np.pi * np.arctan(abs(z)/s) - 1
            )
        return potential


class Inverse(Force):
    """
    A Force class that represent a force with an inverse separation
    relationship

    Attributes
    ----------
        K: float
            the scaling factor for the force

    Methods
    -------
        calculate_force()
            calculates the force between two particles, returns a complex force
        calculate_potential()
            calculates the potential between two particles
    """
    def __init__(self, K: float) -> None:
        self.K = K

    def calculate_force(self, particle1: Particle, particle2: Particle) -> \
            complex:
        z = particle2.centre - particle1.centre
        force = -self.K * particle1.charge * particle2.charge / z.conjugate()
        return force

    def calculate_potential(self, particle1: Particle, particle2: Particle) ->\
            float:
        z = particle2.centre - particle1.centre
        potential = - self.K * particle2.charge * particle1.charge * np.log(z).real
        return potential
