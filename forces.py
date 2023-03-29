'''
NAME
    forces
DESCRIPTION
    a collection of functions that represent different types of forces, and
    their potentials
FUNCTIONS
    gravitational_force
'''
import numpy as np
from abc import ABCMeta, abstractmethod


class Force(metaclass=ABCMeta):
    @abstractmethod
    def calculate_force(self, *args):
        pass

    @abstractmethod
    def calculate_potential(self, *args):
        pass


class Gravity(Force):
    def __init__(self, G, softening):
        self.G = G
        self.SOFTENING = softening

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
    def calculate_force(self, r):
        '''
        Calculates the gravitational force for a given distance and softening

        Arguments
        ---------
            r: float
                the distance between particles
            m1: float
                the mass of the current particle
            m2: float
                the mass of the other particle

        Returns
        -------
            force: float
                the gravitational force between each particle
        '''
        force = self.G / (r ** 2 + self.SOFTENING ** 2)

        return force

    def calculate_potential(self, r):
        potential = np.sum(self.G * (np.pi / 2 - np.arctan(r / self.SOFTENING)) / self.SOFTENING)
        return potential
