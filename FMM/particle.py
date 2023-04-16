"""
NAME
    particle
DESCRIPTION
    contains the particle class
CLASSES
    Particle
"""

import numpy as np
from index import Index


class Particle:
    """
    A class to store the functionality of a particle object

    Attributes
    ----------
        charge: float
            the charge of the particle
        mass: float
            the mass of the particle
        centre: complex
            the position of the particle,
            the real component is the x coordinate
            the imag component is the y coordinate

    Methods
    -------
        index
            returns the index of the particle for a certain level in the
            quadtree
        __key
            returns a hashable key for the particle object
    """

    def __init__(self, charge: float = None, mass: float = None,
                 centre: complex = None) -> None:
        if charge is not None:
            self.charge = charge
        else:
            self.charge = np.random.uniform(-1, 1)
        self.mass = mass if mass else 1.0

        if centre:
            self.centre = centre
        else:
            self.centre = np.random.random() + 1j * np.random.random()

        self.potential = 0

    def index(self, level: int) -> Index:
        """
        returns the index of the cell that the particle lies within, at a
        certain level of the quadtree

        Arguments
        ---------
            level: int
                the depth of the cell in the quadtree to calculate the index

        Returns
        -------
            Index
                the Index object corresponding to the particles position
        """
        i = int(self.centre.imag * 2 ** level)
        j = int(self.centre.real * 2 ** level)

        return Index(i, j, level)

    def __repr__(self) -> str:
        return f"Particle({self.charge}, {self.centre}, {self.potential})"

    def __eq__(self, other) -> bool:
        return (self.charge == other.charge and
                self.centre == other.centre)

    def __key(self) -> tuple[float, complex]:
        """
        generates a hashable key for the object

        Returns
        -------
            tuple[float, complex]
                the hashable key for the particle containing the charge and
                position of the particle
        """
        return self.charge, self.centre

    def __hash__(self) -> int:
        return hash(self.__key())
