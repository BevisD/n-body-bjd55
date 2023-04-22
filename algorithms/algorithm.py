"""
NAME
    algorithm
DESCRIPTION
    contains the abstract class 'Algorithm' that holds all necessary
    functionality for an algorithm class
CLASSES
    Algorithm
"""

from abc import ABC, abstractmethod
from numpy.typing import NDArray
from particle import Particle
import matplotlib.pyplot as plt

__all__ = ["Algorithm"]


class Algorithm(ABC):
    """
    The algorithm class is an abstract class that ensures the every algorithm
    has both a calculate_accelerations and animate function

    Methods
    -------
        calculate_accelerations()
            returns an array of complex accelerations for each particle
        animate()
            adds any additional drawings to a matplotlib figure
    """
    def __init__(self):
        self.calculate_potential = None

    @abstractmethod
    def calculate_accelerations(self, particles: list[Particle]) -> \
            NDArray[complex]:
        """
        calculates the accelerations for each particle

        Arguments
        ---------
            particles: list[Particle]
                the particle objects to calculate the accelerations for

        Returns
        -------
            NDArray[complex]
                the complex accelerations for each particle
        """
        return NDArray[complex]

    @abstractmethod
    def animate(self, fig: plt.Figure, ax: plt.Axes, scatter: plt.Subplot,
                **kwargs) -> None:
        """
        Draws any additional information that may be useful to visualise for
        an algorithm e.g. Quadtree boundaries etc...
        """
        return


