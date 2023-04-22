"""
NAME
    pair_wise
DESCRIPTION
    contains the PairWise Algorithm class
CLASSES
    PairWise
"""

from particle import Particle
from numpy.typing import NDArray
import numpy as np
from ..algorithm import Algorithm
from forces import Force
import matplotlib.pyplot as plt

__all__ = ["PairWise"]


class PairWise(Algorithm):
    """
    The class that holds all the functionality needed for the pair-wise
    algorithm

    Attributes
    ----------
        self.calculate_force: callable
            the function that calculates the force between two particles
        self.calculate_potential:
            the function that calculates the potential between two particles

    Methods
    -------
        calculate_accelerations()
            calculates the accelerations for each particle using the pair-wise
            algorithm
        animate()
            draws additional functionality that may be useful for the animation
    """
    def __init__(self, force: Force) -> None:
        self.calculate_force = force.calculate_force
        self.calculate_potential = force.calculate_potential

    def calculate_accelerations(self, particles: list[Particle]) -> \
            NDArray[complex]:
        """
        calculates the accelerations for each particle using the pair-wise
        algorithm

        Arguments
        ---------
            particles: list[Particle]
                the list of particles to calculate accelerations for

        Returns
        -------
            accelerations: NDArray[complex]
                the accelerations of each particle represented as a complex
                number
        """
        N = len(particles)
        accelerations = np.zeros(N, dtype=complex)
        for i, particle1 in enumerate(particles):
            for j, particle2 in enumerate(particles[i+1:], start=i+1):
                force = self.calculate_force(particle1, particle2)
                accelerations[i] += force / particle1.mass
                accelerations[j] -= force / particle2.mass

        return accelerations

    def animate(self, fig: plt.Figure, ax: plt.Axes, scatter: plt.Subplot,
                **kwargs):
        """
        Draws additional functionality that may be useful for the animation

        Arguments
        ---------
            fig: plt.Figure
                the matplotlib figure to draw onto
            ax: plt.Axes
                the matplotlib axes to draw onto
            scatter: plt.Subplot
                the matplotlib scatter plot to adjust
        """
        return

