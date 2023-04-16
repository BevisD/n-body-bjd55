from particle import Particle
from numpy.typing import NDArray
import numpy as np
from ..algorithm import Algorithm


class PairWise(Algorithm):
    def __init__(self, force):
        self.calculate_force = force.calculate_force
        self.calculate_potential = force.calculate_potential

    def calculate_accelerations(self, particles) -> NDArray[complex]:
        N = len(particles)
        accelerations = np.zeros(N, dtype=complex)
        for i, particle1 in enumerate(particles):
            for j, particle2 in enumerate(particles[i+1:], start=i+1):
                force = self.calculate_force(particle1, particle2)
                accelerations[i] += force / particle1.mass
                accelerations[j] -= force / particle2.mass

        return accelerations

    def animate(self, fig, ax, scatter):
        return

