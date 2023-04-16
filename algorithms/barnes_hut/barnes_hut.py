from algorithms import Algorithm
from numpy.typing import NDArray
from particle import Particle
from quadtree import Square, QuadTree
import numpy as np


class BarnesHut(Algorithm):
    def __init__(self, theta: float = 0.5):
        self.theta = theta
        self.qt = QuadTree(Square(complex(0.5, 0.5), 1), np.inf)

    def update_quadtree(self, particles: list[Particle]) -> None:
        self.qt.clear()
        for particle in particles:
            self.qt.insert(particle)

    def barnes_hut_algorithm(self, qt, particle: Particle):
        if qt.divided:


    def calculate_accelerations(self, particles: list[Particle]) -> \
            NDArray[complex]:
        return NDArray[complex]

    def animate(self, fig, ax, scatter) -> None:
        return
