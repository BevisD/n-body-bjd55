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

    def calculate_centre_of_charges(self, qt: QuadTree):
        if qt.divided:
            centre_of_charges = []
            for sub_quad in qt.quadrants:
                self.calculate_centre_of_charges(sub_quad)
                centre_of_charges.append(sub_quad.boundary.centre_of_charge)

            centres = [p.centre for p in centre_of_charges]
            charges = [p.charge for p in centre_of_charges]

        elif qt.points is not None and len(qt.particles) != 0:
            centres = [p.centre for p in qt.particles]
            charges = [p.charge for p in qt.particles]

        else:
            return Particle(0, 0, 0j, 0j)

        total_charge = np.sum(charges)
        centre_of_charge = np.matmul(charges, centres) / total_charge
        qt.boundary.centre_of_charge = Particle(total_charge, 0,
                                                centre_of_charge)
        return

    def calculate_accelerations(self, particles: list[Particle]) -> \
            NDArray[complex]:
        return NDArray[complex]

    def animate(self, fig, ax, scatter) -> None:
        return
