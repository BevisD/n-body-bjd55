from ..algorithm import Algorithm
from numpy.typing import NDArray
from particle import Particle
from .quadtree import Square, QuadTree
import numpy as np


class BarnesHut(Algorithm):
    def __init__(self, force, theta: float = 0.5):
        self.theta = theta
        self.calculate_force = force.calculate_force
        self.calculate_potential = force.calculate_potential
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

        elif qt.particles is not None and len(qt.particles) != 0:
            centres = [p.centre for p in qt.particles]
            charges = [p.charge for p in qt.particles]

        else:
            qt.boundary.centre_of_charge = Particle(0, 0, 0j, 0j)
            return

        total_charge = np.sum(charges)
        centre_of_charge = np.matmul(charges, centres) / total_charge
        qt.boundary.centre_of_charge = Particle(total_charge, 0,
                                                centre_of_charge)
        return

    def barnes_hut_algorithm(self, qt: QuadTree, particle: Particle):
        centre_of_charge = qt.boundary.centre_of_charge

        interaction_list = []

        d = qt.boundary.d
        r = abs(particle.centre - centre_of_charge.centre)

        if r == 0 or d / r > self.theta:
            if qt.divided:
                for sub_quad in qt.quadrants:
                    particles = self.barnes_hut_algorithm(sub_quad, particle)
                    interaction_list += particles
                return interaction_list
            elif qt.boundary.contains(particle):
                return []
            else:
                return qt.particles
        else:
            return [centre_of_charge]

    def calculate_accelerations(self, particles: list[Particle]) -> \
            NDArray[complex]:
        self.update_quadtree(particles)
        self.calculate_centre_of_charges(self.qt)

        N = len(particles)
        accelerations = np.zeros(N, dtype=complex)

        for i, particle1 in enumerate(particles):
            interaction_list = self.barnes_hut_algorithm(self.qt, particle1)

            for particle2 in interaction_list:
                force = self.calculate_force(particle1, particle2)
                accelerations[i] += force / particle1.mass
        return accelerations

    def animate(self, fig, ax, scatter) -> None:
        return
