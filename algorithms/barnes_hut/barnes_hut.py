"""
NAME
    barnes_hut
DESCRIPTION
    contains the BarnesHut Algorithm class to implement the Barnes-Hut
    algorithm for some given particles
CLASSES
    BarnesHut
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ..algorithm import Algorithm
from numpy.typing import NDArray
from particle import Particle
from .quadtree import Square, QuadTree
import numpy as np

__all__ = ["BarnesHut"]


class BarnesHut(Algorithm):
    """
    A class to hold all the functionality needed to implement the Barnes-Hut
    algorithm

    Attributes
    ----------
        theta: float
            the critical ratio of cell width to cell distance that decides
            whether to approximate in the algorithm
        calculate_force: callable
            the function that returns the force between two particles
        calculate_potential: callable
            the function that returns the potential between two particles
        qt: QuadTree
            the quadtree structure that holds the particles for the algorithm

    Methods
    -------
        update_quadtree()
            clears the quadtree and refills it with particles
        calculate_centre_of_charges()
            recursively updates each boundary in the quadtree with its centre
            of mass
        barnes_hut_algorithm()
            executes the Barnes-Hut algorithm and returns the interacton list
            for a given particle
        calculate_accelerations()
            calculates the accelerations for each particle using the Barnes-Hut
            algorithm
        animate()
            draws additional information to the animation. e.g. Quadtree
            boundaries
    """

    def __init__(self, force, theta: float = 0.5) -> None:
        self.theta = theta
        self.calculate_force = force.calculate_force
        self.calculate_potential = force.calculate_potential
        self.qt = QuadTree(Square(complex(0.5, 0.5), 1), np.inf)

    def update_quadtree(self, particles: list[Particle]) -> None:
        """
        removes all particles from the quadtree and adds new particles back in

        Arguments
        ---------
            particles: list[Particle]
                the new particles to add to the quadtree

        Returns
        -------
            None
        """
        self.qt.clear()
        for particle in particles:
            self.qt.insert(particle)

    def calculate_centre_of_charges(self, qt: QuadTree) -> None:
        """
        Recursively calculates and updated the centre of charges at for each
        quadrant at each level of the quadtree. Centre of charges are given as
        particles with zero mass

        Arguments
        ---------
            qt: QuadTree
                the quadtree to calculate the centre of mass for, when called
                this is the highest level of the quadtree

        Returns
        -------
            None
        """
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

        total_charge = float(np.sum(charges))
        centre_of_charge = np.matmul(charges, centres) / total_charge
        qt.boundary.centre_of_charge = Particle(total_charge, 0,
                                                centre_of_charge)
        return

    def barnes_hut_algorithm(self, qt: QuadTree, particle: Particle) -> \
            list[Particle]:
        """
        executes the Barnes-Hut algorithm, returning the interaction list for
        a specific particle

        Arguments
        ---------
            qt: QuadTree
                the quadtree containing the particles for the Barnes-Hut
                algorithm
            particle: Particle
                the particle to calculate the interaction list for

        Returns
        -------
            interaction_list: list[Particle]
                the list of particles the particle should interact with, centre
                of charges are given as particles with zero mass
        """
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
        """
        calculates the accelerations for each particle using the Barnes-Hut
        algorithm. Accelerations are given as complex numbers with the
        real/imaginary components equal to the x/y components of acceleration

        Arguments
        ---------
            particles: list[Particle]
                the list of particles to calculate accelerations for

        Returns
        -------
            accelerations: NDArray[complex]
                the accelerations of each particle given as a complex number
        """
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

    def draw_cluster_squares(self, qt: QuadTree, particle: Particle,
                             ax: plt.Axes) -> None:
        centre_of_charge = qt.boundary.centre_of_charge

        d = qt.boundary.d
        r = abs(particle.centre - centre_of_charge.centre)

        if qt.divided:
            if r == 0 or d / r > self.theta:
                for sub_quad in qt.quadrants:
                    self.draw_cluster_squares(sub_quad, particle, ax)
                return

        elif len(qt.particles) == 0:
            return

        x = qt.boundary.centre.real
        y = qt.boundary.centre.imag
        r = qt.boundary.r
        d = qt.boundary.d
        patch = patches.Rectangle((x - r, y - r), d, d,
                                  edgecolor="red",
                                  facecolor="none", linewidth=0.5)
        ax.add_patch(patch)
        return

    def animate(self, fig: plt.Figure, ax: plt.Axes, scatter: plt.Subplot,
                show_squares: bool = False, barnes_hut_point: Particle = None,
                **kwargs) -> None:
        """
        Draws any additional information that may be useful to the animation
        e.g. quadtree boundaries

        Arguments
        ---------
            fig: plt.figure
                the matplotlib figure of the plot
            ax: plt.Aces
                the matplotlib axes of the plot
            scatter: plt.Subplot
                the matplotlib scatter plot
            show_squares: bool
                option whether to show the quadtree boundaries

        Returns
        -------
            None
        """
        if show_squares:
            [p.remove() for p in reversed(ax.patches)]

            if barnes_hut_point:
                self.draw_cluster_squares(self.qt, barnes_hut_point, ax)
            else:
                self.qt.draw_quads(ax)
        return
