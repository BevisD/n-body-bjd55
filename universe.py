"""
NAME
    universe
DESCRIPTION
    contains the universe class
CLASSES
    Universe
"""

from algorithms import Algorithm
from particle import Particle
from integration import euler
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numpy.typing import NDArray
import os

__all__ = ["Universe"]


class Universe:
    """
    A class to represent the universe of the simulation

    Attributes
    ----------
        particles: list[Particle]
            the list of particles in the universe
        algorithm: Algorithm
            the algorithm class used to calculate the accelerations of the
            particles
        integration: callable
            the numerical integration function to update the particles with
        dt: float
            the numerical time-step of the universe
        T: float
            the time elapsed for the universe
        periodic_boundary: bool
            the option for periodic boundary conditions for the particles in
            the universe

    Methods
    -------
        update()
            updates the positions and velocities of the particles for one
            time-step
        get_particle_coordinates()
            converts the complex number coordinates into plottable real
            coordinates
        animation()
            displays an animation of motion of the particles in the universe

    """

    def __init__(self, particles: list[Particle], algorithm: Algorithm,
                 dt: float, integration: callable = euler,
                 periodic_boundary: bool = False) -> None:
        self.algorithm = algorithm
        self.particles = particles
        self.integration = integration
        self.periodic_boundary = periodic_boundary
        self.dt = dt
        self.T = 0

    def update(self) -> None:
        """
        Updates the positions and velocities of the particles for one time-step
        using the acceleration algorithm and integration scheme

        Returns
        -------
            None
        """
        self.integration(self.particles, self.algorithm, self.dt)
        if self.periodic_boundary:
            [p.periodic_boundary_conditions() for p in self.particles]
        self.T += self.dt

    def get_particle_coordinates(self) -> tuple[NDArray[float],
    NDArray[float]]:
        """
        Converts complex number coordinates into real plottable coordinates

        Returns
        -------
            x, y: tuple[NDArray[float], NDArray[float]]
                two coordinate arrays representing the x and y components
        """
        s = [(p.centre.real, p.centre.imag) for p in self.particles]
        x, y = np.transpose(s)
        return x, y

    def animation(self, frames: int = 100, iters_per_frame: int = 1,
                  filename: str = None, verbose: int = 0, **kwargs) -> \
            None:
        """
        Displays an animation of the motion of the particles in the universe

        Arguments
        ---------
            frames: int
                the number of frames the animation should last, only has an
                effect if saving to a file
            filename: str
                the name of the file to save the animation to, will save as
                animations/FILENAME
            iters_per_frame: int
                the number of times to update the universe before rendering an
                animation frame
            verbose: int
                how often to print out the frame count

        Returns
        -------
            None
        """
        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        if "text" in kwargs:
            ax.text(.01, 1.01, kwargs["text"], ha="left", va="bottom")
        time_text = ax.text(0.01, 0.99, f"T={self.T:.3f}", ha="left", va="top")
        fig.tight_layout()

        x_0, y_0 = self.get_particle_coordinates()
        scatter = ax.scatter(x_0, y_0, s=1)

        def animate(i):
            if verbose and i % verbose == 0 and i != 0:
                print(f"Frame {i}")

            for _ in range(iters_per_frame):
                self.update()
            coordinates = self.get_particle_coordinates()
            scatter.set_offsets(np.transpose(coordinates))
            self.algorithm.animate(fig, ax, scatter, **kwargs)

            time_text.set_text(f"T={self.T:.3f}")
            return_tuple = tuple([scatter, time_text] + ax.patches)
            return return_tuple

        anim = animation.FuncAnimation(fig, animate, frames, interval=30,
                                       blit=True)

        if filename:
            path = f"animations/{filename}"
            if os.path.isfile(path):
                raise ValueError(f"File already exists - {path}")
            anim.save(path, dpi=300)
        else:
            plt.show()

    def calculate_momentum(self):
        momentum = 0.0j

        for particle in self.particles:
            momentum += particle.mass * particle.velocity

        return momentum

    def calculate_kinetic_energy(self):
        energy = 0.0
        for particle in self.particles:
            energy += 0.5 * particle.mass * abs(particle.velocity) ** 2
        return energy

    def calculate_potential(self):
        energy = 0.0
        for i, particle1 in enumerate(self.particles):
            for particle2 in enumerate(self.particles[i + 1:], start=i + 1):
                energy += self.algorithm.calculate_potential(particle1,
                                                             particle2)
        return energy
