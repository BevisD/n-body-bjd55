"""
NAME
    universe
DESCRIPTION
    This module contains the Universe class
CLASSES
    Universe
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from integration import euler
from acceleration import numpy_pairwise
from forces import InverseSquare
from numpy.typing import NDArray
from typing import Callable


class Universe:
    """
    A class to represent the universe of the simulation

    Attributes
    ----------
    N: int
        the particle number of the universe
    G: float
        the gravitational constant of the universe
    DT: float
        the time-step increment of the universe
    T: float
        the time elapsed from the initial conditions
    s: NDArray
        the positions of particles in the universe
    v: NDArray
        the velocities of particles in the universe
    a: NDArray
        the accelerations of particles in the universe
    m: NDArray
        the masses of particles in the universe
    q: NDArray
        the charges of particles in the universe, equal to masses for gravitation
    SOFTENING: float
        the factor to reduce divergence of force between near particles
    SIZE: float
        the distance from the centre to an edge of the universe
    MARKER_SIZE: float
        the matplotlib display size of each particle
    integrate: function
        the numerical integration scheme to use
    force: function
        the function of force between each particle
    calc_acceleration: function
        the function to calculate the accelerations of each particle

    Methods
    -------
    create_figure()
        creates a figure for the particle positions to be displayed
    render()
        displays the current positions of the universe's particles
    update()
        calculates the positions and velocities of the particles in the next time-step
    animation()
        produces an animation of the motions of the particles
    calc_kinetic()
        calculates the total kinetic energy of the universe
    calc_potential()
        calculates the total potential energy of the universe
    record_energies()
        keeps a record of the different types of energy
    """

    def __init__(self,
                 n: int,
                 g: float,
                 softening: float,
                 dt: float,
                 s: NDArray,
                 v: NDArray,
                 m: NDArray = None,
                 q: NDArray = None,
                 world_size: float = 1.0,
                 point_size: float = 1.0) -> None:

        # Define universe constants
        self.N = n  # Number of particles in the universe
        self.G = g  # Gravitational constant of the universe
        self.DT = dt  # Time-step dt of the universe
        self.T = 0  # Time elapsed since the beginning of the universe
        self.SOFTENING = softening  # The softening factor for gravity
        self.SIZE = world_size  # The radius of the universe
        self.MARKER_SIZE = point_size  # The display size of each particle

        self.s = s  # Positions of particles in the universe
        self.v = v  # Velocities of particles in the universe
        self.a = np.zeros_like(self.v)  # initialise acceleration to zeros
        # initialise masses to ones if not provided
        self.m = m if m else np.ones(n)
        self.q = q if q else np.ones(n)

        # The integration scheme to use, set to euler by default
        self.integrate = euler
        # The type of force between particles, set to gravity by default
        force_type = InverseSquare(g, softening)
        self.force = force_type.calculate_force
        #  The potential energy due to the force defined above
        self.potential = force_type.calculate_potential
        #  The algorithm to apply the force to each particle
        self.calc_acceleration = numpy_pairwise(self.force, self.m, self.q)

        self.pot_hist = []  # The array to store potential energies
        self.kin_hist = []  # The array to store kinetic energies

    def create_figure(self) -> tuple[plt.Figure, plt.Axes]:
        """
        Creates a matplotlib figure and axis for the particle positions

        Returns
        -------
            fig: plt.Figure
            ax: plt.Axes
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-self.SIZE, self.SIZE)
        ax.set_ylim(-self.SIZE, self.SIZE)
        ax.set_aspect("equal")
        return fig, ax

    def render(self) -> None:
        """
        Displays the current positions of all the particles

        Returns
        -------
            None
        """
        fig, ax = self.create_figure()
        x, y = self.s.T
        ax.scatter(x, y, s=self.MARKER_SIZE)
        plt.show()

    def calc_kinetic(self) -> float:
        """
        Calculates the total kinetic energy of the universe

        Returns
        -------
            kinetic_energy: float
                total kinetic energy of the universe
        """
        kinetic_energy = 0.5 * self.m @ (np.linalg.norm(self.v, axis=1) ** 2)
        return kinetic_energy

    def calc_potential(self) -> float:
        """
        Calculates the total potential energy of the universe
        using a numpy pairwise algorithm

        Returns
        -------
            potential: float
                total potential energy of the system
        """
        potential = 0

        # Iterate over all pairs of particles
        for i in range(1, self.N):
            # Get vector between particles separated by index i
            r = np.roll(self.s, i, axis=0) - self.s
            d = np.linalg.norm(r, axis=1)  # Distance between particles
            q2 = np.roll(self.q, i)  # Charges of the other particle
            # Factor of 0.5 as pairs are counted twice
            potential -= 0.5 * np.sum(self.potential(d, self.q, q2))
        return potential

    def record_energies(self) -> None:
        """
        Keeps a record of the Kinetic and Potential energy

        Returns
        -------
            None
        """
        potential = self.calc_potential()
        kinetic = self.calc_kinetic()
        self.kin_hist.append(kinetic)
        self.pot_hist.append(potential)

    def plot_energies(self, filename: str = None) -> None:
        """
        Plots the history of the energy of the universe at every point self.record_energies
        was called. The total energy of the universe should be constant

        Arguments
        ---------
            filename: str
                the name of the file to save the figure to

        Returns
        -------
            None
        """
        fig, ax = plt.subplots()
        times = np.linspace(0, self.T, len(self.kin_hist))

        # Calculate changes in energies
        kinetic = np.subtract(self.kin_hist, self.kin_hist[0])
        potential = np.subtract(self.pot_hist, self.pot_hist[0])
        energy = np.add(kinetic, potential)

        ax.plot(times, kinetic, label="KE", color="blue")
        ax.plot(times, potential, label="V", color="red")
        ax.plot(times, energy, label="E", color="black")
        ax.set_xlabel("Time")
        ax.set_ylabel("Energy")
        ax.set_title(f"Energy against Time, DT={self.DT}, G={self.G}, N={self.N}")
        plt.legend()
        if filename:
            plt.savefig(f"figures/{filename}", dpi=300)
        else:
            plt.show()

    def update(self, record_energies: bool = False) -> None:
        """
        Updates the positions and velocities of each particle for the next time-step
        Optionally records the energy at each time-step

        Arguments
        ---------
            record_energies: bool
                option to calculate and store the potential and kinetic energies
                slows down performance, so only use if needed

        Returns
        -------
            None
        """
        if record_energies:
            self.record_energies()
        self.s, self.v = self.integrate(self.s, self.v, self.DT, self.calc_acceleration)
        self.T += self.DT

    def animation(self, frames: int = 100, iterations_per_frame: int = 1,
                  filename: str = None, **kwargs) -> None:
        """
        Displays an animation of the positions of the particles over time

        Arguments
        ---------
            frames: int
                The number of frames that the animation should last. Only has an
                effect when saving the animation to a file
            iterations_per_frame: int
                The number of times to call update between renders
            filename: str
                The name of the file to save the animation to

        Returns
        -------
            None
        """
        fig, ax = self.create_figure()
        x, y = self.s.T
        scatter = ax.scatter(x, y, s=self.MARKER_SIZE)
        animate = self._animate(ax, scatter, iterations_per_frame, **kwargs)
        anim = animation.FuncAnimation(fig, animate, frames, interval=20)
        if filename:
            writer = animation.PillowWriter(fps=30)
            anim.save(f"animations/{filename}", writer=writer)
        else:
            plt.show()

    def _animate(self, ax: plt.Axes, scatter: plt.scatter,
                 iterations_per_frame: int, **kwargs) -> Callable[[int], None]:
        """
        Creates the animation function that has only the frame count as an argument

        Arguments
        ---------
            ax: plt.Axes
                the axes to draw the animation onto
            scatter: plt.scatter
                the scatter plot that displays the particle positions
            iterations_per_frame: int
                the number of times to call update before rendering the frame

        Returns
        -------
            animate: function
                the animation function that is called each frame

        """
        def animate(_: int) -> None:
            """
            Updates the position of the particles on the figure

            Arguments
            ---------
                _: int
                    the frame count

            Returns
            -------
                None
            """
            for _ in range(iterations_per_frame):
                self.update()
            scatter.set_offsets(self.s)
        return animate
