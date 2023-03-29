'''
NAME
    universe
DESCRIPTION
    This module contains the universe class
CLASSES
    Universe
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from integration import euler
from acceleration import numpy_pairwise
from forces import Gravity


class Universe:
    '''
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
    s: np.ndarray
        the positions of particles in the universe
    v: np.ndarray
        the velocities of particles in the universe
    a: np.ndarray
        the accelerations of particles in the universe
    m: np.ndarray
        the masses of particles in the universe
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
    '''

    def __init__(self,
                 n: int,
                 g: float,
                 softening: float,
                 dt: float,
                 s: np.ndarray,
                 v: np.ndarray,
                 m: np.ndarray = None,
                 world_size: float = 1.0,
                 point_size: float = 1.0):

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

        # The integration scheme to use, set to euler by default
        self.integrate = euler
        # The type of force between particles, set to gravity by default
        gravity = Gravity(g, softening)
        self.force = gravity.calculate_force
        #  The potential energy due to the force defined above
        self.potential = gravity.calculate_potential
        #  The algorithm to apply the force to each particle
        self.calc_acceleration = numpy_pairwise(self.force)

        self.pot_hist = []  # The array to store potential energies
        self.kin_hist = [] # The array to store kinetic energies

    def __setattr__(self, key, value):
        '''Prevents the modification of N and G after the Universe has been initialised'''
        if key == "N" and hasattr(self, "N"):
            raise AttributeError("Cannot modify N")
        elif key == "G" and hasattr(self, "G"):
            raise AttributeError("Cannot modify G")
        else:
            self.__dict__[key] = value

    def __delattr__(self, item):
        '''Prevents the deletion of any attributes of the Universe class'''
        raise AttributeError(f"Cannot delete {item}")

    def create_figure(self):
        '''
        Creates a matplotlib figure and axis for the particle positions

        Arguments
        ---------
            self

        Returns
        -------
            fig: matplotlib.figure.Figure
            ax: matplotlib.axes._axes.Axes
        '''
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-self.SIZE, self.SIZE)
        ax.set_ylim(-self.SIZE, self.SIZE)
        ax.set_aspect("equal")
        return fig, ax

    def render(self):
        '''
        Displays the current positions of all the particles

        Arguments
        ---------
            self

        Returns
        -------
            None
        '''
        fig, ax = self.create_figure()
        x, y = self.s.T
        ax.scatter(x, y, s=self.MARKER_SIZE)
        plt.show()

    def calc_kinetic(self):
        '''
        Calculates the total kinetic energy of the universe

        Arguments
        ---------
            self

        Returns
        -------
            kinetic_energy: float
                total kinetic energy of the universe
        '''
        kinetic_energy = 0.5 * self.m @ (np.linalg.norm(self.v, axis=1) ** 2)
        return kinetic_energy

    def calc_potential(self):
        '''
        Calculates the total potential energy of the universe

        Arguments
        ---------
            self

        Returns
        -------
            potential: float
                total potential energy of the system
        '''
        potential = 0

        # Iterate over all pairs of particles
        for i in range(1, self.N):
            # Get vector between particles separated by index i
            r = np.roll(self.s, i, axis=0) - self.s
            d = np.linalg.norm(r, axis=1) # Distance between particles

            # Factor of 0.5 as pairs are counted twice
            potential -= 0.5 * np.sum(self.potential(d))
        return potential

    def record_energies(self):
        '''
        Keeps a record of the Kinetic and Potential energy

        Arguments
        ---------
            self

        Returns
        -------
            None
        '''
        potential = self.calc_potential()
        kinetic = self.calc_kinetic()
        self.kin_hist.append(kinetic)
        self.pot_hist.append(potential)

    def plot_energies(self):
        '''
        Plots the history of the energy of the universe at every point self.record_energies
        was called. The total energy of the universe should be constant

        Arguments
        ---------
            self

        Returns
        -------
            None
        '''
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
        plt.show()

    def update(self):
        '''
        Updates the positions and velocities of each particle for the next time-step
        Optionally records the energy at each timestep

        Arguments
        ---------
            self

        Returns
        -------
            None
        '''
        self.record_energies()
        self.s, self.v = self.integrate(self.s, self.v, self.DT, self.calc_acceleration)
        self.T += self.DT

    def animation(self, frames=100):
        '''
        Displays an animation of the positions of the particles over time

        Arguments
        ---------
            frames: int
                The number of frames that the animation should last. Only has an
                effect when saving the animation to a file

        Returns
        -------
            None
        '''
        fig, ax = self.create_figure()
        x, y = self.s.T
        scatter = ax.scatter(x, y, s=self.MARKER_SIZE)

        def animate(i):
            '''
            Updates the position of the particles on the figure

            Arguments
            ---------
                i: int
                    the frame count

            Returns
            -------
                None
            '''
            self.update()
            scatter.set_offsets(self.s)

        anim = animation.FuncAnimation(fig, animate, frames, interval=20)
        plt.show()
