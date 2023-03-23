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
from forces import gravitational_force


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

        self.N = n
        self.G = g
        self.DT = dt
        self.s = s
        self.v = v
        self.a = np.zeros_like(self.v)  # initialise acceleration to zeros
        self.m = m if m else np.ones(n)  # initialise masses to ones if not provided
        self.SOFTENING = softening
        self.SIZE = world_size
        self.MARKER_SIZE = point_size
        self.integrate = euler
        self.force = gravitational_force(g, softening)
        self.calc_acceleration = numpy_pairwise(self.force)

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
        fig, ax = plt.subplots(figsize=(6,6))
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

    def update(self):
        '''
        Updates the positions and velocities of each particle for the next time-step

        Arguments
        ---------
            self

        Returns
        -------
            None
        '''
        self.s, self.v = self.integrate(self.s, self.v, self.DT, self.calc_acceleration)

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
