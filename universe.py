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


class Universe:
    '''
    A class to represent the universe of the simulation

    Attributes
    __________
    N : int
        where N is stored
    G : float
        where G is stored
    s : np.ndarray
        where s is stored
    v : np.ndarray
        where v is stored
    m : np.ndarray
        where m is stored
    SIZE : float
        where world_size is stored
    MARKER_SIZE : float
        where point_size is stored

    Methods
    -------
    create_figure()
        creates a figure for the particle positons to be displayed
    render()
        displays the current positions of the universe's particles
    '''

    def __init__(self,
                 N:int,
                 G:float,
                 s: np.ndarray,
                 v: np.ndarray,
                 m: np.ndarray = None,
                 world_size: float = 1.0,
                 point_size: float = 1.0):

        self.N = N
        self.G = G
        self.s = s
        self.v = v
        self.a = np.zeros_like(self.v)  # initialise acceleration to zeros
        self.m = m if m else np.ones(N)  # initialise masses to ones if not provided
        self.SIZE = world_size
        self.MARKER_SIZE = point_size

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
            None

        Returns
        -------
            fig: matplotlib.figure.Figure
            ax: matplotlib.axes._axes.Axes
        '''
        fig, ax = plt.subplots()
        ax.set_xlim(-self.SIZE, self.SIZE)
        ax.set_ylim(-self.SIZE, self.SIZE)
        ax.set_aspect("equal")
        return fig, ax

    def render(self):
        '''
        Displays the current positions of all the particles

        Arguments
        _________
            None

        Returns
        _______
            None
        '''
        fig, ax = self.create_figure()
        x, y = self.s.T
        ax.scatter(x, y)
        plt.show()