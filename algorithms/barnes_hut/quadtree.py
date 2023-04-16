"""
NAME
    quadtree
DESCRIPTION
    contains classes needed for a quadtree structure
CLASSES
    Square
    Quadtree
"""

from __future__ import annotations
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from particle import Particle
from typing import List


class Square:
    """
    A class to store the location of the squares in a quadtree

    Attributes
    ----------
        x, y: float
            the x and y position coordinates of the centre of the square
        d: float
            the width of the square
        r: float
            half the width of the square
        centre_of_charge: Point
            the centre of charge of the square

    Methods
    -------
        contains()
            determines whether a point lies within the square
        subdivide()
            creates 4 new sub-quadrants for the square

    """

    def __init__(self, centre: complex, d: float) -> None:
        self.centre = centre
        self.d = d
        self.r = d / 2
        self.centre_of_charge = None

    def __repr__(self) -> str:
        return f"Square({self.centre}, {self.d})"

    def contains(self, particle: Particle) -> bool:
        """
        Calculates whether a point lies within the square

        Arguments
        ---------
            particle: Particle
                the point to test

        Returns
        -------
            does_contain: bool
                whether the point lies within the square

        """
        x1 = self.centre.real
        y1 = self.centre.imag
        x2 = particle.centre.real
        y2 = particle.centre.imag
        does_contain = (x1 - self.r <= x2 < x1 + self.r) and \
                       (y1 - self.r <= y2 < y1 + self.r)
        return does_contain

    def subdivide(self, quadrant: int) -> Square:
        """
        Creates 4 squares representing the sub-quadrants of the sub-quadtree

        Arguments
        ---------
            quadrant: int
                the quadrant number
                NW: 0
                NE: 1
                SW: 2
                SE: 3

        Returns
        -------
            square: Square
                the sub-quadrant for the new sub-quadtree

        """
        offsets = [
            complex(-self.r / 2, +self.r / 2),
            complex(+self.r / 2, +self.r / 2),
            complex(-self.r / 2, -self.r / 2),
            complex(+self.r / 2, -self.r / 2)
        ]
        return Square(self.centre + offsets[quadrant], self.r)


class QuadTree:
    """
    A class to implement a quadtree

    Attributes
    ----------
        boundary: Square
            the boundary of the quadrant
        points: List[Point]
            the list of points inside the quadrant
        divided: bool
            whether the quadrant has subdivided
        max_d: int
            the maximum recursion depth of the quadtree
        quadrants: List[Square]
            the list of sub-quadrants if the quadtree has divided

    Methods
    -------

    """

    def __init__(self, boundary: Square, max_depth: int = 6) -> None:
        self.boundary = boundary
        self.particles = []
        self.divided = False
        self.max_d = max_depth
        self.quadrants = None

    def clear(self) -> None:
        """
        Empties the quadtree and resets its structure

        Returns
        -------
            None
        """
        self.particles = []
        if self.divided:
            self.divided = False
            del self.quadrants

    def subdivide(self) -> None:
        """
        Subdivides the quadtree into four smaller sub-quad-trees

        Returns
        -------
            None
        """
        self.quadrants = [QuadTree(self.boundary.subdivide(i), self.max_d - 1)
                          for i in range(4)]
        self.divided = True

        for p in self.particles:
            for quad in self.quadrants:
                if quad.insert(p):
                    break
        self.particles = None

    def insert(self, particle: Particle) -> bool:
        """
        Inserts a point into the quadtree

        Arguments
        ---------
            p: Point
                the point to insert into the quadtree

        Returns
        -------
            bool
        """
        if not self.boundary.contains(particle):
            return False

        if not self.divided:
            if len(self.particles) == 0 or self.max_d <= 0:
                self.particles.append(particle)
                return True
            self.subdivide()

        inserted = False
        for quad in self.quadrants:
            if quad.insert(particle):
                inserted = True
                break
        return inserted

    def draw_quads(self, ax: plt.Axes) -> None:
        """
        Draws the boundaries of the quadtree onto the axes

        Arguments
        ---------
            ax: plt.Axes
                the axes to draw the squares onto

        Returns
        -------
            None
        """
        if self.divided:
            for quad in self.quadrants:
                quad.draw_quads(ax)
        else:
            x = self.boundary.centre.real
            y = self.boundary.centre.imag
            r = self.boundary.r
            d = self.boundary.d
            patch = patches.Rectangle((x - r, y - r), d, d, edgecolor="black",
                                      facecolor="none", linewidth=0.5)
            ax.add_patch(patch)
