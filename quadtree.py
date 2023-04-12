"""
NAME
    quadtree
DESCRIPTION
    contains classes needed for a quadtree structure
CLASSES
    Point
    Square
    Quadtree
"""

from __future__ import annotations
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import List


class Point:
    """
    A class to hold all the information needed for a point in the quadtree

    Attributes
    ----------
        x, y: float
            the x and y position coordinates of the point
        q: float
            the charge of the point
        m: float
            the mass of the point

    """
    def __init__(self, p: NDArray, q: float, m: float) -> None:
        self.x, self.y = p
        self.q = q
        self.m = m

    def __repr__(self) -> str:
        return f"Point(x,y=({self.x}, {self.y}), q={self.q}, m={self.m})"


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
    def __init__(self, x, y, d) -> None:
        self.x, self.y = x, y
        self.d = d
        self.r = d/2
        self.centre_of_charge = None

    def __repr__(self) -> str:
        return f"Square(x,y=({self.x}, {self.y}), d={self.d})"

    def contains(self, point: Point) -> bool:
        """
        Calculates whether a point lies within the square

        Arguments
        ---------
            point: Point
                the point to test

        Returns
        -------
            does_contain: bool
                whether the point lies within the square

        """
        does_contain = (self.x - self.r <= point.x < self.x + self.r) and \
                       (self.y - self.r <= point.y < self.y + self.r)
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
        if quadrant == 0:
            return Square(self.x - self.r/2, self.y + self.r/2, self.r)
        if quadrant == 1:
            return Square(self.x + self.r/2, self.y + self.r/2, self.r)
        if quadrant == 2:
            return Square(self.x - self.r/2, self.y - self.r/2, self.r)
        if quadrant == 3:
            return Square(self.x + self.r/2, self.y - self.r/2, self.r)


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
        self.points = []
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
        self.points = []
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

        for p in self.points:
            for quad in self.quadrants:
                if quad.insert(p):
                    break
        self.points = None

    def insert(self, p: Point) -> bool:
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
        if not self.boundary.contains(p):
            return False

        if not self.divided:
            if len(self.points) == 0 or self.max_d <= 0:
                self.points.append(p)
                return True
            self.subdivide()

        inserted = False
        for quad in self.quadrants:
            if quad.insert(p):
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
            patch = patches.Rectangle((self.boundary.x - self.boundary.r,
                                       self.boundary.y - self.boundary.r),
                                      self.boundary.d, self.boundary.d,
                                      edgecolor="black", facecolor="none", linewidth=0.5)
            ax.add_patch(patch)



