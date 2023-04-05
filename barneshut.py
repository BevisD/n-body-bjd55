"""
NAME
    barneshut
DESCRIPTION
    This module contains the BarnesHutUniverse class
CLASSES
    BarnesHutUniverse
"""

from universe import Universe
from quadtree import QuadTree, Square, Point
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class BarnesHutUniverse(Universe):
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
        s: np.ndarray
            the positions of particles in the universe
        v: np.ndarray
            the velocities of particles in the universe
        a: np.ndarray
            the accelerations of particles in the universe
        m: np.ndarray
            the masses of particles in the universe
        q: np.ndarray
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
        boundary: Square
            the boundary that contains the particles
        max_depth: int
            the maximum recursion depth for the Barnes-Hut algorithm
        theta: float
            the critical ratio for the Barnes-Hut algorithm
        qt: QuadTree
            the quadtree for the Barnes-Hut algorithm


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
        acceleration()
            calculates the accelerations for the positions of the particles
            overwrites the inherited acceleration function
        barnes_hut()
            executes the Barnes-Hut algorithm
        calculate_center_of_charges()
            fills the quadtree with the centre of masses of each quadrant
        """
    def __init__(self, *args, theta=0.5, max_depth=6, **kwargs):
        super().__init__(*args, **kwargs)
        self.boundary = Square(0, 0, self.SIZE * 2)  # Create boundary for BH quadtree
        self.max_depth = max_depth  # Max recursion depth for BH
        self.theta = theta  # The critical ratio for BH
        self.qt = QuadTree(self.boundary, max_depth)  # Quadtree for BH
        self.calc_acceleration = self.acceleration  # Overwrite acceleration function
        self.update_qt()  # Initialise quadtree

    def acceleration(self, s):
        """
        Calculates the accelerations of the particles, given their positions

        Arguments
        ---------
            s: np.ndarray
                positions of the particles

        Returns
        -------
            a: np.ndarray
                accelerations of the particles
        """
        # Update centres of charges for quadtree
        self.calculate_centre_of_charge(self.qt)

        a = np.zeros((self.N, 2))  # Initialise acceleration array
        for i, (p, m, q) in enumerate(zip(s, self.m, self.q)):
            # Get array of positions and charges from BH algorithm
            centres, charges = self.barnes_hut(self.qt, p)
            if centres and charges:
                # Convert to numpy array
                centres = np.stack(centres, axis=0)
                charges = np.array(charges)
            else:
                continue

            r = centres - p  # Vectors between particles
            d = np.linalg.norm(r, axis=1)  # Distances between particles
            # Calculate direction between particles
            r_norm = np.divide(r, d[:, np.newaxis], where=(d[:, np.newaxis] != 0))
            # Update acceleration for current particles
            a[i] = np.sum(r_norm * (self.force(d, q, charges) / m)[:, np.newaxis], axis=0)
        return a

    def barnes_hut(self, qt, p):
        """
        Executes the Barnes-Hut algorithm

        Arguments
        ---------
            qt: Quadtree
                the current quadtree to perform the algorithm on
            p: np.ndarray
                the position of the particle for the algorithm

        Returns
        -------
            points: list
                the list of points that need to calculate acceleration for
            charges: list
                the charges of the points to calculate acceleration for
        """
        # Get centre of charge for current quadtree
        centre_of_charge = qt.boundary.centre_of_charge
        charge = centre_of_charge.q
        centre = np.array((centre_of_charge.x, centre_of_charge.y))

        # Initialise list of points and charges to return
        points = []
        charges = []

        if not qt.divided:
            if len(qt.points) == 0:
                return points, charges
            # Check we aren't returning the point itself
            if qt.boundary.contains(Point(p, 0, 0)):
                for point in qt.points:
                    centre = np.array((point.x, point.y))
                    charge = point.q
                    if not np.allclose(centre, p):
                        points.append(centre)
                        charges.append(charge)
            # If quadrant doesn't contain the point itself, return centre of charge
            else:
                charges.append(charge)
                points.append(centre)
            return points, charges
        else:
            d = qt.boundary.d  # Width of the quadrant
            r = np.linalg.norm(centre - p)  # Distance from the centre of charge to the point

            # If point lies on the centre of charge
            # or if BH condition not met, go to next recursion layer
            if r == 0 or d/r > self.theta:
                for quad in qt.quadrants:
                    # Get points and masses from BH algorithm of next layer
                    sub_centres, sub_charges = self.barnes_hut(quad, p)
                    points += sub_centres
                    charges += sub_charges
            # BH condition is met, return points and charges
            else:
                charges.append(charge)
                points.append(centre)
            return points, charges

    def calculate_centre_of_charge(self, qt):
        """
        Updates the quadtree with the centre of charges of each quadrant

        Arguments
        ---------
            qt: Quadtreee
                the current quadrant to calculate centre of mass for

        Returns
        -------
            None
        """
        if qt.divided:
            centre_of_charges = []
            # Get centres of charges for sub quadrants
            for quad in qt.quadrants:
                self.calculate_centre_of_charge(quad)
                centre_of_charges.append(quad.boundary.centre_of_charge)

            # Convert point data to coordinate and charge
            centres = [(p.x, p.y) for p in centre_of_charges]
            charges = [p.q for p in centre_of_charges]
        # If not divided get charges and points of all particles
        elif qt.points is not None and len(qt.points) != 0:
            centres = [(p.x, p.y) for p in qt.points]
            charges = [p.q for p in qt.points]
        # If quadrant is empty, return zero charge
        else:
            qt.boundary.centre_of_charge = Point((0, 0), 0, 0)
            return

        # Update centre of charge
        total_charge = np.sum(charges)
        centre_of_charge = np.matmul(charges, centres)/total_charge
        qt.boundary.centre_of_charge = Point(centre_of_charge, total_charge, 0)
        return

    def update_qt(self):
        """
        Inserts the universe particles into the quadtree, with relevant masses and charges

        Arguments
        ---------
            self

        Returns
        -------
            None
        """
        self.qt.clear()
        for p, q, m in zip(self.s, self.q, self.m):
            self.qt.insert(Point(p, q, m))

    def update(self):
        """
        Updates the positions of the particles

        Arguments
        ---------
            self

        Returns
        -------
            None
        """
        self.update_qt()
        super().update()

    def render(self, show_squares=False):
        """
        Displays the current positions of the particles in the universe

        Arguments
        ---------
            show_squares: boolean
                whether to display the boundaries of the quadrants

        Returns
        -------
            None
        """
        fig, ax = self.create_figure()
        x, y = self.s.T
        ax.scatter(x, y, s=self.MARKER_SIZE)
        if show_squares:
            self.qt.draw_quads(ax)
        plt.show()

    def draw_barnes_hut_quads(self, p, qt, ax):
        # Get centre of charge for current quadtree
        centre_of_charge = qt.boundary.centre_of_charge
        centre = np.array((centre_of_charge.x, centre_of_charge.y))

        d = qt.boundary.d
        r = np.linalg.norm(centre - p)

        if (r == 0 or d / r > self.theta) and qt.divided:
            for quad in qt.quadrants:
                self.draw_barnes_hut_quads(p, quad, ax)
        else:
            if not qt.divided and len(qt.points) == 0:
                return
            patch = patches.Rectangle((qt.boundary.x - qt.boundary.r,
                                       qt.boundary.y - qt.boundary.r),
                                      qt.boundary.d, qt.boundary.d,
                                      edgecolor="red", facecolor="none", linewidth=0.5)
            ax.add_patch(patch)

    def _animate(self, ax, scatter, iterations_per_frame,
                 show_squares=False, barnes_hut_squares=False):
        def animate(i):
            """
            Updates the position of the particles on the figure

            Arguments
            ---------
                i: int
                    the frame count

            Returns
            -------
                None
            """
            for _ in range(iterations_per_frame):
                self.update()

            if show_squares or barnes_hut_squares:
                [p.remove() for p in ax.patches]

            if show_squares:
                self.qt.draw_quads(ax)

            if barnes_hut_squares:
                self.draw_barnes_hut_quads(self.s[0], self.qt, ax)

            scatter.set_offsets(self.s)
        return animate
