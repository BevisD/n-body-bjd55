"""
NAME
    index
DESCRIPTION
    contains the index class
CLASSES
    Index
"""

from __future__ import annotations

__all__ = ["Index"]


class Index:
    """
    A class that represents the index of a particle in a quadtree.
    Each particle is placed into a box with a 2D index

    Attributes
    ----------
    index: tuple(int, int)
        a tuple of length 2 containing the i and j index of the particle
    level: int
        the depth of the index within the quadtree, minimum 0

    Methods
    -------
    children()
        creates the four indices corresponding to the children of the cell
    parent()
        creates the index corresponding to the parent of the cell
    neighbours()
        creates the indices of the cell's direct neighbours (including diagonals)
    well_seperated()
        creates the indices that are well seperated from the cell
        well seperated cells are the children of the cell's parent that are not
         the cell's direct neighbours

    Raises
    ------
        ValueError: for an index with level < 0
        ValueError: for an index out of bounds for a given level

    """

    def __init__(self, i: int, j: int, level: int) -> None:
        if level < 0:
            raise ValueError(f"Index cannot have level {level} < 0")
        if (i < 0 or i >= 2 ** level or
                j < 0 or j >= 2 ** level):
            raise ValueError(
                f"Index ({i}, {j}) is out of bounds for level {level}")

        self.index = (i, j)
        self.level = level

    def children(self) -> list[Index]:
        """
        returns the indices of the cells that are children to the current index

        Returns
        -------
            children: list[Index]
                the list of cells that are directly below the current cell
        """
        i, j = self.index
        children = [Index(2 * i, 2 * j, self.level + 1),
                    Index(2 * i + 1, 2 * j, self.level + 1),
                    Index(2 * i, 2 * j + 1, self.level + 1),
                    Index(2 * i + 1, 2 * j + 1, self.level + 1)]
        return children

    def parent(self) -> Index:
        """
        creates the index of the cell that is the parent of the current cell
        
        Returns
        -------
            parent: Index
                the Index of the cell directly above the current cell

        Raises
        ------
            ValueError
                if trying to get the parent of the lowest level cell
        """
        if self.level == 0:
            raise ValueError("Index of level 0 has no parent")

        i, j = self.index
        parent = Index(i // 2, j // 2, self.level - 1)
        return parent

    def neighbours(self) -> set[Index]:
        """
        Returns the indices of the cells that are direct neighbours to the
        current cell (including diagonals)
        
        Returns
        -------
            neighbours: set[Index]
                the list of cells that are directly adjacent to the current
                cell
        """
        if self.level == 0:
            return set()

        offsets = [(-1, -1),
                   (-1, +0),
                   (-1, +1),
                   (+0, -1),
                   (+0, +1),
                   (+1, -1),
                   (+1, +0),
                   (+1, +1), ]
        neighbours = set()

        for offset in offsets:
            try:
                neighbours.add(self + offset)

            except ValueError:
                continue
        return neighbours

    def well_seperated(self) -> set[Index]:
        """
        Returns the indices of the cells that are well seperated from the
        current cell. Well seperated cells are the children of the parent of
        the cell that are not direct neighbours
        
        Returns
        -------
            well_seperated: set[Index]
                the list of cells that are well-seperated from the current cell
        """
        if self.level <= 1:
            return set()
        parent_neighbours = self.parent().neighbours()
        well_seperated = set()
        for parent_neighbour in parent_neighbours:
            well_seperated.update(parent_neighbour.children())

        well_seperated -= self.neighbours()
        return well_seperated

    def __add__(self, other):
        """
        Raises
        ------
            ValueError
                if trying to add an object that isn't an Index or tuple
            ValueError
                if trying to add indices of different levels
            ValueError
                if adding the index creates an index out of range
        """
        new_i, new_j = 0, 0

        if isinstance(other, Index):
            if other.level != self.level:
                raise ValueError(
                    f"Cannot add indices of levels {self.level} and "
                    f"{other.level}")

            new_i = self.index[0] + other.index[0]
            new_j = self.index[1] + other.index[1]

        elif isinstance(other, tuple):
            new_i = self.index[0] + other[0]
            new_j = self.index[1] + other[1]
        else:
            raise ValueError(f"Cannot add types Index and {type(other)}")

        if (new_i < 0 or new_i >= 2 ** self.level or
                new_j < 0 or new_j >= 2 ** self.level):
            raise ValueError(
                f"Index ({new_i}, {new_j}) is out of bounds for level "
                f"{self.level}")

        return Index(new_i, new_j, self.level)

    def __str__(self):
        return f"Index(({self.index[0]}, {self.index[1]}), level={self.level})"

    def __key(self) -> tuple[int, int, int]:
        """
        Generates a hashable key for the class

        Returns
        -------
            tuple[int, int, int]
                the hashable key for the Index
        """
        return self.index[0], self.index[1], self.level

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other: any):
        """
        checks if an object is equal to itself

        Returns
        -------
            bool
        """
        if isinstance(other, Index):
            return self.index == other.index and self.level == other.level
        elif isinstance(other, tuple):
            if len(other) == 2:
                return self.index == other
            elif len(other) == 3:
                return self.index + (self.level,) == other
        else:
            return False
