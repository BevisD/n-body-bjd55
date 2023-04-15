

from __future__ import annotations
import numpy as np
from typing import List


class Index:
    def __init__(self, i: int, j: int, level: int) -> None:
        if level < 0:
            raise ValueError(f"Index cannot have level {level} < 0")
        if (i < 0 or i >= 2 ** level or
                i < 0 or i >= 2 ** level):
            raise ValueError(f"Index ({i}, {j}) is out of bounds for level {level}")

        self.index = (i, j)
        self.level = level

    def children(self) -> List[Index]:
        i, j = self.index
        return [Index(2 * i, 2 * j, self.level + 1),
                Index(2 * i + 1, 2 * j, self.level + 1),
                Index(2 * i, 2 * j + 1, self.level + 1),
                Index(2 * i + 1, 2 * j + 1, self.level + 1)]

    def parent(self):
        if self.level == 0:
            raise ValueError("Index of level 0 has no parent")

        i, j = self.index
        return Index(i // 2, j // 2, self.level - 1)

    def neighbours(self):
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

    def well_seperated(self):
        if self.level <= 1:
            return set()
        parent_neighbours = self.parent().neighbours()
        well_seperated = set()
        for parent_neighbour in parent_neighbours:
            well_seperated.update(parent_neighbour.children())

        well_seperated -= self.neighbours()
        return well_seperated

    def __add__(self, other):
        new_i, new_j = 0, 0

        if isinstance(other, Index):
            if other.level != self.level:
                raise ValueError(f"Cannot add indices of levels {self.level} and {other.level}")

            new_i = self.index[0] + other.index[0]
            new_j = self.index[1] + other.index[1]

        elif isinstance(other, tuple):
            new_i = self.index[0] + other[0]
            new_j = self.index[1] + other[1]
        else:
            raise ValueError(f"Cannot add types Index and {type(other)}")

        if (new_i < 0 or new_i >= 2 ** self.level or
                new_j < 0 or new_j >= 2 ** self.level):
            raise ValueError(f"Index ({new_i}, {new_j}) is out of bounds for level {self.level}")

        return Index(new_i, new_j, self.level)

    def __str__(self):
        return f"Index(({self.index[0]}, {self.index[1]}), level={self.level})"

    def __key(self):
        return self.index[0], self.index[1], self.level

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Index):
            return self.index == other.index and self.level == other.level
        elif isinstance(other, tuple):
            if len(other) == 2:
                return self.index == other
            elif len(other) == 3:
                return self.index + (self.level,) == other
        else:
            return False
