import numpy as np
from index import Index


class Particle:
    def __init__(self, charge: float = None, centre: complex = None) -> None:
        self.charge = charge if charge is not None else np.random.random() * 2 - 1
        if centre:
            self.centre = centre
        else:
            self.centre = np.random.random() + 1j * np.random.random()

        self.potential = 0

    def index(self, level):
        i = int(self.centre.imag * 2 ** level)
        j = int(self.centre.real * 2 ** level)

        return Index(i, j, level)

    def __repr__(self):
        return f"Particle({self.charge}, {self.centre}, {self.potential})"

    def __eq__(self, other):
        return (self.charge == other.charge and
                self.centre == other.centre)

    def __key(self):
        return self.charge, self.centre

    def __hash__(self):
        return hash(self.__key())

