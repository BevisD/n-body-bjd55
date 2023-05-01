import numpy as np

from particle import Particle
from algorithms import BarnesHut, FMM
from forces import InverseSquare
from universe import Universe


def main():
    N = 100
    G = -1
    DT = 0.01
    THETA = 0.5
    SOFTENING = 0.01
    MAX_DEPTH = 3
    PRECISION = 5

    force = InverseSquare(G, SOFTENING)

    BH_algorithm = BarnesHut(force, theta=THETA)
    FMM_algorithm = FMM(MAX_DEPTH, PRECISION, G)

    particles = [
        Particle(charge=1 / np.sqrt(N)) for _ in
        range(N)]

    universe = Universe(particles, FMM_algorithm, DT)
    universe.animation()

    return


if __name__ == "__main__":
    np.random.seed(0)
    main()
