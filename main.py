import numpy as np
from particle import Particle

from algorithms import PairWise
from physics.forces import InverseSquare
from universe import Universe


def main():
    N = 100
    G = 0.01
    SOFTENING = 0.01
    DT = 0.001

    force = InverseSquare(G, SOFTENING)
    algorithm = PairWise(force)

    particles = [Particle(charge=1) for _ in range(N)]

    universe = Universe(particles, algorithm, DT)
    universe.animation()
    return


if __name__ == "__main__":
    np.random.seed(0)
    main()
