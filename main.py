import numpy as np
from particle import Particle
import matplotlib.pyplot as plt

from algorithms import BarnesHut, FMM, PairWise
from forces import InverseSquare, Inverse
from universe import Universe
from index import Index

def main():
    N = 1000
    G = 0.01
    SOFTENING = 0.01
    DT = 0.01
    EPSILON = 1e-3

    MAX_LEVEL = int(np.log(N)/np.log(4))
    PRECISION = int(-np.log2(EPSILON))

    force = Inverse(G)

    PW_algorithm = PairWise(force)
    FMM_algorithm = FMM(MAX_LEVEL, PRECISION, G)
    particles = [Particle() for _ in range(N)]



    # PW_acc = PW_algorithm.calculate_accelerations(particles)
    # FMM_acc = FMM_algorithm.calculate_accelerations(particles)

    universe = Universe(particles, FMM_algorithm, DT)
    universe.animation()
    return


if __name__ == "__main__":
    np.random.seed(0)
    main()
