import numpy as np
import matplotlib.pyplot as plt

from particle import Particle
from algorithms import BarnesHut, FMM, PairWise
from forces import Inverse
from universe import Universe
from integration import runge_kutta_4
from time import perf_counter


def main():
    N = 100
    G = 1
    DT = 0.01
    THETA = 0.5
    DEPTH = 3
    P = 5
    force = Inverse(G)

    PW_algorithm = PairWise(force)
    BH_algorithm = BarnesHut(force, theta=THETA)
    FMM_algorithm = FMM(DEPTH, P, G)

    particles = [
        Particle(charge=1 / np.sqrt(N)) for _ in
        range(N)]

    universe = Universe(particles, FMM_algorithm, DT)
    universe.animation()

    return


if __name__ == "__main__":
    t1 = perf_counter()
    np.random.seed(0)
    main()
    t2 = perf_counter()
    print(f"Executed in {t2 - t1:.2E}s")
