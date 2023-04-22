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
    G = -1
    DT = 0.001
    THETA = 0.5
    DEPTH = 3
    P = 4
    force = Inverse(G)

    PW_algorithm = PairWise(force)
    BH_algorithm = BarnesHut(force, theta=THETA)
    FMM_algorithm = FMM(DEPTH, P, G)

    particles = [
        Particle(charge=1 / np.sqrt(N),
                 velocity=complex(np.random.uniform(-0.1, 0.1),
                                  np.random.uniform(-0.1, 0.1))) for _ in
        range(N)]

    universe = Universe(particles, BH_algorithm, DT, periodic_boundary=True)
    universe.animation(frames=200,
                       verbose=10,
                       iters_per_frame=10,
                       text=rf"${N=}$, ${G=}$, ${THETA=}$, EULER, F=INVERSE, ALG=BARNES-HUT",
                       filename="BH_clusters.gif",
                       show_squares=True,
                       barnes_hut_point=Particle(centre=complex(0.5, 0.5))
                       )

    return


if __name__ == "__main__":
    t1 = perf_counter()
    np.random.seed(0)
    main()
    t2 = perf_counter()
    print(f"Executed in {t2 - t1:.2E}s")
