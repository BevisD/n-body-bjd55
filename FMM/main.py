from particle import Particle
from index import Index
from fmm import FMM
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)


def main():
    N = 1000
    epsilon = 1e-3

    precision = int(-np.log2(epsilon))
    print(precision)
    max_level = int(np.log(N) / np.log(4))

    particles = [Particle(charge=1) for _ in range(N)]
    fmm = FMM(precision, max_level, particles)

    fmm.fmm_algorithm()
    fmm.calculate_potentials()

    return


if __name__ == "__main__":
    main()
