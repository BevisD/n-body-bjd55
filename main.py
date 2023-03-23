import numpy as np
from universe import Universe
import matplotlib.pyplot as plt
import universe


def main():
    N_BODY = 100
    G = 0.01
    s = np.random.uniform(-1, 1, (N_BODY, 2))
    v = np.random.uniform(-1, 1, (N_BODY, 2))
    uni = Universe(N_BODY, G, s, v)
    uni.render()
    return


if __name__ == "__main__":
    main()
