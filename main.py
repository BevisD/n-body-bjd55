import numpy as np
from universe import Universe

np.random.seed(0)


def main():
    N_BODY = 100
    G = 0.01
    SOFTENING = 0.01
    DT = 0.01

    s = np.random.uniform(-1, 1, (N_BODY, 2))
    v = np.random.uniform(-1, 1, (N_BODY, 2))
    uni = Universe(N_BODY, G, SOFTENING, DT, s, v,
                   world_size=2, point_size=4)

    uni.animation()
    return


if __name__ == "__main__":
    main()
