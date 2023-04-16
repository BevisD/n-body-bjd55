import numpy as np
from universe import Universe
from barneshut import BarnesHutUniverse
from config import load_config
from acceleration import pairwise

def main():
    data = load_config("normal_static_circle")
    N = data["N"]
    s = data["s"]
    v = data["v"]
    # N = 10
    # s = np.random.uniform(-1, 1, (10, 2))
    # v = np.random.uniform(-0, 0, (10, 2))

    G = 0.01
    SOFTENING = 0.01
    DT = 0.01

    uni = Universe(N, G, SOFTENING, DT, s.copy(), v.copy(),
                   world_size=2, point_size=1)

    # uni.render(show_squares=True)
    uni.animation()
    return


if __name__ == "__main__":
    np.random.seed(0)
    main()
