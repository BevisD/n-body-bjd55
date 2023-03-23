import numpy as np
from universe import Universe
from config import load_config
from integration import runge_kutta_4

np.random.seed(0)


def main():
    data = load_config("normal_static_circle")
    N = data["N"]
    s = data["s"]
    v = data["v"]

    G = 0.01
    SOFTENING = 0.01
    DT = 0.01

    uni = Universe(N, G, SOFTENING, DT, s, v,
                   world_size=2, point_size=4)
    uni.integrate = runge_kutta_4

    uni.animation()
    return


if __name__ == "__main__":
    main()
