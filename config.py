import numpy as np
import os


def save_config(name, N, s, v, overwrite=False):
    dir = f"configs/{name}"
    path = dir + "/config.npz"

    if os.path.isdir(dir):
        if not overwrite:
            raise ValueError("Configuration already exists")
    else:
        os.mkdir(dir)

    np.savez(path, N=N, s=s, v=v)


def load_config(name):
    dir = f"configs/{name}"
    path = dir + "/config.npz"
    if not os.path.isfile(path):
        raise ValueError("The configuration does not exist")
    data = np.load(path)
    return data
