import numpy as np
import os


def save_config(name, N, s, v, overwrite=False):
    direc = f"configs/{name}"
    path = direc + "/config.npz"

    if os.path.isdir(direc):
        if not overwrite:
            raise ValueError("Configuration already exists")
    else:
        os.mkdir(direc)

    np.savez(path, N=N, s=s, v=v)


def load_config(name):
    direc = f"configs/{name}"
    path = direc + "/config.npz"
    if not os.path.isfile(path):
        raise ValueError("The configuration does not exist")
    data = np.load(path)
    return data
