"""
NAME
    config
DESCRIPTION
    loads and saves the initial configurations of universes
FUNCTIONS
    save_config
    load_config
"""

import numpy as np
import os
from numpy.typing import NDArray
from typing import Dict
from typing import Any


def save_config(name: str, N: int, s: NDArray, v: NDArray,
                overwrite: bool = False) -> None:
    """
    Saves the initial positions and velocities of a universe to a .npz file

    Arguments
    ---------
        name: str
            the file name to save to (without file extension)
        N: int
            the number of particles in the universe
        s: NDArray
            the positions of the particles
        v: NDArray
            the velocities of the particles
        overwrite: bool
            the option to overwrite a file if it already exists

    Returns
    -------
        None
    """
    direc = f"configs/{name}"
    path = direc + "/config.npz"

    if os.path.isdir(direc):
        if not overwrite:
            raise ValueError("Configuration already exists")
    else:
        os.mkdir(direc)

    np.savez(path, N=N, s=s, v=v)


def load_config(name: str) -> Dict[str, Any]:
    """
    Loads the initial conditions for a universe

    Arguments
    ---------
        name: str
            the filename of the configuration to load (without file extension)

    Returns
    -------
        data: Dict[str, Any]
            the configuration data
    """
    direc = f"configs/{name}"
    path = direc + "/config.npz"
    if not os.path.isfile(path):
        raise ValueError("The configuration does not exist")
    data = np.load(path)
    return data
