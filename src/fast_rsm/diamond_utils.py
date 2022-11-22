"""
This file contains a suite of utility functions for processing data acquired
specifically at Diamond.
"""

from typing import Tuple

import numpy as np
from scipy.constants import physical_constants


def q_to_theta(q_values: np.ndarray, energy: float) -> np.ndarray:
    """
    Takes a set of q_values IN INVERSE ANGSTROMS and and energy IN ELECTRON
    VOLTS and returns the corresponding theta values (TIMES BY 2 TO GET 2THETA).

    Args:
        q_values:
            The q values in units of inverse angstroms (multiplied by 2pi, for
            those that know the different conventions).
        energy:
            The energy of the incident beam in units of electron volts. Not kev.
            Not joules (lol). Electron volts, please.

    Returns:
        A numpy array of the corresponding theta values (NOT TWO THETA). Just
        multiply by 2 to get two theta.
    """
    # First calculate the wavevector of the incident light.
    planck = physical_constants["Planck constant in eV s"][0]
    speed_of_light = physical_constants["speed of light in vacuum"][0] * 1e10
    # My q_values are angular, so my wavevector needs to be angular too.
    ang_wavevector = 2*np.pi*energy/(planck*speed_of_light)

    # Do some basic geometry.
    theta_values = np.arccos(1 - np.square(q_values)/(2*ang_wavevector**2))

    # Convert from radians to degrees.
    return theta_values*180/np.pi


def get_volume_and_bounds(path_to_npy: str) -> Tuple[np.ndarray]:
    """
    Takes the path to a .npy file. Loads the volume stored in the .npy file, and
    also grabs the definition of the corresponding finite differences volume
    from the bounds file.

    Args:
        path_to_npy:
            Path to the .npy file containing the volume of interest.

    Returns:
        A tuple taking the form (volume, start, stop, step), where each element
        of the tuple is a numpy array of values. The first value returned is the
        full reciprocal space volume, and could be very large (up to ~GB).
    """
    # In case we were given a pathlib.path.
    path_to_npy = str(path_to_npy)
    # Work out the path to the bounds file.
    path_to_bounds = path_to_npy[:-4] + "_bounds.txt"

    # Load the data.
    volume = np.load(path_to_npy)
    q_bounds = np.loadtxt(path_to_bounds)

    # Scrape the start/stop/step values.
    start, stop, step = q_bounds[:, 0], q_bounds[:, 1], q_bounds[:, 2]

    # And return what we were asked for!
    return volume, start, stop, step
