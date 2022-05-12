"""
This module contains functions for binning 3D scalar fields.
"""

import numpy as np


def _correct_stop(start: np.ndarray, stop: np.ndarray, step: np.ndarray):
    """
    People might input stop values that aren't an integer number of steps away
    from start. Fix that, moving stop to the nearest possible integer number of
    steps away from start.
    """
    num_steps = np.rint((stop-start)/step)
    return start + num_steps*step


def _fix_delta_q_geometry(arr: np.ndarray) -> np.ndarray:
    """
    If arr.shape is 3D, make it 2D.
    """
    if len(arr.shape) == 3:
        return arr.reshape((arr.shape[0]*arr.shape[1], arr.shape[2]))
    return arr


def _fix_intensity_geometry(arr: np.ndarray) -> np.ndarray:
    """
    If arr.shape is 2D, make it 1D.
    """
    if len(arr.shape) == 2:
        return arr.flatten()
    return arr


def finite_diff_shape(start: np.ndarray, stop: np.ndarray, step: np.ndarray):
    """
    Works out the shape of the finite differences grid that we're going to need
    to store data with this start, stop and step.
    """
    return ((stop-start)/step + 1).astype(np.int32)


def linear_bin(coords: np.ndarray,  # Coordinates of each intensity.
               intensities: np.ndarray,  # The corresponding intensities.
               start: np.ndarray,  # (start_x, start_y, start_z)
               stop: np.ndarray,  # (stop_x, stop_y, stop_z)
               step: np.ndarray  # (delta_x, delta_y, delta_z)
               ) -> np.ndarray:
    """
    Bin intensities with coordinates coords into linearly spaced finite
    differences bins.
    """
    # Fix the geometry of the input arguments.
    coords = _fix_delta_q_geometry(coords)
    intensities = _fix_intensity_geometry(intensities)
    # Fix the stop value; work out dimensions of finite elements volume.
    stop = _correct_stop(start, stop, step)
    shape = finite_diff_shape(start, stop, step)
    size = shape[0]*shape[1]*shape[2]

    # Subtract start values.
    coords -= start
    # Divide by step.
    coords /= step
    # Round to integers. Note that the type is still float64.
    np.rint(coords, out=coords)

    # Now convert to tuple of integer arrays for array indexing to work.
    coords = (coords[:, 0].astype(np.int32),
              coords[:, 1].astype(np.int32),
              coords[:, 2].astype(np.int32))
    # Flatten the coordinates; we need this for np.bincount to work.
    flat_indices = np.ravel_multi_index(coords, shape)
    # Now we can use bincount to work out the intensities.
    bincount = np.bincount(flat_indices, weights=intensities,
                           minlength=size)

    bincount = bincount.reshape(shape)

    return bincount
