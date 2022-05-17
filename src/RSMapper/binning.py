"""
This module contains functions for binning 3D scalar fields.
"""

import numpy as np


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
        return arr.reshape(-1)
    return arr


def finite_diff_shape(start: np.ndarray, stop: np.ndarray, step: np.ndarray):
    """
    Works out the shape of the finite differences grid that we're going to need
    to store data with this start, stop and step.
    """
    return (
        len(np.arange(start[0], stop[0], step[0])),
        len(np.arange(start[1], stop[1], step[1])),
        len(np.arange(start[2], stop[2], step[2]))
    )


def linear_bin(coords: np.ndarray,  # Coordinates of each intensity.
               intensities: np.ndarray,  # The corresponding intensities.
               start: np.ndarray,  # (start_x, start_y, start_z)
               stop: np.ndarray,  # (stop_x, stop_y, stop_z)
               step: np.ndarray  # (delta_x, delta_y, delta_z)
               ) -> np.ndarray:
    """
    Bin intensities with coordinates coords into linearly spaced finite
    differences bins. This is bottlenecking and should be rewritten as a numpy
    C extension to achieve perfect performance, but it's better than numpy's
    histdd function by a *LOT*!
    """
    # Fix the geometry of the input arguments.
    coords = _fix_delta_q_geometry(coords)
    intensities = _fix_intensity_geometry(intensities)

    # Work out dimensions and shape of finite elements volume.
    shape = finite_diff_shape(start, stop, step)
    size = shape[0]*shape[1]*shape[2]

    # Subtract start values.
    coords -= start
    # Divide by step.
    coords /= step
    # Round to integers. Note that the type is still float64.
    np.rint(coords, out=coords)

    # The following routine is bad in memory but makes numpy happy.
    # Kill any coords that are out of bounds.
    # print(np.max(coords[:, 0]), np.min(coords[:, 0]))
    # print(np.max(coords[:, 1]), np.min(coords[:, 1]))
    # print(np.max(coords[:, 2]), np.min(coords[:, 2]))
    coord_in_bounds = (coords >= 0) * (coords < shape)
    intensities *= coord_in_bounds[:, 0]
    intensities *= coord_in_bounds[:, 1]
    intensities *= coord_in_bounds[:, 2]
    # And finally bin those empty intensities to the origin.
    coords *= coord_in_bounds
    # print(np.max(coords[:, 0]), np.min(coords[:, 0]))
    # print(np.max(coords[:, 1]), np.min(coords[:, 1]))
    # print(np.max(coords[:, 2]), np.min(coords[:, 2]))

    # Now convert to tuple of integer arrays for array indexing to work.
    coords = (coords[:, 0].astype(np.int32),
              coords[:, 1].astype(np.int32),
              coords[:, 2].astype(np.int32))
    # Flatten the coordinates; we need this for np.bincount to work.
    flat_indices = np.ravel_multi_index(coords, shape)
    # Now we can use bincount to work out the intensities.
    bincount = np.bincount(flat_indices, weights=intensities,
                           minlength=size)
    # print(f"Bincount shape: {bincount.shape}")
    bincount = bincount.reshape(shape)

    return bincount


def hist_shape(start, stop, step):
    """
    Returns the shape of the histogram returned by linear_bin_histdd.
    """
    bins_0 = np.arange(start[0], stop[0]+step[0], step[0])
    bins_1 = np.arange(start[1], stop[1]+step[1], step[1])
    bins_2 = np.arange(start[2], stop[2]+step[2], step[2])
    return (len(bins_0)-1, len(bins_1)-1, len(bins_2)-1)


def linear_bin_histdd(coords: np.ndarray,  # Coordinates of each intensity.
                      intensities: np.ndarray,  # Corresponding intensities.
                      start: np.ndarray,  # (start_x, start_y, start_z)
                      stop: np.ndarray,  # (stop_x, stop_y, stop_z)
                      step: np.ndarray  # (delta_x, delta_y, delta_z)
                      ) -> np.ndarray:
    """
    Uses numpy's histogramdd to do some binning. This is, unfortunately, a very
    slow routine.
    """
    # Fix the geometry of the input arguments.
    coords = _fix_delta_q_geometry(coords)
    intensities = _fix_intensity_geometry(intensities)

    data, _ = np.histogramdd(coords,
                             bins=hist_shape(start, stop, step),
                             range=((start[0], stop[0]),
                                    (start[1], stop[1]),
                                    (start[2], stop[2])),
                             weights=intensities)
    return data
