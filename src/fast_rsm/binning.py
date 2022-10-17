"""
This module contains functions for binning 3D scalar fields.
"""

import time

import fast_histogram as fast
import numpy as np

import mapper_c_utils


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

    # Subtract start values. As usual, subtract by scalars for speed.
    coords[:, 0] -= start[0]
    coords[:, 1] -= start[1]
    coords[:, 2] -= start[2]
    # Divide by step, component by component so as to divide by scalars.
    coords[:, 0] /= step[0]
    coords[:, 1] /= step[1]
    coords[:, 2] /= step[2]
    # Round to integers. Note that the type is still float64.
    np.rint(coords, out=coords)

    # The following routine is bad in memory but makes numpy happy.
    # Kill any coords that are out of bounds.

    coord_in_bounds = coords >= 0
    coord_in_bounds[:, 0] *= coords[:, 0] < shape[0]
    coord_in_bounds[:, 1] *= coords[:, 1] < shape[1]
    coord_in_bounds[:, 2] *= coords[:, 2] < shape[2]
    intensities *= coord_in_bounds[:, 0]
    intensities *= coord_in_bounds[:, 1]
    intensities *= coord_in_bounds[:, 2]
    # And finally bin those empty intensities to the origin.
    coords *= coord_in_bounds

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


def fast_linear_bin(coords: np.ndarray,  # Coordinates of each intensity.
                    intensities: np.ndarray,  # Corresponding intensities.
                    start: np.ndarray,  # (start_x, start_y, start_z)
                    stop: np.ndarray,  # (stop_x, stop_y, stop_z)
                    step: np.ndarray  # (delta_x, delta_y, delta_z)
                    ) -> np.ndarray:
    """
    Binning using the fast-histogram library.
    """
    # Fix the geometry of the input arguments.
    coords = _fix_delta_q_geometry(coords)
    intensities = _fix_intensity_geometry(intensities)

    # Work out dimensions and shape of finite elements volume.
    shape = finite_diff_shape(start, stop, step)

    _range = ([start[0], stop[0]],
              [start[1], stop[1]],
              [start[2], stop[2]])

    # Run the histogram.
    # t1 = time()
    dd = fast.histogramdd(coords, shape, _range, intensities)
    # print(f"Time spend in histogramdd: {(time() - t1)*1000} ms")
    return dd


def weighted_bin_3d(coords: np.ndarray, weights: np.ndarray,
                    out: np.ndarray, count: np.ndarray,
                    start: np.ndarray, stop: np.ndarray, step: np.ndarray,
                    min_intensity=None
                    ) -> np.ndarray:
    """
    This is an alias to the native C function _weighted_bin_3d, which adds a
    useful protective layer. A lot of explicit type conversions are carried out,
    which prevents segfaults on the C side.

    Args:
        coords:
            A numpy array of the coordinates we're going to bin. Should be the
            output of Image.q_vectors.
        weights:
            The intensities measured at each of the q_vectors stored in coords.
            Should be the output of Image.data
        out:
            The current state of the binned reciprocal space map.
        count:
            The number of times each voxel in out has been binned to so far.
        start:
            A numpy array that looks like [qx_min, qy_min, qz_min].
        stop:
            A numpy array that looks like [qx_max, qy_max, qz_max]. Together
            with start, these arrays specify the bounds of the region of
            reciprocal space that we're binning to.
        step:
            The side-length of each reciprocal space voxel.
        min_intensity:
            Any intensities recorded below this intensity should be completely
            ignored. This is used for masking. Defaults to None (i.e. no
            masking). This is not the same thing as background subtraction.
    """
    # Uncomment these timestamps to benchmark the binning routine.
    # time_1 = time.time()
    coords = _fix_delta_q_geometry(coords)
    weights = _fix_intensity_geometry(weights)
    # Work out the shape array on the python end, as opposed to on the C end.
    # Life's easier in python, so do what we can here.
    shape = np.array(finite_diff_shape(start, stop, step)).astype(np.int32)

    # pylint: disable=c-extension-no-member
    start = start.astype(np.float32)
    step = step.astype(np.float32)

    # Make sure that min_intensity is a float32.
    if min_intensity is None:
        min_intensity = -np.inf
    min_intensity = np.array([min_intensity]).astype(np.float32)

    if weights.dtype != np.float32:
        raise ValueError("Weights must have dtype=np.float32")
    if coords.dtype != np.float32:
        raise ValueError("Coords must have dtype=np.float32")
    if out.dtype != np.float32:
        raise ValueError("out must have dtype=np.float32")
    if count.dtype != np.uint32:
        raise ValueError("Count must have dtype=np.int32")

    # Now we're ready to call the function.
    mapper_c_utils.weighted_bin_3d(coords, start, step, shape,
                                   weights, out, count, min_intensity)

    # time_taken = time.time() - time_1
    # print(f"Binning time: {time_taken}")
    return out


def weighted_bin_1d(coords: np.ndarray,
                    weights: np.ndarray,
                    out: np.ndarray,
                    count: np.ndarray,
                    start: float,
                    stop: float,
                    step: float):
    """
    Entirely analogous to weighted_bin_3d, but histograms in only 1 dimension.
    This is useful for re-histogramming for e.g. 2-theta and |Q| projections.

    Args:
        coords:
            A numpy array of the coordinates we're going to bin. Should be a
            numpy array with shape (N,).
        weights:
            The intensities measured at each of the q_vectors stored in coords.
            Should be a numpy array with shape (N,)
        out:
            If the results of this binning are to be appended to another
            histogram, out should be the histogram to which the results will be
            appended. Otherwise, np.zeros() please. This should have the shape
            of your desired output. So, if you're binning into 1000 equally
            spaced bines, it should have shape (1000,)
        count:
            Same shape as out. This array just counts how many times you bin
            into each bin. This is needed for normalisation.
        start:
            The minimum coordinate that we want to bin to. Any coordinate
            smaller than this value will be thrown away.
        stop:
            The maximum coordinate we want to bin to. Any coordinate larger
            than this value will be thrown away.
        step:
            The side-length of each bin voxel.
    """
    # Pylint will, as usual, complain because I guess it doesn't like my minimal
    # C extension.
    # pylint: disable=c-extension-no-member

    # Make sure start, stop and step are floats. Explicit type checking is
    # normally unpythonic, because you want an exception to be raised naturally.
    # Well, on the C end, we'd just get a segfault - this is the last line
    # of defence! For things to _feel_ pythonic, this is exactly the point where
    # we need explicit type checking!
    if not isinstance(start, float):
        raise ValueError("start Argument must have type float. "
                         f"Instead, it had type {type(start)}")
    if not isinstance(stop, float):
        raise ValueError("stop Argument must have type float. "
                         f"Instead, it had type {type(stop)}")
    if not isinstance(step, float):
        raise ValueError("step Argument must have type float. "
                         f"Instead, it had type {type(step)}")

    # Work out how many bins we're going to need.
    shape = np.arange(start, stop, step).shape[0]

    # Make sure that everything is a float32. We don't need double length
    # floating point precision, and this is, well, 2x faster!
    if weights.dtype != np.float32:
        raise ValueError("Weights must have dtype=np.float32")
    if coords.dtype != np.float32:
        raise ValueError("Coords must have dtype=np.float32")
    if out.dtype != np.float32:
        raise ValueError("out must have dtype=np.float32")
    if count.dtype != np.uint32:
        raise ValueError("Count must have dtype=np.int32")

    # Now we're ready to call the function.
    mapper_c_utils.weighted_bin_1d(
        coords, start, step, shape, weights, out, count)

    return out


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
