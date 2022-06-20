"""
This module provides tests for the functions in the fast_rsm.binning module.
"""

import numpy as np

from fast_rsm.binning import finite_diff_shape, linear_bin


def test_finite_diff_shape():
    """
    Make sure that we can work out what shape (in the ndarray.shape sense) our
    binned data will need to have.
    """
    start = np.array([1.0, 1.0, 2.0])
    stop = np.array([2.6, 3, 3])
    step = np.array([0.1, 0.1, 0.1])

    assert finite_diff_shape(start, stop, step) == (16, 20, 10)


def test_linear_bin():
    """
    Make sure that linear_bin is working.
    """
    points = np.array([np.array([2.14, 2.14, 2.14]),
                       np.array([2.18, 2.18, 2.18]),
                       np.array([2.24, 2.24, 2.24])])
    intensities = np.array([1, 2, 3])

    start = np.array([1.0, 1.0, 2.0])
    stop = np.array([2.6, 3, 3])
    step = np.array([0.1, 0.1, 0.1])

    linearly_binned = linear_bin(points, intensities, start, stop, step)
    # print(linearly_binned)
    assert linearly_binned[11, 11, 1] == 1
    assert linearly_binned[12, 12, 2] == 5
