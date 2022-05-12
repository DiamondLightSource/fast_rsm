"""
This module provides tests for the functions in the RSMapper.binning module.
"""

import numpy as np

from RSMapper.binning import _correct_stop, finite_diff_shape, linear_bin


def test_correct_stop():
    """
    Make sure that correct_stop is working properly.
    """
    start = np.array([1.0, 1.0, 2.0])
    stop = np.array([2.56, 3, 3])
    step = np.array([0.1, 0.1, 0.1])

    assert (_correct_stop(start, stop, step) == [2.6, 3, 3]).all()


def test_finite_diff_shape():
    """
    Make sure that we can work out what shape (in the ndarray.shape sense) our
    binned data will need to have.
    """
    start = np.array([1.0, 1.0, 2.0])
    stop = np.array([2.6, 3, 3])
    step = np.array([0.1, 0.1, 0.1])

    assert (finite_diff_shape(start, stop, step) == [17, 21, 11]).all()


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
