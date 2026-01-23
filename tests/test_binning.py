"""
This module provides tests for the functions in the fast_rsm.binning module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from fast_rsm.binning import finite_diff_shape,fix_intensity_geometry,\
    fix_delta_q_geometry,finite_diff_grid,typecheckfloats

def test_fix_deltaqgeom():
    arr1=np.ones((2,3,4))
    outarr=fix_delta_q_geometry(arr1)
    assert np.shape(outarr)==(6,4)


def test_fix_intgeom():
    arr1=np.ones((2,5))
    outarr=fix_intensity_geometry(arr1)
    assert np.shape(outarr)==(10,)

def test_typecheckfloats():
    weights_int=np.array([1,2,3]).astype(np.uint32)
    weights_fl=np.array([1.0,2.0,3.0])
    testdict1={'weights1':weights_int}
    testdict2={'weights1':weights_fl,'weights2':weights_int}
    assert typecheckfloats(testdict1)
    with pytest.raises(ValueError):
        typecheckfloats(testdict2)

def test_finite_diff_grid():
    start = np.array([1.0, 1.0, 2.0])
    stop = np.array([2.6, 3, 3])
    step = np.array([0.1, 0.1, 0.1])
    xgrid=np.arange(1.0,2.6,0.1)
    ygrid=np.arange(1.0,3,0.1)
    zgrid=np.arange(2,3,0.1)
    outgrid=finite_diff_grid(start,stop,step)
    assert_allclose(outgrid[0],xgrid)
    assert_allclose(outgrid[1],ygrid)
    assert_allclose(outgrid[2],zgrid)

def test_finite_diff_shape():
    """
    Make sure that we can work out what shape (in the ndarray.shape sense) our
    binned data will need to have.
    """
    start = np.array([1.0, 1.0, 2.0])
    stop = np.array([2.6, 3, 3])
    step = np.array([0.1, 0.1, 0.1])

    assert finite_diff_shape(start, stop, step) == (16, 20, 10)


# def test_linear_bin():
#     """
#     Make sure that linear_bin is working.
#     """
#     points = np.array([np.array([2.14, 2.14, 2.14]),
#                        np.array([2.18, 2.18, 2.18]),
#                        np.array([2.24, 2.24, 2.24])])
#     intensities = np.array([1, 2, 3])

#     start = np.array([1.0, 1.0, 2.0])
#     stop = np.array([2.6, 3, 3])
#     step = np.array([0.1, 0.1, 0.1])

#     linearly_binned = linear_bin(points, intensities, start, stop, step)
#     # print(linearly_binned)
#     assert linearly_binned[11, 11, 1] == 1
#     assert linearly_binned[12, 12, 2] == 5
