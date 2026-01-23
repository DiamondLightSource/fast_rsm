from fast_rsm.rsm_metadata import RSMMetadata
from types import SimpleNamespace
from pytest import fixture
import pytest
import numpy as np
from numpy.testing import assert_allclose
@fixture
def testRSM():
    test_diff=SimpleNamespace(\
        data_file=SimpleNamespace(image_shape=(100,200),\
                                  is_rotated=True))
    beam_centre=(20,40)
    return RSMMetadata(test_diff,beam_centre) 

def test_beamcentre_swap(testRSM):
    testRSM.swap_beam_centre()
    assert testRSM.beam_centre==(40,20)
    testRSM.swap_beam_centre()
    assert testRSM.beam_centre==(20,40)

def test_beamcentre_rot(testRSM):
    testRSM.check_beam_centre_rot()
    assert testRSM.beam_centre==(60,20)

def test_beamcentre_range(testRSM):
    assert testRSM.beam_centre_range_check()
    testRSM.beam_centre=(1000,20)
    with pytest.raises(IndexError):
        testRSM.beam_centre_range_check()

def test_vertpixoffsets(testRSM):
    testRSM.data_file.image_shape=(5,5)
    testRSM.beam_centre=2,4
    outpixels=np.array([2,1, 0, -1, -2]).reshape(5,1).repeat(5,axis=1)
       #np.arange(5).reshape(5,1).repeat(4, axis=1)
    #testRSM._init_vertical_pixel_offsets()
    assert_allclose(testRSM.vertical_pixel_offsets,outpixels)
    testRSM.data_file.pixel_size=0.1
    outdists=np.array([0.2,0.1, 0.0, -0.1, -0.2]).reshape(5,1).repeat(5,axis=1)
    assert_allclose(testRSM.get_vertical_pixel_distances(0),outdists)

def test_getdetdist(testRSM):
    testRSM.data_file.detector_distance=0.89
    assert testRSM.get_detector_distance(0)==0.89
    testRSM.data_file.using_dps=False
    assert testRSM.get_detector_distance(0)==0.89
    testRSM.data_file.dpsz=np.array([0.005,0.005])
    testRSM.data_file.dpsz2=np.array([0.005,0.005])
    testRSM.data_file.using_dps=True
    assert testRSM.get_detector_distance(1)==0.90


def test_horpixoffsets(testRSM):
    testRSM.data_file.image_shape=(5,5)
    testRSM.beam_centre=2,3
    outpixels=np.tile([3,2,1, 0, -1], (5, 1))
       #np.arange(5).reshape(5,1).repeat(4, axis=1)
    #testRSM._init_horizontal_pixel_offsets()
    assert_allclose(testRSM.horizontal_pixel_offsets,outpixels)
    testRSM.data_file.pixel_size=0.1
    outdists=np.tile([0.3,0.2,0.1, 0.0, -0.1], (5, 1))
    assert_allclose(testRSM.get_horizontal_pixel_distances(0),outdists)

