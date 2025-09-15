"""
This module contains tests for the fast_rsm.scan.Scan class.
"""

# pylint: disable=unreachable

from time import time

import numpy as np
from numpy.testing import assert_allclose

from diffraction_utils import Frame
from fast_rsm.scan import Scan


def test_binned_rsm_i07_421595(path_to_resources: str):
    """
    Make sure that we can run an rsm, and that it exactly matches that of a
    dataset that is known to have correct data.

    TODO: set memory limitations on this test. Set perhaps more rigorous time
        limitations would be nice, too.
    """
    # Prepare a Scan object using the from_i07 classmethod.
    path_to_data = \
        "/Users/richard/Data/i07/rsm_soller_test/421595/i07-421595.nxs"
    beam_centre = (739, 1329)
    detector_distance = 502.6e-3
    setup = 'horizontal'
    sample_oop = [0, 1, 0]
    scan = Scan.from_i07(path_to_data, beam_centre, detector_distance, setup,
                         sample_oop, path_to_data)

    lab_frame = Frame(Frame.lab)
    start = np.array([0, -0.8, -0.9])
    stop = np.array([1.5, 0.1, 0])
    step = np.array([0.005, 0.005, 0.005])

    # Carry out and (roughly) time the reciprocal space map.
    time1 = time()
    rsmap = scan.binned_reciprocal_space_map(lab_frame, start, stop, step, 9)
    time2 = time()
    time_taken = time2 - time1

    # Assert that this took under 20 ms per image collected (5x faster than
    # collection is an absolute minimum speed requirement).
    num_images = scan.metadata.data_file.scan_length
    assert time_taken / num_images < 50e-3

    print(f"Time taken for the map: {time_taken}s")

    # Load the reciprocal space map that is known to be correct.
    data = np.load(path_to_resources + "i07_421595_coarse_map.npy")

    # Make sure that we can re-create a map that is known to be correct.
    assert_allclose(data, rsmap)


def test_binned_rsm_i10_693862(i10_scan: Scan, path_to_resources: str):
    """
    Make sure that we can deal with linearly binned data.

    TODO: set memory/time limitations on this test. Test output vs a stored .vtk
    file. Make this optional and make slow tests run via a flag.
    """
    # The necessary setup.
    frame = Frame(Frame.hkl, i10_scan.metadata.diffractometer)
    start = np.array([-0.0015, 0.11, -0.005])
    stop = np.array([0.002, 0.115, 0.003])
    step = np.array([0.0002, 0.0002, 0.0002])

    # Do the reciprocal space map. Time it.
    time1 = time()
    rsm = i10_scan.binned_reciprocal_space_map(frame, start, stop, step, 10)
    time2 = time()
    time_taken = time2 - time1

    # Make sure that it was done in < 60 ms per 4M image.
    num_images = i10_scan.metadata.data_file.scan_length
    time_per_image = time_taken / num_images
    print(time_per_image)
    assert time_per_image < 60e-3

    # Make sure that our RSM checks out against the one known to be good.
    data = np.load(path_to_resources + "i10_693862_coarse_map.npy")
    assert_allclose(rsm, data)
