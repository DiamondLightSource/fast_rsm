"""
This module contains tests for the RSMapper.scan.Scan class.
"""

# pylint: disable=unreachable

from time import time

import numpy as np
from numpy.testing import assert_allclose

from diffraction_utils import Frame
from RSMapper.scan import Scan


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
    assert time_taken/num_images < 50e-3

    print(f"Time taken for the map: {time_taken}s")

    # Load the reciprocal space map that is known to be correct.
    data = np.load(path_to_resources + "i07_421595_coarse_map.npy")

    # Make sure that we can re-create a map that is known to be correct.
    assert_allclose(data, rsmap)
