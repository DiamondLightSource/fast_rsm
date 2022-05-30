"""
This module contains tests for the RSMapper.scan.Scan class.
"""

# pylint: disable=unreachable

import os
from time import time

import numpy as np

from diffraction_utils import Frame
from RSMapper.scan import Scan


def test_binned_rsm(i10_scan: Scan):
    """
    Make sure that we can run an rsm in reasonable time.

    TODO: set memory/time limitations on this test. Test output vs a stored .vtk
    file. Make this optional and make slow tests run via a flag.
    """
    return
    frame = Frame(Frame.hkl, i10_scan.metadata.diffractometer)

    start = np.array([-10, -10, -10])
    stop = np.array([10, 10, 10])
    step = np.array([0.1, 0.1, 0.1])
    time1 = time()
    rsm = i10_scan.binned_reciprocal_space_map(frame, start, stop, step,
                                               os.cpu_count())

    time2 = time()

    print(f"Time taken for the map: {time2 - time1}s")
    assert np.max(rsm) != 0  # Make sure something has happened!
