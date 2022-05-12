"""
This module contains tests for the RSMapper.scan.Scan class.
"""

import numpy as np

from diffraction_utils import Frame
from RSMapper.scan import Scan


def test_binned_rsm(i10_scan: Scan):
    """
    Make sure that we can run an rsm in reasonable time.

    TODO: set memory/time limitations on this test.
    """
    frame = Frame(Frame.hkl, i10_scan.metadata.diffractometer)

    start = np.array([-10, -10, -10])
    stop = np.array([10, 10, 10])
    step = np.array([0.1, 0.1, 0.1])
    rsm = i10_scan.binned_reciprocal_space_map(frame, start, stop, step)

    print(rsm)
    print(np.argmax(rsm))
    assert 1 == 2
