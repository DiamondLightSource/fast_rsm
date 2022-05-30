"""
This module tests RSMapper's ability to write data.
"""

# pylint: disable=unreachable

import numpy as np

from diffraction_utils import Frame

from RSMapper.writing import linear_bin_to_vtk
from RSMapper.scan import Scan


def test_linear_bin_to_vtk(i10_scan: Scan):
    """
    Make sure that we can deal with linearly binned data.

    TODO: set memory/time limitations on this test. Test output vs a stored .vtk
    file. Make this optional and make slow tests run via a flag.
    """
    return
    frame = Frame(Frame.hkl, i10_scan.metadata.diffractometer)

    start = np.array([-0.0015, 0.11, -0.005])
    stop = np.array([0.002, 0.115, 0.003])
    step = np.array([0.00002, 0.00002, 0.00002])
    rsm = i10_scan.binned_reciprocal_space_map(frame, start, stop, step, 10)

    assert np.max(rsm) > 0
    linear_bin_to_vtk(rsm, "test.vtk", start, stop, step)
    assert 1 == 2
