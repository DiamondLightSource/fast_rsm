"""
This module contains tests for the RSMapper.scan.Scan class.
"""

import numpy as np

from diffraction_utils import Frame
from RSMapper.scan import Scan


def test_rsm(i10_scan: Scan):
    """
    Make sure that we can run an rsm in reasonable time.

    TODO: set memory/time limitations on this test.
    """
    frame = Frame(Frame.hkl, i10_scan.metadata.diffractometer)
    i10_scan.reciprocal_space_map(frame)
