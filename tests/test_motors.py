"""
This module tests the RSMapper.motors.Motors class, which is used to calculate
detector position in spherical polars from motor positions.
"""

# pylint: disable=protected-access

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from RSMapper.io import i07_nexus_parser
from RSMapper.metadata import Metadata
from RSMapper.motors import Motors


def test_reflection(metadata_01: Metadata):
    """
    Make sure that we're calling the correct methods via reflection when we
    call detector_theta and detector_phi.
    """
    motors = Motors(metadata_01)

    # metadata_01.instrument = "my_instrument"
    motors._theta_from_my_instrument = lambda: 1/0
    motors._phi_from_my_instrument = lambda: 1/0

    with pytest.raises(ZeroDivisionError):
        _ = motors.detector_theta

    with pytest.raises(ZeroDivisionError):
        _ = motors.detector_phi


def test_i07_phi_theta(path_to_i07_nx_01: str,
                       i07_beam_centre_01: tuple,
                       i07_detector_distance_01: float):
    """
    Make sure that we can work out where the detector is on I07.
    """
    images, _ = i07_nexus_parser(path_to_i07_nx_01,
                                 i07_beam_centre_01,
                                 i07_detector_distance_01)

    motors = images[0].motors

    assert_almost_equal(motors.detector_theta, 90, 2)
    assert_almost_equal(motors.detector_phi, np.arange(15, 75, 15), 2)
