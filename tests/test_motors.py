"""
This module tests the RSMapper.motors.Motors class, which is used to calculate
detector position in spherical polars from motor positions.
"""

# pylint: disable=protected-access

import pytest

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
