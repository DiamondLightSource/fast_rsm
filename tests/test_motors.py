"""
This module tests the RSMapper.motors.Motors class, which is used to calculate
detector position in spherical polars from motor positions.
"""

# pylint: disable=protected-access

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from RSMapper.io import i07_nexus_parser, i10_nxs_parser
from RSMapper.metadata import Metadata
from RSMapper.motors import Motors


def test_reflection(metadata_01: Metadata):
    """
    Make sure that we're calling the correct methods via reflection when we
    call detector_polar and detector_azimuth.
    """
    motors = Motors(metadata_01)

    # metadata_01.instrument = "my_instrument"
    motors._my_instrument_detector_polar = lambda: 1/0
    motors._my_instrument_detector_azimuth = lambda: 1/0

    with pytest.raises(ZeroDivisionError):
        _ = motors.detector_polar

    with pytest.raises(ZeroDivisionError):
        _ = motors.detector_azimuth


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

    assert_almost_equal(motors.detector_polar, 90, 2)
    assert_almost_equal(motors.detector_azimuth, np.arange(15, 75, 15), 2)


def test_i10_detector_theta(i10_nx_01: str,
                            i10_beam_centre_01,
                            i10_pimte_detector_distance):
    """
    Make sure that we can correctly retrieve the detector's spherical polar
    polar angle.
    """
