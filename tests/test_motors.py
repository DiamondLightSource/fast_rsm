"""
This module tests the RSMapper.motors.Motors class, which is used to calculate
detector position in spherical polars from motor positions.
"""

# pylint: disable=protected-access

from typing import List, Tuple

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.spatial.transform import Rotation

from RSMapper.image import Image
from RSMapper.io import i07_nexus_parser
from RSMapper.metadata import Metadata
from RSMapper.motors import Motors, vector_to_azimuth_polar


def test_reflection(metadata_01: Metadata):
    """
    Make sure that we're calling the correct methods via reflection when we
    call detector_polar and detector_azimuth.
    """
    motors = Motors(metadata_01, 0)

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


def test_i10_detector_polar_coords_01(
        i10_parser_output_01: Tuple[List[Image], Metadata]):
    """
    Make sure that we can correctly retrieve the detector's spherical polar
    polar angle. Check this on frame 70 of the scan.
    """
    # Grab the parser output.
    images, _ = i10_parser_output_01

    # On frame 70, these are our tth and chi values.
    tth = 96.519
    chi = -1

    # Prepare some rotations.
    tth_rot = Rotation.from_euler('xyz', degrees=True,
                                  angles=[-tth, 0, 0])
    chi_rot = Rotation.from_euler('xyz', degrees=True,
                                  angles=[0, 0, chi])
    total_rot = chi_rot * tth_rot

    # Rotate the beam.
    beam_direction = [0, 0, 1]
    beam_direction = total_rot.apply(beam_direction)
    azimuth, polar = vector_to_azimuth_polar(beam_direction)

    # Make sure this was correctly calculated by the Motors class.
    motors = images[70].motors
    assert_almost_equal(azimuth, motors.detector_azimuth, decimal=5)
    assert_almost_equal(polar, motors.detector_polar, decimal=5)


def test_i10_detector_polar_coords_02(
        i10_parser_output_01: Tuple[List[Image], Metadata]):
    """
    As above, but checking frame 0 of the scan, using the fact that tth should
    be 3.5 degrees above 96.519 on the first frame (which I got by looking at
    the nexus file).
    """
    # Grab the parser output.
    images, _ = i10_parser_output_01

    # On frame 70, these are our tth and chi values.
    tth = 96.519 + 3.5
    chi = -1

    # Prepare some rotations.
    tth_rot = Rotation.from_euler('xyz', degrees=True,
                                  angles=[-tth, 0, 0])
    chi_rot = Rotation.from_euler('xyz', degrees=True,
                                  angles=[0, 0, chi])
    total_rot = chi_rot * tth_rot

    # Rotate the beam.
    beam_direction = [0, 0, 1]
    beam_direction = total_rot.apply(beam_direction)
    azimuth, polar = vector_to_azimuth_polar(beam_direction)

    # Make sure this was correctly calculated by the Motors class.
    motors = images[0].motors
    assert_almost_equal(azimuth, motors.detector_azimuth, decimal=5)
    assert_almost_equal(polar, motors.detector_polar, decimal=5)
