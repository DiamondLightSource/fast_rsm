"""
This module tests the RSMapper.motors.Motors class, which is used to calculate
detector position in spherical polars from motor positions.
"""

# pylint: disable=protected-access

from typing import List, Tuple

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_allclose
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


def test_i10_detector_polar_coords_lab_frame_01(
        i10_parser_output_01: Tuple[List[Image], Metadata]):
    """
    *THIS TEST IS TESTING ANGLES IN THE LAB FRAME.*

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
    azimuth_lab, polar_lab = motors._i10_detector_angles_lab_frame

    assert_almost_equal(azimuth, azimuth_lab, decimal=5)
    assert_almost_equal(polar, polar_lab, decimal=5)


def test_i10_detector_polar_coords_lab_frame_02(
        i10_parser_output_01: Tuple[List[Image], Metadata]):
    """
    *THIS TEST IS TESTING ANGLES IN THE LAB FRAME.*

    As above, but checking frame 0 of the scan, using the fact that tth should
    be 3.5 degrees above 96.519 on the first frame (which I got by looking at
    the nexus file).
    """
    # Grab the parser output.
    images, _ = i10_parser_output_01

    # On frame 0, these are our tth and chi values.
    tth = 96.519 + 3.5
    chi = -1

    # Prepare some rotations.
    tth_rot = Rotation.from_euler('xyz', degrees=True, angles=[-tth, 0, 0])
    chi_rot = Rotation.from_euler('xyz', degrees=True, angles=[0, 0, chi])
    total_rot = chi_rot * tth_rot

    # Rotate the beam.
    beam_direction = [0, 0, 1]
    beam_direction = total_rot.apply(beam_direction)
    azimuth, polar = vector_to_azimuth_polar(beam_direction)

    # Make sure this was correctly calculated by the Motors class.
    motors = images[0].motors
    azimuth_lab, polar_lab = motors._i10_detector_angles_lab_frame

    assert_almost_equal(azimuth, azimuth_lab, decimal=5)
    assert_almost_equal(polar, polar_lab, decimal=5)


def test_i10_sample_rotation(
        i10_parser_output_01: Tuple[List[Image], Metadata]):
    """
    Make sure that we can rotate into the frame of our sample. Do this by
    manually reading theta and chi from the .nxs file. Apply the rotation of
    the sample surface to a [0, 1, 0] vector, where the rotation is calculated
    by the Motors class. Then, use the manually read theta and chi to invert
    the transformation.
    """
    # Grab the parser output.
    images, _ = i10_parser_output_01

    surface_normal = [0, 1, 0]
    surface_normal = images[70].motors.sample_rotation.apply(surface_normal)

    # Sample theta read manually from .nxs file
    theta = 49.6284 - 3.5/2
    chi = -1  # wasn't scanned.

    # Now try to use these two values to invert the rotation!
    reverse_theta_rot = Rotation.from_euler('xyz', [theta, 0, 0], True)
    reverse_chi_rot = Rotation.from_euler('xyz', [0, 0, -chi], True)
    total_reverse_rot = reverse_theta_rot * reverse_chi_rot

    surface_normal = total_reverse_rot.apply(surface_normal)
    assert_allclose(surface_normal, np.array([0, 1, 0]), atol=0.02)


def test_i10_sample_frame_angles_01(
        i10_parser_output_01: Tuple[List[Image], Metadata]):
    """
    Make sure that we can calculate quantities properly in the frame of
    reference of the sample, which is rotated with respect to the lab.
    """
    images, _ = i10_parser_output_01
    motors = images[0].motors

    # tth, th values on frame 0.
    tth = 96.519 + 3.5
    theta = 49.6284
    chi = -1

    # Highly scuffed approximation (I'm ignoring chi).
    # NOTE that the "90 - " part comes from spherical polars being defined from
    # the y-axis.
    scuffed_detector_polar = 90 - (tth - theta)
    # The azimuthal angle is hard to estimate. It probably shouldn't be too big,
    # since the detector is reasonably far from 90 degrees and chi is small.
    # Also, minus signs are tricky because rotations have a dumb handedness.
    extra_scuffed_azimuth = -2*chi

    # Assert that our chi-less approx is within 0.05 deg of being correct.
    # The scuffed approx is so accurate because chi is small, cos(chi) ~ 0.
    assert_allclose(scuffed_detector_polar, motors.detector_polar*180/np.pi,
                    atol=0.05)
    assert_allclose(extra_scuffed_azimuth, motors.detector_azimuth*180/np.pi,
                    atol=0.5)


def test_i10_sample_frame_incident_beam(
        i10_parser_output_01: Tuple[List[Image], Metadata]):
    """
    Make sure that we can compute a normal vector parallel to the incident beam
    in the Motors class for experiments carried out in RASOR in i10.
    """
    images, _ = i10_parser_output_01
    motors = images[0].motors
    theta_rad = 49.6284*np.pi/180

    incident_beam = motors.incident_beam

    # Incident beam should always be in the y-z plane in RASOR.
    assert incident_beam[0] == 0
    assert_almost_equal(incident_beam[1], -np.sin(theta_rad), decimal=5)
    assert_almost_equal(incident_beam[2], np.cos(theta_rad), decimal=5)
