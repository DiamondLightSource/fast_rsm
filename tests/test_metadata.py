"""
This module contains tests for the fast_rsm.metadata module. As the angular
metadata is an important part of the reciprocal space mapping, instances of
fast_rsm.metadata.Metadata actually do a reasonable amount of heavy lifting.
"""

# Obviously we need to test protected members.
# pylint: disable=protected-access

# Pylint hates electron volts.
# pylint: disable=invalid-name

from copy import deepcopy

import numpy as np
from numpy.testing import assert_almost_equal

from fast_rsm.rsm_metadata import RSMMetadata


def test_metadata_init(i10_metadata: RSMMetadata, i10_nxs_path):
    """
    For whatever reason, I'm partial to having a test explode when I rename
    attributes set in __init__.
    """

    assert i10_metadata.data_file.local_path == i10_nxs_path
    assert i10_metadata.data_file.detector_distance == 0.1363
    assert i10_metadata.data_file.pixel_size == 13.5e-6
    assert i10_metadata.data_file.probe_energy == 931.7725
    assert i10_metadata.data_file.image_shape == (2048, 2048)
    assert i10_metadata.beam_centre == (998, 1016)
    assert i10_metadata._solid_angles is None
    assert i10_metadata._relative_azimuth is None
    assert i10_metadata._relative_polar is None


def test_init_relative_polar(i10_metadata: RSMMetadata):
    """
    Make sure that the all-important metadata._relative_polar is initialized
    correctly. This is used in all reciprocal space mapping routines.
    """
    i10_metadata._init_relative_polar()

    # Now check some values (manually calculated).
    assert (i10_metadata._relative_polar[998, :] == 0).all()
    assert_almost_equal(i10_metadata._relative_polar[1020, 15], -0.00217901343)
    assert_almost_equal(i10_metadata._relative_polar[0, 1278], 0.0985280567)
    assert_almost_equal(i10_metadata._relative_polar[19, 0], 0.0966640471)
    assert_almost_equal(i10_metadata._relative_polar[21, 1999], 0.0964677961)


def test_init_relative_azimuth(i10_metadata: RSMMetadata):
    """
    Make sure that metadata._relative_azimuth is initialized correctly. This is
    used in all reciprocal space mapping routines.
    """
    i10_metadata._init_relative_azimuth()

    # Now check some values (manually calculated).
    assert (i10_metadata._relative_azimuth[:, 1016] == 0).all()
    assert_almost_equal(i10_metadata._relative_azimuth[0, 0], 0.100293327)
    assert_almost_equal(i10_metadata._relative_azimuth[234, -1], -0.101763908)
    assert_almost_equal(
        i10_metadata._relative_azimuth[1263, 1500], -0.047901699)
    assert_almost_equal(
        i10_metadata._relative_azimuth[12, 945], 0.00703216581)


def test_relative_polar(i10_metadata: RSMMetadata):
    """
    Make sure that init relative theta is only called once.
    """
    _ = i10_metadata.relative_polar
    i10_metadata._init_relative_polar = lambda: 1/0

    # This shouldn't raise because _init_relative_polar shouldn't run again.
    assert (i10_metadata.relative_polar[998, :] == 0).all()


def test_relative_azimuth(i10_metadata: RSMMetadata):
    """
    Make sure that init relative phi is only called once. Make sure that it's
    returning the correct array.
    """
    _ = i10_metadata.relative_azimuth
    i10_metadata._init_relative_azimuth = lambda: 1/0

    # This shouldn't raise because _init_relative_azimuth shouldn't run again.
    assert (i10_metadata.relative_azimuth[:, 1016] == 0).all()


def test_init_solid_angles(i10_metadata: RSMMetadata):
    """
    Make sure that initializing the solid angles doesn't corrupt our relative
    theta/phi arrays. This isn't trivial since, during the calculation of solid
    angles, the data shape is hacked and relative_polar+relative_azimuth are
    recalculated twice each.
    """
    relative_polar = deepcopy(i10_metadata.relative_polar)
    relative_azimuth = deepcopy(i10_metadata.relative_azimuth)
    i10_metadata._init_solid_angles()

    assert (relative_polar == i10_metadata.relative_polar).all()
    assert (relative_azimuth == i10_metadata.relative_azimuth).all()


def test_solid_angle_init_once(i10_metadata: RSMMetadata):
    """
    Make sure that solid angles are initialized exactly once.
    """
    _ = i10_metadata.solid_angles
    i10_metadata._init_solid_angles = lambda: 1/0

    # This shouldn't raise because _init_solid_angle shouldn't run again.
    _ = i10_metadata.solid_angles


def test_solid_angle_values(i10_metadata: RSMMetadata):
    """
    Make sure that the solid angles follow the correct trend (the biggest solid
    angle should be at the PONI pixel). Note that, as a quirk of the current
    approximate solid angle calculation, there are 4 pixels with a maximal solid
    angle.
    """
    max_pixels = np.where(i10_metadata.solid_angles == np.max(
        i10_metadata.solid_angles))
    min_pixels = np.where(i10_metadata.solid_angles == np.min(
        i10_metadata.solid_angles))
    assert 998 in max_pixels[0]
    assert 1016 in max_pixels[1]
    assert 2047 in min_pixels[0]
    assert 2047 in min_pixels[1]
    assert len(max_pixels[0]) == 4
    assert len(max_pixels[1]) == 4


def test_incident_wavelength(i10_metadata: RSMMetadata):
    """
    Assert that we can correctly calculate wavelength from energies.
    """
    # We're using the Cu L-3 edge.
    assert_almost_equal(i10_metadata.incident_wavelength, 13.30627304411753, 6)


def test_k_incident_length(i10_metadata: RSMMetadata):
    """
    Make sure that our q-vector calculation is correct. Test it against Cu
    k-alpha.
    """
    # We're using the Cu L-3 edge.
    assert_almost_equal(i10_metadata.k_incident_length, 1/13.30627304411753)
