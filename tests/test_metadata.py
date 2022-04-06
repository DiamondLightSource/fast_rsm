"""
This module contains tests for the RSMapper.metadata module. As the angular
metadata is an important part of the reciprocal space mapping, instances of
RSMapper.metadata.Metadata actually do a reasonable amount of heavy lifting.
"""

# Obviously we need to test protected members.
# pylint: disable=protected-access

from copy import deepcopy

import numpy as np
from numpy.testing import assert_almost_equal

from RSMapper.metadata import Metadata


def test_metadata_init():
    """
    For whatever reason, I'm partial to having a test explode when I rename
    attributes set in __init__.
    """
    # Make some nonsense metadata.
    metadata = Metadata(None, "0", 1, 2, 3, (6, 7), (4, 5))

    assert metadata.instrument == "0"
    assert metadata.detector_distance == 1
    assert metadata.pixel_size == 2
    assert metadata.energy == 3
    assert metadata.data_shape == (6, 7)
    assert metadata.beam_centre == (4, 5)
    assert metadata._solid_angles is None
    assert metadata._relative_phi is None
    assert metadata._relative_theta is None


def test_init_relative_theta(metadata_01: Metadata):
    """
    Make sure that the all-important metadata._relative_theta is initialized
    correctly. This is used in all reciprocal space mapping routines.
    """
    metadata_01._init_relative_theta()

    # Now check some values (manually calculated).
    assert (metadata_01._relative_theta[20, :] == 0).all()
    assert_almost_equal(
        metadata_01._relative_theta[1020, 15], -0.0499583957, 9)
    assert_almost_equal(
        metadata_01._relative_theta[0, 1278], 0.000999999667, 12)
    assert_almost_equal(
        metadata_01._relative_theta[19, 0], 5e-5, 12)
    assert_almost_equal(
        metadata_01._relative_theta[21, 1999], -5e-5, 12)


def test_init_relative_phi(metadata_01: Metadata):
    """
    Make sure that metadata._relative_phi is initialized correctly. This is
    used in all reciprocal space mapping routines.
    """
    metadata_01._init_relative_phi()

    # Now check some values (manually calculated).
    assert (metadata_01._relative_phi[:, 80] == 0).all()
    assert_almost_equal(
        metadata_01._relative_phi[0, 0], -0.00399997867, 10)
    assert_almost_equal(  # Index -1 is index 1999
        metadata_01._relative_phi[234, -1], 0.0956571644, 10)
    assert_almost_equal(
        metadata_01._relative_phi[1263, 1500], 0.0708810559, 10)
    assert_almost_equal(
        metadata_01._relative_phi[1863, 945], 0.0432230629, 10)


def test_relative_theta(metadata_01: Metadata):
    """
    Make sure that init relative theta is only called once.
    """
    _ = metadata_01.relative_theta
    metadata_01._init_relative_theta = lambda: 1/0

    # This shouldn't raise because _init_relative_theta shouldn't run again.
    assert (metadata_01.relative_theta[20, :] == 0).all()


def test_relative_phi(metadata_01: Metadata):
    """
    Make sure that init relative phi is only called once. Make sure that it's
    returning the correct array.
    """
    _ = metadata_01.relative_phi
    metadata_01._init_relative_phi = lambda: 1/0

    # This shouldn't raise because _init_relative_phi shouldn't run again.
    assert (metadata_01.relative_phi[:, 80] == 0).all()


def test_init_solid_angles(metadata_01: Metadata):
    """
    Make sure that initializing the solid angles doesn't corrupt our relative
    theta/phi arrays. This isn't trivial since, during the calculation of solid
    angles, the data shape is hacked and relative_theta+relative_phi are
    recalculated twice each.
    """
    relative_theta = deepcopy(metadata_01.relative_theta)
    print(relative_theta.shape)
    relative_phi = deepcopy(metadata_01.relative_phi)
    metadata_01._init_solid_angles()

    assert (relative_theta == metadata_01.relative_theta).all()
    assert (relative_phi == metadata_01.relative_phi).all()


def test_solid_angle_init_once(metadata_01: Metadata):
    """
    Make sure that solid angles are initialized exactly once.
    """
    _ = metadata_01.solid_angles
    metadata_01._init_solid_angles = lambda: 1/0

    # This shouldn't raise because _init_solid_angle shouldn't run again.
    _ = metadata_01.solid_angles


def test_solid_angle_values(metadata_01: Metadata):
    """
    Make sure that the solid angles follow the correct trend (the biggest solid
    angle should be at the PONI pixel). Note that, as a quirk of the current
    approximate solid angle calculation, there are 4 pixels with a maximal solid
    angle.
    """
    max_pixels = np.where(metadata_01.solid_angles == np.max(
        metadata_01.solid_angles))
    min_pixels = np.where(metadata_01.solid_angles == np.min(
        metadata_01.solid_angles))
    assert 20 in max_pixels[0]
    assert 80 in max_pixels[1]
    assert 1999 in min_pixels[0]
    assert 1999 in min_pixels[1]
    assert len(max_pixels[0]) == 4
    assert len(max_pixels[1]) == 4
