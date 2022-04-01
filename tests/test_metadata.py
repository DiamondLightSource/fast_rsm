"""
This module contains tests for the RSMapper.metadata module. As the angular
metadata is an important part of the reciprocal space mapping, instances of
RSMapper.metadata.Metadata actually do a reasonable amount of heavy lifting.
"""

# Obviously we need to test protected members.
# pylint: disable=protected-access

from numpy.testing import assert_almost_equal

from RSMapper.metadata import Metadata


def test_metadata_init():
    """
    For whatever reason, I'm partial to having a test explode when I rename
    attributes set in __init__.
    """
    # Make some nonsense metadata.
    metadata = Metadata(1, 2, 3, (6, 7), (4, 5))

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
    assert (metadata_01._relative_theta[20][:] == 0).all()
    assert_almost_equal(
        metadata_01._relative_theta[1020][15], -0.0499583957, 9)
    assert_almost_equal(
        metadata_01._relative_theta[0][1278], 0.000999999667, 12)
    assert_almost_equal(
        metadata_01._relative_theta[19][0], 5e-5, 12)
    assert_almost_equal(
        metadata_01._relative_theta[21][1999], -5e-5, 12)
