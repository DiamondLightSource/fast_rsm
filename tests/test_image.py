"""
This module contains unit tests for the RSMapper.image.Image class.

As of 07/04/2022, I'm lazily grabbing images from scans. This could be improved
by manually loading instances of Image, which would just require a slightly
clever fixture.
"""

# pylint: disable=protected-access

from RSMapper.scan import Scan


def test_data(i10_scan_01: Scan):
    """
    Make sure that image_instance.data is properly normalized.
    """
    data = i10_scan_01.images[0].data
    raw_data = i10_scan_01.images[0]._raw_data
    solid_angles = i10_scan_01.metadata.solid_angles

    assert (data == raw_data/solid_angles).all()
