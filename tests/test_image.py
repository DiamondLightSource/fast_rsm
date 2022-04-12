"""
This module contains unit tests for the RSMapper.image.Image class.

As of 07/04/2022, I'm lazily grabbing images from scans. This could be improved
by manually loading instances of Image, which would just require a slightly
clever fixture.
"""

# pylint: disable=protected-access

from typing import Tuple, List

import numpy as np
from numpy.testing import assert_allclose


from RSMapper.image import Image
from RSMapper.metadata import Metadata


def test_data(i10_parser_output_01: Tuple[List[Image], Metadata]):
    """
    Make sure that image_instance.data is properly normalized.
    """
    images, metadata = i10_parser_output_01
    assert (images[0].data == images[0]._raw_data/metadata.solid_angles).all()


def test_delta_q_01(i10_parser_output_01: Tuple[List[Image], Metadata]):
    """
    Make sure that the scattering vectors are being correctly calculated. This
    is tricky to do exactly, but we can get pretty close by grabbing the
    brightest pixel and assuming that we scatter through the (100) to get there.
    """
    images, _ = i10_parser_output_01

    # We should be on the Bragg peak in the centre image.
    image = images[70]

    # Brightest pixel should be on the Bragg peak.
    max_intensity_pixel = np.where(image.data == np.max(image.data))
    max_intensity_q = np.linalg.norm(image.delta_q[max_intensity_pixel])
    wavelength = 1/max_intensity_q

    # Published value of CSO unit cell length is 8.931 Å.
    # Make sure that the brightest pixel corresponds to scattering from the
    # (100) of a crystal whose lattice constant is within 1% of this value.
    assert_allclose(wavelength, 8.931, rtol=1e-2)
