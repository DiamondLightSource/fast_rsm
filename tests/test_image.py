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

from diffraction_utils import Frame

from RSMapper.scan import Scan


def test_data(i10_scan: Scan):
    """
    Make sure that image_instance.data is properly normalized.
    """
    images = i10_scan.images
    metadata = i10_scan.metadata
    assert (images[0].data == images[0]._raw_data/metadata.solid_angles).all()


def test_delta_q_01(i10_scan: Scan):
    """
    Make sure that the scattering vectors are being correctly calculated. This
    is tricky to do exactly, but we can get pretty close by grabbing the
    brightest pixel and assuming that we scatter through the (100) to get there.
    """
    # We should be on the Bragg peak in the centre image.
    image = i10_scan.images[70]

    frame = Frame(Frame.hkl, i10_scan.metadata.diffractometer)

    # Brightest pixel should be on the Bragg peak.
    max_intensity_pixel = np.where(image.data == np.max(image.data))
    max_intensity_q = np.linalg.norm(image.delta_q(frame)[max_intensity_pixel])
    wavelength = 1/max_intensity_q

    # Published value of CSO unit cell length is 8.931 Å.
    # Make sure that the brightest pixel corresponds to scattering from the
    # (100) of a crystal whose lattice constant is within 1% of this value.
    assert_allclose(wavelength, 8.931, rtol=1e-2)
