"""
This module contains unit tests for the RSMapper.image.Image class.

As of 07/04/2022, I'm lazily grabbing images from scans. This could be improved
by manually loading instances of Image, which would just require a slightly
clever fixture.
"""

# pylint: disable=protected-access

from time import time

import numpy as np
from numpy.testing import assert_allclose

from diffraction_utils import Frame

from RSMapper.binning import linear_bin
from RSMapper.scan import Scan


def test_data(i10_scan: Scan):
    """
    Make sure that image_instance.data is properly normalized.
    """
    image = i10_scan.load_image(0)
    metadata = i10_scan.metadata
    assert (image.data == image._raw_data/metadata.solid_angles).all()


def test_delta_q_01(i10_scan: Scan):
    """
    Make sure that the scattering vectors are being correctly calculated. This
    is tricky to do exactly, but we can get pretty close by grabbing the
    brightest pixel and assuming that we scatter through the (100) to get there.
    """
    # We should be on the Bragg peak in the centre image.
    image = i10_scan.load_image(70)

    frame = Frame(Frame.hkl, i10_scan.metadata.diffractometer)

    # Brightest pixel should be on the Bragg peak.
    max_intensity_pixel = np.where(image.data == np.max(image.data))
    max_intensity_q = np.linalg.norm(image.delta_q(frame)[max_intensity_pixel])
    wavelength = 1/max_intensity_q

    # Published value of CSO unit cell length is 8.931 Å.
    # Make sure that the brightest pixel corresponds to scattering from the
    # (100) of a crystal whose lattice constant is within 1% of this value.
    assert_allclose(wavelength, 8.931, rtol=1e-2)


def test_binning(i10_scan: Scan):
    """
    Make sure that no data is being lost when binning is carried out. This test
    is implicitly testing _fix_intensity_geometry and _fix_delta_q_geometry.
    """
    image = i10_scan.load_image(67)
    frame = Frame(Frame.hkl, i10_scan.metadata.diffractometer)
    delta_q = image.delta_q(frame)

    min0 = np.min(delta_q[:, :, 0])
    min1 = np.min(delta_q[:, :, 1])
    min2 = np.min(delta_q[:, :, 2])
    max0 = np.max(delta_q[:, :, 0])
    max1 = np.max(delta_q[:, :, 1])
    max2 = np.max(delta_q[:, :, 2])

    step = np.array([(max0 - min0)/20, (max1 - min1)/20, (max2 - min2)/100])
    start = np.array([min0, min1, min2])
    stop = np.array([max0, max1, max2]) + step

    # Try to bin. Add an assert 1 == 2 to see single threaded bin performance.
    time1 = time()
    binned = linear_bin(delta_q, image.data, start, stop, step)
    time2 = time()
    print(time2 - time1)

    assert_allclose(np.sum(binned), np.sum(image.data))
