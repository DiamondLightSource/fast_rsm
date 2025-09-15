"""
This module contains a series of corrections that can be applied to detector
images.
"""

import numpy as np

import mapper_c_utils

# pylint: disable=c-extension-no-member


def lorentz(intensities: np.ndarray, k_in: np.ndarray, k_out: np.ndarray):
    """
    Applies Lorentz corrections to the intensities passed as an argument. This
    function wraps mapper_c_utils.lorentz_correction, making sure that arguments
    have the correct dtype and shape.

    Args:
        intensities:
            The intensity values we would like to correct. This should have
            shape (N).
        k_in:
            The wavevector of the incident light. This should have shape (3,).
        k_out:
            The wavevector of the outgoing light. This should have shape (N, 3).

    Returns:
        The updated intensities. You don't need to use the returned value since
        the original intensity array will have been updated, but this way the
        function *just works* however you use it.
    """
    # Make sure the dtype is good.
    if intensities.dtype != np.float32:
        intensities = intensities.astype(np.float32)
    if k_in.dtype != np.float32:
        k_in = k_in.astype(np.float32)
    if k_out.dtype != np.float32:
        k_out = k_out.astype(np.float32)

    # Make sure that the shapes are good.
    intensity_shape = intensities.shape
    k_out_shape = k_out.shape
    intensities = intensities.reshape(intensities.size)
    k_out = k_out.reshape((int(k_out.size / 3), 3))

    # Call the C function. This directly affects the elements of the
    # intensities array.
    mapper_c_utils.lorentz_correction(k_in, k_out, intensities)

    # #hardcode debugging lines to save correction factors
    # intensitiesones=np.ones(np.shape(intensities))
    # mapper_c_utils.lorentz_correction(k_in, k_out, intensitiesones)
    # intensitiesones = intensitiesones.reshape(intensity_shape)
    # np.save('/home/rpy65944/fast_rsm/lorentzcorrs',intensitiesones)

    # Return the shapes to their original values.
    intensities = intensities.reshape(intensity_shape)
    k_out = k_out.reshape(k_out_shape)


def linear_polarisation(intensities: np.ndarray, k_out: np.ndarray,
                        polarisation_vector: np.ndarray):
    """
    Carries out an exact polarisation correction assuming that the incident
    light is perfectly linearly polarised.

    Args:
        intensities:
            The intensity values we would like to correct. This should have
            shape (N).
        k_out:
            The wavevector of the outgoing light. This should have shape (N, 3).
        polarisation_vector:
            A vector describing the polarisation of the incident beam.

    Returns:
        The updated intensities. You don't need to use the returned value since
        the original intensity array will have been updated, but this way the
        function *just works* however you use it.
    """
    # Make sure the dtype is good.
    if intensities.dtype != np.float32:
        intensities = intensities.astype(np.float32)
    if k_out.dtype != np.float32:
        k_out = k_out.astype(np.float32)
    if polarisation_vector.dtype != np.float32:
        polarisation_vector = polarisation_vector.astype(np.float32)

    # Make sure that the shapes are good.
    intensity_shape = intensities.shape
    k_out_shape = k_out.shape
    intensities = intensities.reshape(intensities.size)
    k_out = k_out.reshape((int(k_out.size / 3), 3))

    # Call the C function. This directly affects the elements of the
    # intensities array.
    # print('corrected version with linear polarisation correction')
    mapper_c_utils.linear_pol_correction(
        polarisation_vector, k_out, intensities)

    # add in incorrect line to replicate previous version
    # print("previous version - incorrectly has double  'lorentz'")
    # mapper_c_utils.lorentz_correction(polarisation_vector, k_out, intensities)

    # #hardcode debugging lines to save correction factors
    # intensitiesones=np.ones(np.shape(intensities))
    # mapper_c_utils.linear_pol_correction(polarisation_vector, k_out, intensitiesones)
    # intensitiesones = intensitiesones.reshape(intensity_shape)
    # np.save('/home/rpy65944/fast_rsm/linpolcorrs',intensitiesones)

    # Return the shapes to their original values.
    intensities = intensities.reshape(intensity_shape)
    k_out = k_out.reshape(k_out_shape)
