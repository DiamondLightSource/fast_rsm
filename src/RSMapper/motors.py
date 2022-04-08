"""
This module contains a convenience class for tracking motor positions.
"""

# Because of the dumb way that nexusformat works.
# pylint: disable=protected-access

from typing import Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation

from .metadata import Metadata


def vector_to_azimuth_polar(vector: np.ndarray):
    """
    Takes a 3D vector. Returns phi, theta spherical polar angles.

    Polar angle is measured from synchrotron y (vertically up).
    Azimuthal angle is measured from synchrotron z (along the beam).

    Args:
        vector:
            The vector to map to spherical polars.

    Returns:
        A tuple of (azimuthal_angle, polar_angle)
    """
    theta = np.arccos(vector[1])
    phi = np.arccos(vector[2]/np.sin(theta))

    return phi, theta


class Motors:
    """
    Can calculate relative detector/sample orientation from motor positions.

    Attrs:
        metadata:
            The scan's metadata.
        index:
            Motor positions vary throughout a scan. The index attribute lets
            our Motor instance know which image it referes to. For example, if
            four images were taken in a scan and self.index=3, then this
            instance of Motors refers to the motor positions for the final
            image.
    """

    def __init__(self, metadata: Metadata, index: int) -> None:
        self.metadata = metadata
        self.index = index

    def array_to_correct_element(
            self, maybe_array: Union[float, np.ndarray]) -> float:
        """
        Takes motor positions, which could be a float (if the motor isn't
        scanned during this scan) or an array (if the motor is [one of] the
        scan's independent variables).

        Args:
            maybe_array:
                Either a float, or an array, depending of if this maybe_array
                represents the values of an independent variable or not.

        Returns maybe_array if it wasn't a float, maybe_array[self.index]
            otherwise.
        """
        if isinstance(maybe_array, np.ndarray):
            return maybe_array[self.index]
        return maybe_array

    @property
    def sample_rotation(self) -> Rotation:
        """
        Returns a scipy.spatial.transform.Rotation representation of the
        rotation that the motors have applied to the sample. This can be used
        to map vectors into coordinate systems tied to the sample.
        """
        return getattr(self, f"_{self.metadata.instrument}_sample_rotation")

    @property
    def _i10_sample_rotation(self) -> Rotation:
        """
        Samples can only be affected by theta and chi in RASOR, so this one's
        pretty easy.
        """
        chi = self.array_to_correct_element(self.metadata.metadata_file[
            "/entry/instrument/rasor/diff/chi"]._value - 90)
        theta = self.array_to_correct_element(self.metadata.metadata_file[
            "/entry/instrument/th/value"]._value - 90)

        # Prepare rotation matrices.
        tth_rot = Rotation.from_euler('xyz', degrees=True,
                                      angles=[-theta, 0, 0])
        chi_rot = Rotation.from_euler('xyz', degrees=True,
                                      angles=[0, 0, chi])

        # Return the properly ordered composition of these rotations.
        return chi_rot * tth_rot

    @property
    def detector_polar(self) -> float:
        """
        Returns the detector's spherical polar theta value.
        """
        # Call the appropriate function for the instrument in use.
        return getattr(self, f"_{self.metadata.instrument}_detector_polar")()

    @property
    def detector_azimuth(self) -> float:
        """
        Returns the detector's spherical polar phi value.
        """
        return getattr(self, f"_{self.metadata.instrument}_detector_azimuth")()

    @property
    def _i07_phi_theta(self) -> Tuple[float, float]:
        """
        Returns (phi, theta) assuming that the metadata file is an I07 file.
        """
        angles_dict = {}
        angles_dict["alpha"] = self.metadata.metadata_file[
            "/entry/instrument/diff1alpha/value"]._value
        angles_dict["gamma"] = self.metadata.metadata_file[
            "/entry/instrument/diff1gamma/value"]._value
        angles_dict["delta"] = self.metadata.metadata_file[
            "/entry/instrument/diff1delta/value"]._value
        angles_dict["chi"] = self.metadata.metadata_file[
            "/entry/instrument/diff1chi/value"]._value
        angles_dict["omega"] = self.metadata.metadata_file[
            "/entry/instrument/diff1omega/value"]._value
        angles_dict["theta"] = self.metadata.metadata_file[
            "/entry/instrument/diff1theta/value"]._value

        # ...maths goes here...

        theta = angles_dict['theta']
        phi = angles_dict['gamma']

        return phi, theta

    def _i07_detector_polar(self) -> float:
        """
        Parses self.metadata.metadata_file to calculate our current theta;
        assumes that the data was recorded at beamline I07 at Diamond.
        """
        return self._i07_phi_theta[1]

    def _i07_detector_azimuth(self) -> float:
        """
        Parses self.metadata.metadata_file to calculate our current phi; assumes
        that the data was acquired at Diamond's beamline I07.
        """
        return self._i07_phi_theta[0]

    @property
    def _i10_detector_angles(self):
        """
        Calculates the detector's azimuthal and polar angles, assuming we're in
        the RASOR diffractometer at beamline I10 in Diamond.

        TODO: check orientation of chi with beamline to fix a sign.
        """
        tth_area = self.array_to_correct_element(-self.metadata.metadata_file[
            "/entry/instrument/tth/value"]._value + 90)
        chi = self.array_to_correct_element(self.metadata.metadata_file[
            "/entry/instrument/rasor/diff/chi"]._value - 90)

        # Prepare rotation matrices.
        tth_rot = Rotation.from_euler('xyz', degrees=True,
                                      angles=[-tth_area, 0, 0])
        chi_rot = Rotation.from_euler('xyz', degrees=True,
                                      angles=[0, 0, chi])
        total_rot = chi_rot * tth_rot  # This does a proper composition.

        # Apply the rotation.
        beam_direction = np.array([0, 0, 1])
        beam_direction = total_rot.apply(beam_direction)

        # Return the (azimuth, polar) angles.
        return vector_to_azimuth_polar(beam_direction)

    def _i10_detector_polar(self):
        """
        Parses self.metadata.metadata_file to calculate our detector's polar
        angle; assumes that the data was recorded at beamline I10 in the RASOR
        diffractometer.
        """
        return self._i10_detector_angles[1]

    def _i10_detector_azimuth(self):
        """
        Parses self.metadata.metadata_file to calculate our detector's azimuthal
        angle; assumes that the data was recorded at beamline I10 in the RASOR
        diffractometer.
        """
        return self._i10_detector_angles[0]
