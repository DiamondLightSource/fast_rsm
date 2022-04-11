"""
This module contains a convenience class for tracking motor positions.
"""

# Because of the dumb way that nexusformat works.
# pylint: disable=protected-access

from typing import Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation

from .metadata import Metadata


def vector_to_azimuth_polar(vector: np.ndarray) -> Tuple[float]:
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
    polar = np.arccos(vector[1])
    azimuth = np.arctan2(vector[0], vector[2])

    return azimuth, polar


def azimuth_polar_to_vector(azimuth: float, polar: float) -> np.ndarray:
    """
    Converts two spherical polar angles to a unit vector pointing in that
    direction.

    Args:
        azimuth:
            The azimuthal anlge (measured from the z-axis).
        polar:
            The polar angle (measured from the y-axis).

    Returns:
        numpy array of length 1.
    """
    return np.array([
        np.sin(polar)*np.sin(azimuth),
        np.cos(polar),
        np.sin(polar)*np.cos(azimuth)
    ])


class Motors:
    """
    Instances of this class have knowledge of metadata acquired during this
    experiment. They use this metadata to calculate the angular position of the
    detector, and a unit vector parallel to the incident light, *IN THE FRAME OF
    REFERENCE OF THE SAMPLE*. In a scattering experiment, all other frames of
    reference are meaningless.

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
    def sample_polar(self) -> float:
        """
        Returns the spherical polar polar angle in a coordinate system anchored
        to the sample.
        """
        return getattr(self, f"_{self.metadata.instrument}_sample_polar")()

    @property
    def sample_azimuth(self) -> float:
        """
        Returns the spherical polar azimuthal angle in a coordinate system
        anchored to the sample.
        """
        return getattr(self, f"_{self.metadata.instrument}_sample_azimuth")()

    @property
    def incident_beam(self) -> np.ndarray:
        """
        Returns a unit vector parallel to the incident beam in the sample's
        frame of reference.
        """
        return getattr(self, f"_{self.metadata.instrument}_incident_beam")

    @property
    def _i10_incident_beam(self):
        """
        Returns a unit vector pointing parallel to the incident beam. Assumes
        the experiment took place in I10 on RASOR.
        """
        return self.sample_rotation.inv().apply([0, 0, 1])

    @property
    def _i10_sample_rotation(self) -> Rotation:
        """
        Samples can only be affected by theta in RASOR; chi only tilts the
        camera.
        """
        theta = self.array_to_correct_element(180-self.metadata.metadata_file[
            "/entry/instrument/th/value"]._value)
        # Prepare rotation matrices.
        return Rotation.from_rotvec([-theta, 0, 0], degrees=True)

    @property
    def _i10_detector_angles_lab_frame(self):
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
        tth_rot = Rotation.from_rotvec([-tth_area, 0, 0], degrees=True)
        chi_rot = Rotation.from_rotvec([0, 0, chi], degrees=True)
        total_rot = chi_rot * tth_rot  # This does a proper composition.

        # Apply the rotation.
        beam_direction = np.array([0, 0, 1])
        beam_direction = total_rot.apply(beam_direction)

        # Return the (azimuth, polar) angles in radians.
        return vector_to_azimuth_polar(beam_direction)

    @property
    def _i10_detector_angles_sample_frame(self):
        """
        Returns the detectors azimuthal and polar angles, as measured from the
        frame of reference tied to the sample.
        """
        # Grab the angles in the lab frame; turn them into a vector.
        azi_lab, polar_lab = self._i10_detector_angles_lab_frame
        q_out_lab = azimuth_polar_to_vector(azi_lab, polar_lab)

        # Rotate this vector into the sample frame; convert it to angles.
        q_out_sample = self._i10_sample_rotation.inv().apply(q_out_lab)
        return vector_to_azimuth_polar(q_out_sample)

    def _i10_detector_polar(self):
        """
        Parses self.metadata.metadata_file to calculate our detector's polar
        angle; assumes that the data was recorded at beamline I10 in the RASOR
        diffractometer.
        """
        return self._i10_detector_angles_sample_frame[1]

    def _i10_detector_azimuth(self):
        """
        Parses self.metadata.metadata_file to calculate our detector's azimuthal
        angle; assumes that the data was recorded at beamline I10 in the RASOR
        diffractometer.
        """
        return self._i10_detector_angles_sample_frame[0]

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
