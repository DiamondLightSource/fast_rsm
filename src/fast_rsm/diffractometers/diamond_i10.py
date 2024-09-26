"""
This module contains an implementation of

diffraction_utils.diffractometer_base.DiffractometerBase

for the RASOR diffractometer at Diamond Light Source's beamline I10.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from ..diffractometer_base import DiffractometerBase
from ..frame_of_reference import Frame
from ..io import I10Nexus
from ..vector import Vector3


class I10RasorDiffractometer(DiffractometerBase):
    """
    Implementation of DiffractometerBase for Diamond's RASOR diffractometer on
    beamline I10.

    Args:
        data_file (I10Nexus):
            An instance of diffraction_utils.io.I10Nexus corresponding to the
            nexus file that contains the diffractometer description.
        sample_oop (np.ndarray-like):
            A [h, k, l] array describing the sample's OOP vector in hkl-space.
        detector (str):
            Which detector are we using? Can be "point" or "area".
    """

    point_detector = "point"
    area_detector = "area"

    def __init__(self, data_file: I10Nexus, sample_oop: np.ndarray,
                 detector: str) -> None:
        super().__init__(data_file, sample_oop)
        self.detector = detector

    def get_u_matrix(self, scan_index: int) -> Rotation:
        # The following are the axes in the lab frame when all motors are @0.
        theta_axis = np.array([1, 0, 0])
        chi_axis = np.array([0, 0, 1])

        # We need to work out what our current chi, theta and 2theta values are.
        if self.detector == I10RasorDiffractometer.area_detector:
            theta = self.data_file.theta_area[scan_index]
        elif self.detector == I10RasorDiffractometer.point_detector:
            theta = self.data_file.theta[scan_index]
        else:
            raise ValueError("Detector must be either 'point' or 'area'.")
        chi = self.data_file.chi[scan_index]

        # Create the rotation objects.
        theta_rot = Rotation.from_rotvec(theta_axis*theta, degrees=True)
        chi_rot = Rotation.from_rotvec(chi_axis*chi, degrees=True)

        # Chi acts after theta (i.e. theta is attached to chi), so
        # chi_rot*theta_rot would generate a rotation that would map from the
        # lab frame to the sample holder frame. This is precisely the inverse
        # of the U matrix!
        return (chi_rot*theta_rot).inv()

    def get_detector_vector(self, frame: Frame) -> Vector3:
        # First we need a vector pointing at the detector in the lab frame.
        two_theta_axis = np.array([-1, 0, 0])
        if self.detector == I10RasorDiffractometer.area_detector:
            two_theta = self.data_file.two_theta_area[frame.scan_index]
        if self.detector == I10RasorDiffractometer.point_detector:
            two_theta = self.data_file.two_theta[frame.scan_index]

        # Create the rotation object.
        two_theta_rot = Rotation.from_rotvec(two_theta_axis*two_theta,
                                             degrees=True)

        # Act this rotation on the beam.
        beam_direction = np.array([0, 0, 1])
        detector_vec = Vector3(two_theta_rot.apply(beam_direction),
                               Frame(Frame.lab, self, frame.scan_index))

        # Finally, rotate this vector into the frame that we need it in.
        self.rotate_vector_to_frame(detector_vec, frame)
        return detector_vec
