"""
This module contains an implementation of

diffraction_utils.diffractometer_base.DiffractometerBase

for the diffractometer in I07's experimental hutch 1.
"""

import numpy as np
from scipy.constants import Planck, speed_of_light, elementary_charge
from scipy.spatial.transform import Rotation

from ..diffractometer_base import DiffractometerBase
from ..frame_of_reference import Frame
from ..io import I07Nexus
from ..vector import Vector3


INSB_LATTICE_PARAMETER = 6.479  # In Å.


def _energy_to_wavelength(energy_in_ev):
    """
    Converts the incident beam energy into wavelength in Å.

    Args:
        Energy of the probe particle in electron volts.
    """
    return Planck*speed_of_light / (energy_in_ev*elementary_charge) * 1e10


class I07Diffractometer(DiffractometerBase):
    """
    Implementation of DiffractometerBase for the diffractometer in Diamond's I07
    beamline's experimental hutch 1.

    Args:
        data_file (I10Nexus):
            An instance of diffraction_utils.io.I10Nexus corresponding to the
            nexus file that contains the diffractometer description.
        sample_oop (np.ndarray-like):
            A [h, k, l] array describing the sample's OOP vector in hkl-space.
    """

    horizontal = "horizontal"
    vertical = "vertical"
    dcd = "DCD"

    def __init__(self, data_file: I07Nexus, sample_oop: np.ndarray,
                 setup: str = "horizontal") -> None:
        super().__init__(data_file, sample_oop)
        self.setup = setup

    def get_u_matrix(self, scan_index: int) -> Rotation:
        # The following are the axis in the lab frame when all motors are @0.
        # Note that omega is like theta but for the vertical axis (I think!)
        alpha_axis = np.array([0, 1, 0])
        omega_axis = np.array([-1, 0, 0])

        alpha = self.data_file.alpha[scan_index]
        omega = self.data_file.omega[scan_index]

        # Create the rotation objects.
        alpha_rot = Rotation.from_rotvec(alpha_axis*alpha, degrees=True)
        omega_rot = Rotation.from_rotvec(omega_axis*omega, degrees=True)

        if self.setup != self.vertical:
            # These axes aren't always parsed in vertical geometries.
            theta_axis = np.array([0, 1, 0])
            chi_axis = np.array([1, 0, 0])
            theta = self.data_file.theta[scan_index]
            chi = self.data_file.chi[scan_index]   
            if self.setup==self.dcd:
                theta=0
                chi=0
                alpha_rot = Rotation.from_rotvec(alpha_axis*0, degrees=True)

            theta_rot = Rotation.from_rotvec(theta_axis*theta, degrees=True)
            chi_rot = Rotation.from_rotvec(chi_axis*chi, degrees=True)

            # Alpha acts after chi, which acts after theta. So, apply rotations
            # in the correct order to get the U matrix:
            return alpha_rot*chi_rot*theta_rot

        # If execution reaches here, we're in vertical scattering and we need
        # alpha and omega.
        return alpha_rot*omega_rot

    def get_detector_vector(self, frame: Frame) -> Vector3:
        # The following are the axis in the lab frame when all motors are @0.
        gamma_axis = np.array([0, 1, 0])
        delta_axis = np.array([-1, 0, 0])

        gamma = self.data_file.gamma[frame.scan_index]
        delta = self.data_file.delta[frame.scan_index]
        #add in correction so that dcd is always along z - this assumes there is never the need for a single crystal scanning when using the dcd
        if self.setup==self.dcd:
            lab_beam_vector = self._dcd_incident_beam_lab
            lab_beam_arr=lab_beam_vector.array
            tan_inc_angle=lab_beam_arr[0]/lab_beam_arr[2]
            inc_hor_angle=np.degrees(np.arctan(tan_inc_angle))
            gamma-=inc_hor_angle
            

        # Create the rotation objects.
        gamma_rot = Rotation.from_rotvec(gamma_axis*gamma, degrees=True)
        delta_rot = Rotation.from_rotvec(delta_axis*delta, degrees=True)

        # Combine them (gamma acts after delta).
        total_rot = gamma_rot * delta_rot

        # Act this rotation on the [0, 0, 1], which is the vector pointing
        # to the detector when gamma, delta = 0, 0.
        to_detector = np.array([0, 0, 1])
        detector_vec = Vector3(total_rot.apply(to_detector),
                               Frame(Frame.lab, self, frame.scan_index))
        # Finally, rotate this vector into the frame that we need it in.
        self.rotate_vector_to_frame(detector_vec, frame)
        return detector_vec
    

    def get_incident_beam(self, frame: Frame) -> Vector3:
        if self.setup != self.dcd:
            return super().get_incident_beam(frame)

        # For the DCD we need to put a bit more effort into calculating the
        # position of the incident beam.
        lab_beam_vector = self._dcd_incident_beam_lab
        #rotate vector so that direction is along h in reciprocal space (this assumes dcd ignores single crystal samples)
        lab_beam_arr=lab_beam_vector.array
        tan_inc_angle=lab_beam_arr[0]/lab_beam_arr[2]
        inc_hor_angle=np.degrees(np.arctan(tan_inc_angle))
        rot_angle=-inc_hor_angle
        rotation_axis= np.array([0, 1, 0])
        dcd_rot=Rotation.from_rotvec(rotation_axis*rot_angle, degrees=True)
        lab_beam_vector.array=dcd_rot.apply(lab_beam_arr)

        self.rotate_vector_to_frame(lab_beam_vector, frame)
        return lab_beam_vector

    @property
    def _dcd_incident_beam_lab(self) -> Vector3:
        """
        Returns a unit vector instance of vector3 that points from the 2nd DCD
        crystal to the sample.
        """
        # First get the displacement between the beam from the synchrotron and
        # the 2nd crystal in the DCD setup. Note that we need to convert to rad.
        omega = self.data_file.dcd_omega * np.pi/180
        beam_crystal_vector = np.array([np.cos(omega), np.sin(omega), 0])
        beam_crystal_vector *= self.data_file.dcd_circle_radius

        # But, the beam is travelling *from* the crystal *to* the sample, so
        # we actually need the -ve of both of these values.
        beam_crystal_vector = -beam_crystal_vector

        # Then simply add the displacement along the z-direction to the sample.
        beam_crystal_vector += np.array(
            [0, 0, self._dcd_sample_distance])

        # We're expecting a unit vector.
        beam_crystal_vector /= np.linalg.norm(beam_crystal_vector)

        # Now make an appropriate Vector3 object.
        return Vector3(beam_crystal_vector, Frame(Frame.lab, self))

    @property
    def _dcd_sample_distance(self):
        """
        The distance between the 2nd crystal in the DCD and the sample surface.
        This is strictly along the direction of the beam as it leaves the
        synchtrotron.
        """
        return self.data_file.dcd_circle_radius/np.tan(self._dcd_cone_angle_rad)

    @property
    def _dcd_cone_angle(self):
        """
        Imagine the path that light takes from the DCD to the sample for a
        particular value of dcd_omega. It's a line that goes from the 2nd DCD
        crystal to the sample. Now, imagine the volume of revolution of this
        line (i.e. the locus of all lines for all values of dcd_omega). It is a
        cone, with a certain cone angle. This method returns this cone angle,
        which turns out to be simply the difference between two Bragg angles.
        """
        return (self._insb_220_theta - self._insb_111_theta)*2

    @property
    def _dcd_cone_angle_rad(self):
        """
        As above, but in radians.
        """
        return self._dcd_cone_angle * np.pi/180

    @property
    def _insb_111_theta(self):
        """
        Returns the scattering theta of the InSb (111) reflection. Needed to
        calculate the incident beam orientation in the DCD setup.
        """
        wavelength = _energy_to_wavelength(self.data_file.probe_energy)
        d_111 = INSB_LATTICE_PARAMETER/np.sqrt(3)
        return np.arcsin(wavelength/(2*d_111))*180/np.pi

    @property
    def _insb_220_theta(self):
        """
        Returns the scattering theta of the InSb (220) reflection. Needed to
        calculate the incident beam orientation in the DCD setup.
        """
        wavelength = _energy_to_wavelength(self.data_file.probe_energy)
        d_220 = INSB_LATTICE_PARAMETER/np.sqrt(8)
        return np.arcsin(wavelength/(2*d_220))*180/np.pi
