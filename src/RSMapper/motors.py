"""
This module contains a convenience class for tracking motor positions.
"""

# Because of the dumb way that nexusformat works.
# pylint: disable=protected-access

from typing import Tuple

from .metadata import Metadata


class Motors:
    """
    Can calculate relative detector/sample orientation from motor positions.
    """

    def __init__(self, metadata: Metadata) -> None:
        self.metadata = metadata

    @property
    def detector_theta(self) -> float:
        """
        Returns the detector's spherical polar theta value.
        """
        # Call the appropriate function for the instrument in use.
        return getattr(self, f"_theta_from_{self.metadata.instrument}")()

    @property
    def detector_phi(self) -> float:
        """
        Returns the detector's spherical polar phi value.
        """
        return getattr(self, f"_phi_from_{self.metadata.instrument}")()

    @property
    def _i07_phi_theta(self) -> Tuple[float, float]:
        """
        Returns (phi, theta) assuming that the metadata file is an I07 file.
        """
        alpha = self.metadata.metadata_file[
            "/entry/instrument/diff1alpha/value"]
        gamma = self.metadata.metadata_file[
            "/entry/instrument/diff1gamma/value"]._value
        delta = self.metadata.metadata_file[
            "/entry/instrument/diff1delta/value"]._value
        chi = self.metadata.metadata_file[
            "/entry/instrument/diff1chi/value"]
        omega = self.metadata.metadata_file[
            "/entry/instrument/diff1omega/value"]
        theta = self.metadata.metadata_file[
            "/entry/instrument/diff1theta/value"]

        # ...maths goes here...

        return delta, gamma

    def _theta_from_i07(self) -> float:
        """
        Parses self.metadata.metadata_file to calculate our current theta;
        assumes that the data was recorded at beamline I07 at Diamond.
        """
        return self._i07_phi_theta[1]

    def _phi_from_i07(self) -> float:
        """
        Parses self.metadata.metadata_file to calculate our current phi; assumes
        that the data was acquired at Diamond's beamline I07.
        """
        return self._i07_phi_theta[0]
