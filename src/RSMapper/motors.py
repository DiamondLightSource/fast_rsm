"""
This module contains a convenience class for tracking motor positions.
"""


from .metadata import Metadata


class Motors:
    """
    Can calculate relative detector/sample orientation from motor positions.
    """

    def __init__(self, metadata: Metadata) -> None:
        self.metadata = metadata

    @property
    def detector_theta(self):
        """
        Returns the detector's spherical polar theta value.
        """
        # Call the appropriate function for the instrument in use.
        return getattr(self, f"_theta_from_{self.metadata.instrument}")()

    @property
    def detector_phi(self):
        """
        Returns the detector's spherical polar phi value.
        """
        return getattr(self, f"_phi_from_{self.metadata.instrument}")()

    def _theta_from_i07(self):
        """
        Parses self.metadata.metadata_file to calculate our current theta;
        assumes that the data was recorded at beamline I07 at Diamond.
        """

    def _phi_from_i07(self):
        """
        Parses self.metadata.metadata_file to calculate our current phi; assumes
        that the data was acquired at Diamond's beamline I07.
        """
